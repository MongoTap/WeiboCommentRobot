import os,json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os,time, datetime
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
# Setting up the device for GPU usage
from torch import cuda
# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console


class WeiboDataset(Dataset):
    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        微博数据集的装载，为训练时dataloader提供Dataset类

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


class Trainer():
    def __init__(self, model_params, output_dir):
        # 训练类，包含训练方法和验证方法 

        # 参数样例
        # model_params = {
        #     "MODEL": "./model/",  # model_type
        #     "TRAIN_BATCH_SIZE": 14,  # training batch size, 8
        #     "VALID_BATCH_SIZE": 14,  # validation batch size,8 
        #     "TRAIN_EPOCHS": 1,  # number of training epochs
        #     "VAL_EPOCHS": 1,  # number of validation epochs
        #     "LEARNING_RATE": 1e-4,  # learning rate
        #     "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text, 512
        #     "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text,64
        #     "SEED": 42,  # set seed for reproducibility
        # }
        self.model_params = model_params
        self.output_dir = output_dir
        # define a rich console logger
        self.device = self.model_params['CUDA_DEVICE'] if cuda.is_available() else 'cpu'
        self.console = Console(record=True)

        self.training_logger = Table(
            Column("Epoch", justify="center"),
            Column("Steps", justify="center"),
            Column("Loss", justify="center"),
            title="Training Status",
            pad_edge=False,
            box=box.ASCII,
        )

    # 训练方法，self.train_fn()为self.train()工作
    def train_fn(self, epoch, tokenizer, model, device, loader, optimizer):
        model.train()
        time1=datetime.datetime.now()
        for _, data in enumerate(loader, 0):
            y = data["target_ids"].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous() 
            lm_labels = y[:, 1:].clone().detach() 
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )
            loss = outputs[0]
            # 每100步打印日志
            if _ % 100 == 0 and _!=0:
                time2=datetime.datetime.now()
                print(_,"epoch:"+str(epoch)+"-loss:"+str(loss)+"; total-train-time spent:"+str((time2 - time1).seconds)+'s')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def validate(self, epoch, tokenizer, model, device, loader,max_length):
        """
        验证方法：输入用于验证的数据，返回模型预测的结果和正确的标签
        """
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for _, data in enumerate(loader, 0):
                y = data['target_ids'].to(device, dtype = torch.long)
                ids = data['source_ids'].to(device, dtype = torch.long)
                mask = data['source_mask'].to(device, dtype = torch.long)

                generated_ids = model.generate(
                    input_ids = ids,
                    attention_mask = mask, 
                    max_length=max_length, 
                    num_beams=2,
                    repetition_penalty=2.5, 
                    length_penalty=1.0, 
                    early_stopping=True
                    )
                preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
                target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
                if _%1000==0:
                    self.console.print(f'Completed {_}')

                predictions.extend(preds)
                actuals.extend(target)
        return predictions, actuals

    def train(self, dataframe, source_text, target_text):
        torch.manual_seed(self.model_params["SEED"])
        np.random.seed(self.model_params["SEED"])
        torch.backends.cudnn.deterministic = True

        # logging
        self.console.log(f"""[Model]: Loading {self.model_params["MODEL"]}...\n""")

        # tokenzier for encoding the text
        tokenizer = T5Tokenizer.from_pretrained(self.model_params["MODEL"])

        model = T5ForConditionalGeneration.from_pretrained(self.model_params["MODEL"])
        model = model.to(self.device)

        # logging
        self.console.log(f"[Data]: Reading data...\n")

        dataframe = dataframe[[source_text, target_text]]

        train_size = 0.94
        train_dataset = dataframe.sample(frac=train_size, random_state=self.model_params["SEED"])
        val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
        train_dataset = train_dataset.reset_index(drop=True)
        
        # 打印数据集相关日志：数据量、训练步数
        self.console.print(f"FULL Dataset: {dataframe.shape}")
        self.console.print(f"TRAIN Dataset: {train_dataset.shape}")
        self.console.print(f"TEST Dataset: {val_dataset.shape}\n")
        total_train_steps=int((train_dataset.shape[0] * self.model_params["TRAIN_EPOCHS"])/self.model_params["TRAIN_BATCH_SIZE"])
        self.console.print(f"Total Train Steps: {total_train_steps}\n")

        # Creating the Training and Validation dataset for further creation of Dataloader
        training_set = WeiboDataset(
            train_dataset,
            tokenizer,
            self.model_params["MAX_SOURCE_TEXT_LENGTH"],
            self.model_params["MAX_TARGET_TEXT_LENGTH"],
            source_text,
            target_text,
        )
        val_set = WeiboDataset(
            val_dataset,
            tokenizer,
            self.model_params["MAX_SOURCE_TEXT_LENGTH"],
            self.model_params["MAX_TARGET_TEXT_LENGTH"],
            source_text,
            target_text,
        )

        train_params = {
            "batch_size": self.model_params["TRAIN_BATCH_SIZE"],
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,

        }

        val_params = {
            "batch_size": self.model_params["VALID_BATCH_SIZE"],
            "shuffle": False,
            "num_workers": 4,
            "pin_memory": True,
        }


        training_loader = DataLoader(training_set, **train_params)
        val_loader = DataLoader(val_set, **val_params)


        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=self.model_params["LEARNING_RATE"]
        )

        self.console.log(f"[Initiating Fine Tuning]...\n")

        for epoch in range(self.model_params["TRAIN_EPOCHS"]):
            # 1) 训练一个epoch
            self.train_fn(epoch, tokenizer, model, self.device, training_loader, optimizer)
            
            # 2) 保存该epoch的模型
            self.console.log(f"[Saving Model]...\n")
            path = os.path.join(self.output_dir, "model_files")
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)

            # # 3) 验证模型，暂时可注释掉
            # self.console.log(f"[Initiating Validation]...\n")
            # with torch.no_grad():
            #     #for epoch in range(model_params["VAL_EPOCHS"]):
            #     predictions, actuals = self.validate(epoch, tokenizer, model, self.device, val_loader,model_params["MAX_TARGET_TEXT_LENGTH"])
            #     final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
            #     final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

        self.console.save_text(os.path.join(self.output_dir, "logs.txt"))

        self.console.log(f"[Validation Completed.]\n")
        self.console.print(
            f"""[Model] Model saved @ {os.path.join(self.output_dir, "model_files")}\n"""
        )
        # self.console.print(
        #     f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
        # )
        self.console.print(f"""[Logs] Logs saved @ {os.path.join(self.output_dir,'logs.txt')}\n""")

