from dataset.process_traindata import preprocess_weibo
from model import Trainer

if __name__ == '__main__':
    # 模型参数设置
    model_params = {
        "MODEL": "./model",  # model path
        "TRAIN_BATCH_SIZE": 14,  # training batch size, 8
        "VALID_BATCH_SIZE": 14,  # validation batch size,8 
        "TRAIN_EPOCHS": 1,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 1e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text, 512
        "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text,64
        "SEED": 42,  # set seed for reproducibility
        "CUDA_DEVICE": 'cuda:0'
    }

    # 加载预处理数据
    # source_text: 微博在dataframe中的列名
    # target_text: 回复在dataframe中的列名
    df = preprocess_weibo('dataset/train_data.csv', source_text="text", target_text="comment")
    
    # 创建训练器
    trainer = Trainer(model_params=model_params, output_dir='./result')

    # 开始训练
    # source_text: 微博在dataframe中的列名
    # target_text: 回复在dataframe中的列名
    trainer.train(dataframe=df, source_text="input", target_text="target")
