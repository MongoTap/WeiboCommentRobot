import time
import torch
import random
import pandas as pd
import re
import numpy as np
# 加载训练后的模型
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from dataset.preprocess import data_clean
# from bert_score import score

tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForSeq2SeqLM.from_pretrained("./model") 

device = torch.device('cuda:0') # cuda
model.to(device)

# 将模型输出的字符‘_’转换为换行符（模型定义如此）
def postprocess(text):
  return text.replace("_", "\n")

# 以下三个函数均为对回复内容的限制，不允许回复内容只有语气标点
def only_include(text):
    return bool(re.match('^[!！。？?]+$', text))

# 不允许回复内容太长和太短
def length_check(text):
    return len(text) > 32 or len(text) < 4

# 不允许回复出现无关内容，
def content_check(text):
    return '转发微博' in text

# 预测函数，即模型输出。给定一个微博文本，返回回复内容
def answer_fn(text, sample=True, top_p=0.8):
    '''
    sample: 是否抽样。生成任务, 可以设置为True;
    top_p: 0-1之间, 生成的内容越多样.
    '''
    origin = text
    # text = preprocess(text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device)
    if not sample: # 不进行采样
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128, num_beams=4, length_penalty=0.6)
    else: # 采样（生成）
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128, do_sample=True, top_p=top_p)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    res = postprocess(out_text[0])
    # res = out_text[0]
    return  answer_fn(origin, sample=True, top_p=top_p) if content_check(res) or length_check(res) or only_include(res) else res

# 预测前，需预处理输入的微博文本
def process_test_weibo(path):
    df = pd.read_csv(path, sep='\t')
    text = df['text'].tolist()
    weibo_id = df['weibo_id'].tolist()

    preprocess_weibo = []
    for weibo in tqdm(text):
        pre_weibo, _ = data_clean(weibo, max_len=512, is_control_length=True, is_format=False)
        preprocess_weibo.append('评论以下微博：_' + pre_weibo)

    return weibo_id, preprocess_weibo

# 预测入口，给定测试文件路径输出预测结果
def predict(path, output):
    weibo_id, weibo = process_test_weibo(path)

    pred_list = []
    # 生成回复时尽量避免雷同（与前两条回复需不同）
    tmp_txt_up1 = ''
    tmp_txt_up2 = ''
    for text in tqdm(weibo):
        for retry_i in range(20):
            result = answer_fn(text, sample=True, top_p=0.8)
            if result != tmp_txt_up1 and result != tmp_txt_up2:
                break
        tmp_txt_up2 = tmp_txt_up1
        tmp_txt_up1 = result
        pred_list.append(result)

    pd.DataFrame({'weibo_id': weibo_id, 'comment': pred_list}).to_csv(output, sep='\t', index=False)


if __name__ == '__main__':
    # 数据输入的文件格式与text_A.csv保持一致即可
    # 输出为./result/submission.csv
    predict(path='dataset/test_A.csv', output='submission.csv')

# def generateN(weibo, N):
#     tmp_txt = ''
#     candidates = []
#     refs = []
#     for i in range(N):
#         for retry_i in range(20):
#             result = answer_fn(weibo, sample=True, top_p=0.8)
#             if result != tmp_txt:
#                 break
#         tmp_txt = result
#         candidates.append(result)
#         refs.append(weibo)
    
#     # candidates = list(set(candidates))
#     # length = len(candidates)

#     return candidates, refs

# def answer_better_by_bertscore(topK, weibo):
#     score_list = []
#     for i in range(15):
#         txt_answer = answer_fn(weibo, sample=True, top_p=0.95)
#         P, R, F1 = score([txt_answer], [weibo],  lang='zh', verbose=True)
#         score_list.append((F1, txt_answer))
    
#     res = sorted(score_list[:topK], key=lambda tup: tup[0])
#     return res


# total_cands = []
# total_refs = []
# all_tiku = list(set(raw_text_list))
# for weibo in tqdm(all_tiku):
#     cands, refs = generateN(weibo=weibo, N=10)
#     total_cands.extend(cands)
#     total_refs.extend(refs)

# P, R, F1 = score(total_cands, total_refs, lang='zh', verbose=True)

# F1 = F1.tolist()
# total = []
# for cand, f1 in zip(total_cands, F1):
#     total.append((f1, cand))

# print('开始排序')
# answer_repo = dict()
# for i in tqdm(np.arange(0, len(total_cands), 10)):
#     weibo = total_refs[i]
#     candidates = total[i:i+10]
#     candidates = sorted(candidates, key=lambda tup: tup[0])
#     answer_repo[weibo] = candidates[::-1]


# weibo_list = []
# resp_list = []
# up_weibo = ''
# count = 0
# i = 0
# for weibo_txt in tqdm(raw_text_list):
#     i += 1
#     weibo_list.append(weibo_txt)
#     if up_weibo == '':
#         up_weibo = weibo_txt
#         count += 1
#         continue
#     if weibo_txt != up_weibo:
#         candidates = answer_repo[up_weibo]
#         first = candidates[0]
#         other = random.sample(candidates[1:-1], count-1)
#         resp_list.append(first)
#         resp_list.extend(other)
#         up_weibo = weibo_txt
#         count = 1
#     else:
#         count += 1
#         if i == len(raw_text_list):
#             candidates = answer_repo[weibo_txt]
#             first = candidates[0]
#             other = random.sample(candidates[1:-1], count-1)
#             resp_list.append(first)
#             resp_list.extend(other)

# resp_drop_score = []
# for sco, resp in resp_list:
#     resp_drop_score.append(resp)