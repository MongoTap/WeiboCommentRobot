import json
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('.')

from dataset.preprocess import data_clean

# 训练之前需要预处理微博文本，path = 'dataset/train_data.csv'
def preprocess_weibo(path, source_text, target_text):
    df = pd.read_csv(path, sep='\t')
    text = df[source_text].tolist()
    response = df[target_text].tolist()

    preprocess_weibo = []
    preprocess_resp = []
    data = list(zip(text, response))
    for weibo, resp in tqdm(data):
        pre_weibo, _ = data_clean(weibo, max_len=512, is_control_length=True, is_format=False)
        pre_resp, _ = data_clean(resp, max_len=512, is_control_length=True, is_format=False)
        preprocess_weibo.append('评论以下微博：_' + pre_weibo)
        preprocess_resp.append(pre_resp)

    res = pd.DataFrame({'input': preprocess_weibo, 'target': preprocess_resp})
    res.to_csv('dataset/weibo.csv', index=False, sep='\t')
    print('finish the weibo preprocess, save path: dataset/weibo.csv')
    return res
