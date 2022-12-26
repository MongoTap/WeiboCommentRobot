import re
from harvesttext import HarvestText
from summary.summary import get_summary


ht = HarvestText()

def data_clean(weibo_text, max_len=500, is_format=True, is_control_length=False):
    '''微博文本预处理

    :param text: 微博文本
    :param max_len: 设置文本处理的最大长度需求，基于textrank实现
    :param is_format: 是否格式化文本（清除所有Tab和空格，并重新为每个字之间增加空格
    :param is_control_length: 是否控制文本长度，当为False时，max_len无效
    :return: 处理后的微博文本
    '''

    text = weibo_text
    # 清除微博@符
    text = ht.clean_text(text)
    # 删除【】括号
    text = re.sub(r'[【】]', '', text)
    # 删除##括号
    text = re.sub(r'#', '', text)
    # 删除()括号中内容
    text = re.sub(r'\(.*?\)', '', text)
    # 删除（）括号中内容
    text = re.sub(r'（.*?）', '', text)
    # 删除xxx的微博视频
    text = re.sub(r'网友：', '', text)
    # 删除‘网页链接’
    text = re.sub(r'网页链接', '', text)
    # 删除xxx的微博视频
    text = re.sub(r'的微博视频', '', text)

    # 控制文本长度
    if is_control_length:
        text = control_text_length(text, max_len=max_len)  

    # 是否格式化文本
    if is_format:
        # 删除space
        text = re.sub(r'\s', '', text)
        # 删除tab
        text = re.sub(r'\t', '', text)
        # token之间添加空格
        text = " ".join(re.findall(".{1}",text))

    # 统计长度
    length = len(text)
    
    return text, length

def control_text_length(text, max_len=500):
    if len(text) <= max_len:
        return text

    # 开始summarize，如果只有一句话,就暴力截断
    docs = ht.cut_sentences(text)
    if len(docs) == 1:
        return text[:max_len]
    
    # len_list = [len(item) for item in docs]

    res = get_summary(docs, maxlen=max_len, sorted_by_order=True)
    return ''.join(res)