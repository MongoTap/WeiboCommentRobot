import pandas as pd
import re

def check(file_path):

    # 检查文件格式
    if file_path != "submission.csv":
        return "请将文件名改为submission.csv"

    # 检查文件字段
    truth_labels = ['weibo_id', 'comment']
    data = pd.read_csv(file_path, sep='\t')
    print(data.shape[0])
    data_label = data.columns.to_list() 
    if len(truth_labels) != len(data_label):
        return "标签字段有误，请检查字段，注意csv文件每列需以'\t'分割"
    for label in data_label:
        if label not in truth_labels:
            return "标签字段有误，请检查字段，注意csv文件每列需以'\t'分割"

    # 检查文件行数是否满足需求
    if data.shape[0] != 8470:
        return "生成文本条数不够"

    # 检查文件是否存在空值

    if data[pd.isnull(data['comment'])==False].shape[0] != data.shape[0]:
        return "文件中存在空值"

    return "文件格式暂未发现问题，请上传文件"


if __name__ == "__main__":
    file_path = 'submission.csv'
    result = check(file_path)
    print(result)