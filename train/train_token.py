# import gradio as gr
# from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
# import argparse
# from threading import Thread
# from gradio_highlightedtextbox import HighlightedTextbox
# import torch
# import numpy as np

# from ..webui.utils import *
import json
import sys
sys.path.append('./')
from transformers import AutoTokenizer
import TACS.tfqa.model.llama as llama
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
# 创建一个空列表，用于存储所有转换后的字典

# 打开.jsonl文件

def load_dataset():
    dict_list=[]
    with open('/mnt/workspace/TACS/train/TruthfulQA.jsonl', 'r', encoding='utf-8') as file:
        # 逐行读取文件
        for line in file:
            # 将每行字符串转换为字典
            dict_obj = json.loads(line)
            # 将转换后的字典添加到列表中
            dict_list.append(dict_obj)
            

    data1=[]
    for data in dict_list:
        
        X=data['single_evidence_true_false']
        y=data['single_evidence_true_false_label']
        data1.append((X,int(y)))
        X=data['single_evidence_false_true']
        y=data['single_evidence_false_true_label']
        data1.append((X,(y)))
        
    return data1

def init_SVM():
    init_svm=[]
    for i in range(32):
        init_svm.append(svm.SVC(kernel='linear', C=1.0))

    return init_svm

def init_dataForSVM(data):
    R= torch.zeros(32,len(data),4096)
    Y= torch.zeros(32,len(data))
    # for i in range(32):
    #     temp1=[]
    #     temp2=[]
    #     R.append(temp1)
    #     Y.append(temp2)

    model_path = '/mnt/workspace/TACS/tfqa/data/models--meta-llama--Llama-2-7b-chat-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    tokenizer.pad_token_id = 0 
    tokenizer.padding_side = 'left'
    model = llama.LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    for index,item in enumerate(data):
        print(index)
        inputs = item[0]
        y = item[1]
        #print(inputs)
        #print(type(input))
        encodings = tokenizer([inputs], return_tensors='pt', padding=True)
        encodings=encodings.to('cuda:0')
        tokenized = [tokenizer.convert_ids_to_tokens(i) for i in encodings.input_ids]
        start_id=0
        end_id=0
        info_begin=0
        i=0
        for j in range(len(tokenized[i])):
                if tokenized[i][j] == 'Information':
                    if tokenized[i][j+1] == ':':
                        start_id=j+1
                        info_begin=1
                elif info_begin and tokenized[i][j]=='.':
                    end_id=j
                    break
        with torch.no_grad():            
            output = model(**encodings)
        


        for layer in range(32):
            attn_score = model.model.layers[layer].self_attn.attn_score
            #attn_score的维度为(1,句子长度，4096)
            head_score = attn_score[0,start_id:end_id]
            #head_score的维度为(句子长度,4096)
            #随机取一个数
            ran_i=random.randrange(end_id-start_id)
            head_score = head_score[ran_i]
            #head_score是一个(4096,)的array
            R[layer][index]=torch.from_numpy(head_score)
            Y[layer][index]=torch.tensor(y)

    return R,Y

if __name__ == '__main__':
    #本次训练每个样本只有一条infomation
    data=load_dataset()
    #初始化向量机
    svmlist=init_SVM()
    R,Y=init_dataForSVM(data)
    

    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for i in range(32):
        temp1=[]
        temp2=[]
        temp3=[]
        temp4=[]
        test_data.append(temp1)
        train_data.append(temp2)
        train_label.append(temp3)
        test_label.append(temp4)


    for layer in range(32):
        X_train, X_test, y_train, y_test = train_test_split(R[layer], Y[layer], test_size=0.2, random_state=42)
        test_data[layer].append(X_test)
        test_label[layer].append(y_test)
        train_data[layer].append(X_train)
        train_label[layer].append(y_train)
    

    acc=[]
    for layer in range(32):
        print('layer:',layer)
        svmlist[layer].fit(train_data[layer][0], train_label[layer][0])
        y_pred = svmlist[layer].predict(test_data[layer][0])
        acc.append(accuracy_score(test_label[layer][0], y_pred))
        print('Accuracy:', acc[layer])

    torch.save(svmlist, '/mnt/workspace/TACS/train/svm_token.pt')
    torch.save(acc,'/mnt/workspace/TACS/train/svm_token_acc.pt')


