import gradio as gr
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import argparse
from threading import Thread
from gradio_highlightedtextbox import HighlightedTextbox
from fastchat.conversation import get_conv_template
import torch
import numpy as np
import sys
sys.path.append("/mnt/workspace/TACS/tfqa")
sys.path.append('./')
sys.path.append("/mnt/workspace/TACS/train")

#from model.TACS import TACS_model
from TACS import TACS_model
from utils import *
import os
os.environ['HF_DATASETS_OFFILINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

model_name='/mnt/workspace/TACS/tfqa/data/models--meta-llama--Llama-2-7b-chat-hf'
#token_svm_path='/mnt/workspace/TACS/webui/svm/svm_single_evidence_Llama-2-7b-chat-hf_fold2.pt'
#token_svm_acc='/mnt/workspace/TACS/webui/svm/acc_single_evidence_Llama-2-7b-chat-hf_fold2.pt'
sentence_svm_path='/mnt/workspace/TACS/train/svm_sentence.pt'
sentence_svm_acc='/mnt/workspace/TACS/train/svm_sentence_acc.pt'
TACS_mode='DEMO_sentence' 
svm_num=5

token_svm_path='/mnt/workspace/TACS/train/svm_token.pt'
token_svm_acc='/mnt/workspace/TACS/train/svm_token_acc.pt'

TACS_model_1 = TACS_model(model_path=model_name, TACS_mode=None)


TACS_model_2 = TACS_model(model_path=model_name, TACS_mode=TACS_mode)
TACS_model_2.svm = torch.load(token_svm_path)
TACS_model_2.acc = torch.load(token_svm_acc)
TACS_model_2.sorted_indices = np.argsort(TACS_model_2.acc)[-svm_num:]
TACS_model_2.layer_indices = TACS_model_2.sorted_indices


html_output = """
    <div style="font-weight: bold; font-size: 20px">
        Demo: Truth-Aware Context Selection: Mitigating the Hallucinations of Large Language Models Being Misled by Untruthful Contexts
    </div>
    <div style="font-weight: bold; font-size: 16px">
        Authors: Tian Yu, Shaolei Zhang, and Yang Feng
    </div>
"""
def change_mode(drop):
    TACS_model_2.TACS_mode = drop
    if 'sentence' in drop:
        # TACS_model_2.svm = torch.load(args.sentence_svm_path)
        # TACS_model_2.acc = torch.load(args.sentence_svm_acc)
        # TACS_model_2.sorted_indices = np.argsort(TACS_model_2.acc)[-args.svm_num:]
        # TACS_model_2.layer_indices = TACS_model_2.sorted_indices.numpy()
        TACS_model_2.svm = torch.load(sentence_svm_path)
        TACS_model_2.acc = torch.load(sentence_svm_acc)
        TACS_model_2.sorted_indices = np.argsort(TACS_model_2.acc)[-svm_num:]
        TACS_model_2.layer_indices = TACS_model_2.sorted_indices.numpy()
    else:
        # TACS_model_2.svm = torch.load(args.token_svm_path)
        # TACS_model_2.acc = torch.load(args.token_svm_acc)
        # TACS_model_2.sorted_indices = np.argsort(TACS_model_2.acc)[-args.svm_num:]
        # TACS_model_2.layer_indices = TACS_model_2.sorted_indices.numpy()
        TACS_model_2.svm = torch.load(token_svm_path)
        TACS_model_2.acc = torch.load(token_svm_acc)
        TACS_model_2.sorted_indices = np.argsort(TACS_model_2.acc)[-svm_num:]
        TACS_model_2.layer_indices = TACS_model_2.sorted_indices.numpy()

def change_threshold(slider):
    TACS_model_2.threshold = slider

def truth_detection(question, information):
    inputs = format_prompt(question, information)
    encodings = TACS_model_2.tokenizer([inputs], return_tensors='pt', padding=True)
    tokenized = [TACS_model_2.tokenizer.convert_ids_to_tokens(i) for i in encodings.input_ids]

    start_id = tokenized[0].index('Information')+2
    end_id = tokenized[0].index('Answer')-2

    encodings = TACS_model_2.truth_detection(encodings)
   
    decoded_text = ""
    for i in range(len(tokenized[0])):
        if i < 31:
            continue
        if i < start_id:
            decoded_text += tokenized[0][i]
        elif i < end_id:
            if encodings['attention_mask'][0][i] == 1:
                decoded_text += '<a>'+tokenized[0][i]+'</a>'
            else:
                decoded_text += '<b>'+tokenized[0][i]+'</b>'
        else:
            break
    decoded_text = decoded_text.replace('▁', ' ')
    decoded_text = decoded_text.replace('<0x0A>', '\n')
    
    return convert_tagged_text_to_highlighted_text(decoded_text)

def generation_without_TACS(question, information):
    inputs = format_prompt(question, information)
    
    
    encodings = TACS_model_1.tokenizer([inputs], return_tensors='pt', padding=True)
    encodings=encodings.to('cuda:0')
    output=TACS_model_1.model.generate(encodings.input_ids)
    print(TACS_model_1.tokenizer.decode(output[0],skip_special_tokens=False))
    # streamer = TextIteratorStreamer(TACS_model_1.tokenizer, timeout=10, skip_special_tokens=True, skip_prompt=True)
    # generate_kwargs = dict(
    #     encodings,
    #     streamer=streamer,
    #     max_new_tokens=512,
    #     do_sample=False,
    #     top_p=1,
    #     temperature=1.0,
    #     num_beams=1,
        
    # )
    
    # #print("generate_kwargs:",generate_kwargs)
    # t=Thread(target=TACS_model_1.model.generate, kwargs=generate_kwargs)
    # t.start()
    
    # partial_message = ""
    # for new_token in streamer:
    #     partial_message += new_token
    #     yield partial_message

    


def generation_with_TACS(question, information):
    inputs = format_prompt(question, information)
    encodings = TACS_model_2.tokenizer([inputs],return_tensors='pt', padding=True)
    encodings = TACS_model_2.truth_detection(encodings)
    encodings=encodings.to('cuda:0')

    output=TACS_model_2.model.generate(encodings.input_ids,attention_mask=encodings.attention_mask)
    print(TACS_model_2.tokenizer.decode(output[0],skip_special_tokens=False))

    # streamer = TextIteratorStreamer(TACS_model_2.tokenizer, timeout=10, skip_special_tokens=True, skip_prompt=True)
    # generate_kwargs = dict(
    #     encodings,
    #     streamer=streamer,
    #     max_new_tokens=512,
    #     do_sample=False,
    #     top_p=1,
    #     temperature=1.0,
    #     num_beams=1,
    # )
    
    # t=Thread(target=TACS_model_2.model.generate, kwargs=generate_kwargs)
    # t.start()
    
    # partial_message = ""
    # for new_token in streamer:
    #     partial_message += new_token
    #     yield partial_message



# question='Is a pen or a sword more useful in a physical fight?'
# information='A pen is mightier than a sword in a physical fight.A sword is mightier than a pen in a physical fight.'
# truth_detection(question=question,information=information)

def main():
    print("Welcone to use TASC!!!")
    # while True:
    #     question=input('\nPlease enter the question. Question:')
    #     information=input('\nPlease enter the information. Information:')
    #     print('Llama:')
    #     llama=generation_without_TACS(question=question,information=information)
    #     print(llama)
    #     print('llama:')
    #     TACS=generation_with_TACS(question=question,information=information)
    #     print('TACS')

    question='What are the benefits of eating an apple a day?'
    information=' Apples protect against diabetes.Apples stop you from getting sick.'
    # print('TACS_model_1.model.device',TACS_model_1.model.device)
    # print('TACS_model_2.model.device',TACS_model_2.model.device)
    # print('next(TACS_model_1.model.parameters).device:')
    # print(next(TACS_model_1.model.parameters()).device)
    generation_with_TACS(question=question,information=information)
    #print('llama:')
    # for token in llama:
    #     print(token)

    #generation_with_TACS(question=question,information=information)
    # for token in TACS:
    #     print(token)

if __name__=="__main__":
    main()



