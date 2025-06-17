# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import sys
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm, trange
import multiprocessing

sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.append('../../../python_parser')
sys.path.append('../../../../hashsmooth')
sys.path.append('../../../../randomsmooth')
sys.path.append('../../../../torchware')

from model import Model
from model import RandomSmooth4LLM, RandomDelSmooth4LLM, HashSmooth4LLM, binom_test

from randomsmooth.random_tran import RandomTransformer
from torchmalware.random_del_wrapper import RandomDeleter
from hashsmooth import EditLSHTransformerTorch
from hashsmooth.utils_hash import lower_confidence_interval, upper_confidence_interval

cpu_cont = 16
logger = logging.getLogger(__name__)

from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}
#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
    
#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
             input_tokens_1,
             input_ids_1,
             position_idx_1,
             dfg_to_code_1,
             dfg_to_dfg_1,
             input_tokens_2,
             input_ids_2,
             position_idx_2,
             dfg_to_code_2,
             dfg_to_dfg_2,
             label,
             url1,
             url2

    ):
        #The first code function
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1=position_idx_1
        self.dfg_to_code_1=dfg_to_code_1
        self.dfg_to_dfg_1=dfg_to_dfg_1
        
        #The second code function
        self.input_tokens_2 = input_tokens_2
        self.input_ids_2 = input_ids_2
        self.position_idx_2=position_idx_2
        self.dfg_to_code_2=dfg_to_code_2
        self.dfg_to_dfg_2=dfg_to_dfg_2
        
        #label
        self.label=label
        self.url1=url1
        self.url2=url2
        

def convert_examples_to_features(item):
    #source
    url1,url2,label,tokenizer, args,cache,url_to_code=item
    parser=parsers['java']
    
    for url in [url1,url2]:
        if url not in cache:
            func=url_to_code[url]
            
            #extract data flow
            code_tokens,dfg=extract_dataflow(func,parser,'java')
            code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
            ori2cur_pos={}
            ori2cur_pos[-1]=(0,0)
            for i in range(len(code_tokens)):
                ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
            code_tokens=[y for x in code_tokens for y in x]  
            
            #truncating
            code_tokens = code_tokens[:args.code_length - 2]
            source_tokens = [tokenizer.cls_token] + code_tokens
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
            if len(source_tokens) < args.code_length - 1:
                padding_length = args.code_length - len(source_tokens) - 1
                position_idx += [tokenizer.pad_token_id] * padding_length
                source_ids += [tokenizer.pad_token_id] * padding_length
            source_tokens += [tokenizer.sep_token]
            source_ids += [tokenizer.sep_token_id]
            position_idx += [tokenizer.pad_token_id]
            dfg = dfg[:args.code_length + args.data_flow_length - len(source_ids)]
            source_tokens += [x[0] for x in dfg]
            position_idx += [0 for x in dfg]
            source_ids += [tokenizer.unk_token_id for x in dfg]
            padding_length = args.code_length + args.data_flow_length - len(source_ids)
            position_idx += [tokenizer.pad_token_id] * padding_length
            source_ids += [tokenizer.pad_token_id] * padding_length
            
            #reindex
            reverse_index={}
            for idx,x in enumerate(dfg):
                reverse_index[x[1]]=idx
            for idx,x in enumerate(dfg):
                dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
            dfg_to_dfg=[x[-1] for x in dfg]
            dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
            length=len([tokenizer.cls_token])
            dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
            cache[url]=source_tokens,source_ids,position_idx,dfg_to_code,dfg_to_dfg

        
    source_tokens_1,source_ids_1,position_idx_1,dfg_to_code_1,dfg_to_dfg_1=cache[url1]   
    source_tokens_2,source_ids_2,position_idx_2,dfg_to_code_2,dfg_to_dfg_2=cache[url2]   
    return InputFeatures(source_tokens_1,source_ids_1,position_idx_1,dfg_to_code_1,dfg_to_dfg_1,
                   source_tokens_2,source_ids_2,position_idx_2,dfg_to_code_2,dfg_to_dfg_2,
                     label,url1,url2)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train'):
        postfix=file_path.split('/')[-1].split('.txt')[0]

        self.examples = []
        self.args=args
        index_filename=file_path
        
        #load index
        logger.info("Creating features from index file at %s ", index_filename)
        url_to_code={}

        folder = '/'.join(file_path.split('/')[:-1]) # 得到文件目录
        cache_file_path = os.path.join(folder, 'cached_{}'.format(postfix))
        # 保存下对应的code1和code2
        code_pairs_file_path = os.path.join(folder, 'cached_{}.pkl'.format(postfix))
        code_pairs = []
        try:
            self.examples = torch.load(cache_file_path)
            with open(code_pairs_file_path, 'rb') as f:
                code_pairs = pickle.load(f)
            logger.info("Loading features from cached file %s", cache_file_path)
            if args.do_certify:
                logger.info(
                    "=======*Using {} features from cached file {} for certification=======*".format(args.n_examples,
                                                                                                     cache_file_path))
                self.examples = self.examples[:args.n_examples]
                code_pairs = code_pairs[:args.n_examples]

        except:
            logger.info("Creating features from dataset file at %s", file_path)
            with open('/'.join(index_filename.split('/')[:-1])+'/data.jsonl') as f:
                for line in f:
                    line=line.strip()
                    js=json.loads(line)
                    url_to_code[js['idx']]=js['func']
                    
            #load code function according to index
            data=[]
            cache={}
            f=open(index_filename)
            with open(index_filename) as f:
                for line in f:
                    line=line.strip()
                    url1,url2,label=line.split('\t')
                    if url1 not in url_to_code or url2 not in url_to_code:
                        continue
                    if label=='0':
                        label=0
                    else:
                        label=1
                    data.append((url1,url2,label,tokenizer, args,cache,url_to_code))
                
            #only use 10% valid data to keep best model        
            # if 'valid' in file_path:
            #     data=random.sample(data,int(len(data)*0.1))
            for sing_example in data:
                code_pairs.append([sing_example[0], 
                                    sing_example[1], 
                                    url_to_code[sing_example[0]], 
                                    url_to_code[sing_example[1]]])
            with open(code_pairs_file_path, 'wb') as f:
                pickle.dump(code_pairs, f)
            #convert example to input features    
            self.examples=[convert_examples_to_features(x) for x in tqdm(data,total=len(data))]
            torch.save(self.examples, cache_file_path)
        
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens_1: {}".format([x.replace('\u0120','_') for x in example.input_tokens_1]))
                logger.info("input_ids_1: {}".format(' '.join(map(str, example.input_ids_1))))       
                logger.info("position_idx_1: {}".format(example.position_idx_1))
                logger.info("dfg_to_code_1: {}".format(' '.join(map(str, example.dfg_to_code_1))))
                logger.info("dfg_to_dfg_1: {}".format(' '.join(map(str, example.dfg_to_dfg_1))))
                
                logger.info("input_tokens_2: {}".format([x.replace('\u0120','_') for x in example.input_tokens_2]))
                logger.info("input_ids_2: {}".format(' '.join(map(str, example.input_ids_2))))       
                logger.info("position_idx_2: {}".format(example.position_idx_2))
                logger.info("dfg_to_code_2: {}".format(' '.join(map(str, example.dfg_to_code_2))))
                logger.info("dfg_to_dfg_2: {}".format(' '.join(map(str, example.dfg_to_dfg_2))))


    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask_1= np.zeros((self.args.code_length+self.args.data_flow_length,
                        self.args.code_length+self.args.data_flow_length),dtype=bool)
        #calculate begin index of node and max length of input
        # node_index=sum([i>1 for i in self.examples[item].position_idx_1])
        # max_length=sum([i!=1 for i in self.examples[item].position_idx_1])
        # #sequence can attend to sequence
        # attn_mask_1[:node_index,:node_index]=True
        # #special tokens attend to all tokens
        # for idx,i in enumerate(self.examples[item].input_ids_1):
        #     if i in [0,2]:
        #         attn_mask_1[idx,:max_length]=True
        # #nodes attend to code tokens that are identified from
        # for idx,(a,b) in enumerate(self.examples[item].dfg_to_code_1):
        #     if a<node_index and b<node_index:
        #         attn_mask_1[idx+node_index,a:b]=True
        #         attn_mask_1[a:b,idx+node_index]=True
        # #nodes attend to adjacent nodes
        # for idx,nodes in enumerate(self.examples[item].dfg_to_dfg_1):
        #     for a in nodes:
        #         if a+node_index<len(self.examples[item].position_idx_1):
        #             attn_mask_1[idx+node_index,a+node_index]=True
        # calculate begin index of node and max length of input
        k_random = self.args.k_random if self.args.smooth != 'hash' else self.args.kmer * self.args.k_random
        node_index = sum([i > 1 for i in self.examples[item].position_idx_1 if i <= k_random + 1])
        # max_length=sum([i!=1 for i in self.examples[item].position_idx])
        max_length = sum([i == 0 for i in self.examples[item].position_idx_1])
        # sequence can attend to sequence
        attn_mask_1[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids_1):
            if i in [0, 2]:
                attn_mask_1[idx, :max_length] = True
                    
        #calculate graph-guided masked function
        attn_mask_2= np.zeros((self.args.code_length+self.args.data_flow_length,
                        self.args.code_length+self.args.data_flow_length),dtype=bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx_2])
        max_length=sum([i!=1 for i in self.examples[item].position_idx_2])
        #sequence can attend to sequence
        attn_mask_2[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].input_ids_2):
            if i in [0,2]:
                attn_mask_2[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code_2):
            if a<node_index and b<node_index:
                attn_mask_2[idx+node_index,a:b]=True
                attn_mask_2[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg_2):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx_2):
                    attn_mask_2[idx+node_index,a+node_index]=True                      
                    
        return (torch.tensor(self.examples[item].input_ids_1),
                torch.tensor(self.examples[item].position_idx_1),
                torch.tensor(attn_mask_1), 
                torch.tensor(self.examples[item].input_ids_2),
                torch.tensor(self.examples[item].position_idx_2),
                torch.tensor(attn_mask_2),                 
                torch.tensor(self.examples[item].label))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    
    #build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    args.max_steps=args.epochs*len( train_dataloader)
    args.save_steps=len(train_dataloader)
    args.warmup_steps=args.max_steps//5
    model.to(args.device)
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step=0
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_f1=0

    model.zero_grad()
 
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            (inputs_ids_1,position_idx_1,attn_mask_1,
            inputs_ids_2,position_idx_2,attn_mask_2,
            labels)=[x.to(args.device)  for x in batch]
            model.train()
            inputs_ids_1_mask = model.transform(inputs_ids_1, tokenizer)
            loss,logits = model(inputs_ids_1_mask,position_idx_1,attn_mask_1,inputs_ids_2,position_idx_2,attn_mask_2,labels)

            if args.n_gpu > 1:
                loss = loss.mean()
                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
                
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_when_training=True)    
                    
                    # Save model checkpoint
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        
def evaluate(args, model, tokenizer, eval_when_training=False):
    #build dataloader
    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for sample_idx in range(args.n_sampling):
        for batch in tqdm(eval_dataloader):
            (inputs_ids_1,position_idx_1,attn_mask_1,
            inputs_ids_2,position_idx_2,attn_mask_2,
            labels)=[x.to(args.device)  for x in batch]
            with torch.no_grad():
                inputs_ids_1_mask = model.transform(inputs_ids_1, tokenizer)
                lm_loss,logit = model(inputs_ids_1_mask,position_idx_1,attn_mask_1,inputs_ids_2,position_idx_2,attn_mask_2,labels)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
            if sample_idx == 0:
                y_trues.append(labels.cpu().numpy())
                nb_eval_steps += 1

    #calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5

    preds = logits[:, 1] > best_threshold
    preds = np.array(preds).reshape([args.n_sampling, len(eval_dataset.examples)]).transpose([1, 0])
    preds_onehot = np.eye(2)[preds.astype(int)].sum(axis=-2)
    y_preds = np.argmax(preds_onehot, axis=-1)

    eval_acc = np.mean(y_trues == preds)
    eval_loss = eval_loss / (nb_eval_steps * args.n_sampling)
    perplexity = torch.tensor(eval_loss)
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='macro')
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 6)
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 5)))

    return result


def measurement(_y_true, _y_pred):
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
    accuracy = accuracy_score(_y_true, _y_pred)
    b_accuracy = balanced_accuracy_score(_y_true, _y_pred)
    tn, fp, fn, tp = confusion_matrix(_y_true, _y_pred).ravel()
    fpr = fp / float(tn + fp)
    fnr = fn / float(tp + fn)
    f1 = f1_score(_y_true, _y_pred, average='binary')

    return accuracy, b_accuracy, fnr, fpr, f1


def test(args, model, tokenizer, best_threshold=0.5, alpha=0.05):
    #build dataloader
    eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for sample_idx in range(args.n_sampling):
        for batch in tqdm(eval_dataloader):
            (inputs_ids_1,position_idx_1,attn_mask_1,
            inputs_ids_2,position_idx_2,attn_mask_2,
            labels)=[x.to(args.device)  for x in batch]
            with torch.no_grad():
                inputs_ids_1_mask = model.transform(inputs_ids_1, tokenizer)
                lm_loss,logit = model(inputs_ids_1_mask,position_idx_1,attn_mask_1,inputs_ids_2,position_idx_2,attn_mask_2,labels)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
            if sample_idx == 0:
                y_trues.append(labels.cpu().numpy())
                nb_eval_steps += 1
    

    #output result
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)

    preds = logits[:, 1] > best_threshold
    preds = np.array(preds).reshape([args.n_sampling, len(eval_dataset.examples)]).transpose([1, 0])
    preds_onehot = np.eye(2)[preds.astype(int)].sum(axis=-2).astype(int)
    y_preds = np.argmax(preds_onehot, axis=-1)

    top2 = preds_onehot.argsort(axis=-1)[:, ::-1][:, :2]
    count1 = preds_onehot[range(y_preds.shape[0]), top2[:, 0]]
    count2 = preds_onehot[range(y_preds.shape[0]), top2[:, 1]]
    abstain_flag = binom_test(count1, count1 + count2, prop=0.5) > alpha

    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='macro')
    result = {
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1)
    }
    accuracy, b_accuracy, fnr, fpr, f1 = measurement(y_trues, y_preds)
    logger.info(
        "Model of {} achieves the accuracy: {:.4f}%, balanced accuracy: {:.4f}%".format(args.smooth, accuracy * 100,
                                                                                        b_accuracy * 100))
    MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
    logger.info(MSG.format(fnr * 100, fpr * 100, f1 * 100))

    y_trues = y_trues[~abstain_flag]
    y_preds = y_preds[~abstain_flag]
    logger.info('Filter out outlier: {}.'.format(np.sum(abstain_flag)))
    recall2 = recall_score(y_trues, y_preds, average='macro')
    precision2 = precision_score(y_trues, y_preds, average='macro')
    f12 = f1_score(y_trues, y_preds, average='macro')
    result.update({
        "test_recall_filter": float(recall2),
        "test_precision_filter": float(precision2),
        "test_f1_filter": float(f12)
    })

    accuracy, b_accuracy, fnr, fpr, f1 = measurement(y_trues, y_preds)
    logger.info("After filter outliers\n")
    logger.info(
        "Model of {} achieves the accuracy: {:.5f}%, balanced accuracy: {:.5f}%".format(args.smooth, accuracy * 100,
                                                                                        b_accuracy * 100))
    MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
    logger.info(MSG.format(fnr * 100, fpr * 100, f1 * 100))

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 5)))

    return result


def certify(args, model, tokenizer, prefix="",pool=None,threshold=0.5):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Certification on Test Dataset *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    radius_all = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        (inputs_ids_1, position_idx_1, attn_mask_1,
         inputs_ids_2, position_idx_2, attn_mask_2,
         label) = [x.to(args.device) for x in batch]
        bs, dim = inputs_ids_1.shape
        with torch.no_grad():
            logits = []
            for sample_idx in range(args.n_sampling):
                inputs_ids_1_mask = model.transform(inputs_ids_1, tokenizer)
                lm_loss, logit = model(inputs_ids_1_mask, position_idx_1, attn_mask_1,
                                       inputs_ids_2, position_idx_2, attn_mask_2, label)
                logits.append(logit.cpu().numpy())
            logits = np.concatenate(logits)
            preds = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]
            preds = np.array(preds).reshape([args.n_sampling, bs]).transpose([1, 0])
            preds_selection = np.eye(2)[preds.astype(int)].sum(axis=-2)
            c_pred = preds_selection.argmax(axis=-1)

            logits = []
            for sample_idx in range(args.n_estimation):
                inputs_ids_1_mask = model.transform(inputs_ids_1, tokenizer)
                lm_loss, logit = model(inputs_ids_1_mask, position_idx_1, attn_mask_1,
                                       inputs_ids_2, position_idx_2, attn_mask_2, label)
                logits.append(logit.cpu().numpy())
            logits = np.concatenate(logits)
            preds = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]
            preds = np.array(preds).reshape([args.n_estimation, bs]).transpose([1, 0])
            preds_estimation = np.eye(2)[preds.astype(int)].sum(axis=-2)

            n_targeted = preds_estimation[range(len(c_pred)), c_pred]
            prob_underlined = lower_confidence_interval(n_targeted, args.n_estimation, args.alpha)
            n_runnerup = preds_estimation[range(len(c_pred)), 1 - c_pred]
            prob_overlined = upper_confidence_interval(n_runnerup, args.n_estimation, args.alpha)

            radius = np.zeros_like(c_pred, dtype=object)
            abstain_indicator = prob_underlined <= prob_overlined
            radius[abstain_indicator] = model.ABSTAIN
            incorrect_indicator = (c_pred != label.cpu().numpy())
            incorrect_indicator_true = incorrect_indicator == True
            if np.any(incorrect_indicator_true):
                incorrect_indicator[incorrect_indicator_true] = incorrect_indicator[incorrect_indicator_true] ^ \
                                                                abstain_indicator[incorrect_indicator_true]
                radius[incorrect_indicator] = model.ABSTAIN - 1
            total_abstain_indicator = abstain_indicator | incorrect_indicator
            if not all(total_abstain_indicator):
                if args.smooth == 'random':
                    radius[~total_abstain_indicator] = model.calc_radius(
                        np.array(prob_underlined[~total_abstain_indicator])[..., None],
                        dim,
                        args.k_random)
                elif args.smooth == 'randdel':
                    radius[~total_abstain_indicator] = model.calc_radius(
                        np.array(prob_underlined[~total_abstain_indicator])[..., None])
                elif args.smooth == 'hash':
                    radius[~total_abstain_indicator] = model.calc_radius(prob_underlined[~total_abstain_indicator],
                                                                         prob_overlined[~total_abstain_indicator],
                                                                         k_hashcode=args.k_random, max_radius=0.1,
                                                                         n_grid=100
                                                                         )
                else:
                    raise ValueError
            radius_all.append(radius)

            nb_eval_steps += 1

    radius_all = np.concatenate(radius_all)
    checkpoint_prefix = 'checkpoint-best-f1'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, '{}-{}-cert.npy'.format(args.smooth, int(args.k_random)))
    np.save(output_dir, radius_all)
    logger.info("Saving certification to %s", output_dir)

    return radius_all

                                           
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_certify", action='store_true',
                        help="Whether to run certification on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    parser.add_argument('--smooth', type=str, default='random', choices=['random', 'randdel', 'hash'],
                        help="random smoothing method.")
    parser.add_argument('--k_random', type=int, default=128, help="number of random codes.")
    parser.add_argument('--n_sampling', type=int, default=100, help="number of sampling.")
    parser.add_argument('--n_estimation', type=int, default=10000,
                        help="number of sampling for certification estimation.")
    parser.add_argument('--n_examples', type=int, default=200,
                        help="number of examples for certification or attacking.")
    parser.add_argument('--kmer', type=int, default=2, help="number of adjacent tokens.")
    parser.add_argument('--default_mode', action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--alpha", default=0.05, type=float,
                        help="confidence interval.")

    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu)


    # Set seed
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels=1
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config)    

    # model=Model(model,config,tokenizer,args)
    if args.smooth == "random":
        input_transformer = RandomTransformer(args.k_random,
                                              mask_id=tokenizer.pad_token_id,
                                              reuse_noise=True,
                                              seed=args.seed)
        model = RandomSmooth4LLM(model, config, tokenizer, input_transformer, args=args)
    elif args.smooth == 'randdel':
        input_transformer = RandomDeleter(p_del=1. - (float(args.k_random) / tokenizer.model_max_length),
                                          mask_id=tokenizer.pad_token_id,
                                          reuse_noise=True,
                                          seed=args.seed)
        model = RandomDelSmooth4LLM(model, config, tokenizer, input_transformer, args=args)
    elif args.smooth == 'hash':
        input_transformer = EditLSHTransformerTorch(args.code_length + args.data_flow_length,  # where are 2 tokens?
                                                    args.k_random,
                                                    args.kmer,
                                                    l_chucksize=1,
                                                    null_value=tokenizer.pad_token_id,
                                                    pad_value=tokenizer.pad_token_id,
                                                    position_fixed=False,
                                                    seed=args.seed
                                                    )
        model = HashSmooth4LLM(model, config, tokenizer, input_transformer,
                               args=args)

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/model_{}_{}.bin'.format(args.smooth, int(args.k_random))
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        results = evaluate(args, model, tokenizer)
        
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model_{}_{}.bin'.format(args.smooth, int(args.k_random))
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        results = test(args, model, tokenizer,best_threshold=0.5)

    if args.do_certify:
        checkpoint_prefix = 'checkpoint-best-f1/model_{}_{}.bin'.format(args.smooth, int(args.k_random))
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = certify(args, model, tokenizer)
        logger.info("***** Certify results *****")
        print('the mean:', np.mean(result))

    return results


if __name__ == "__main__":
    main()

