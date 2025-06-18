# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from tinycss2 import tokenizer
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
import math

from statsmodels.stats.proportion import proportion_confint, binom_test
from randomsmooth.random_tran import RandomTransformer
from hashsmooth.core import HashSmoothBase
from torchmalware.certification.perturbation import RandomPerturbation as TorchMalwareRandomPerturb
from torchmalware.certification import deletion_mech
import torchmalware
binom_test = np.vectorize(binom_test)
ABSTAIN = -1



class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
        self.query = 0
    
        
    def forward(self, input_ids=None,labels=None): 
        input_ids=input_ids.view(-1,self.args.block_size)
        outputs = self.encoder(input_ids= input_ids,attention_mask=input_ids.ne(self.tokenizer.pad_token_id))[0]
        logits=self.classifier(outputs)
        prob=F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob


    def get_results(self, dataset, batch_size, threshold=0.5):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=4,pin_memory=False)

        ## Evaluate Model

        eval_loss = 0.0
        self.eval()
        logits=[] 
        labels=[]
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")       
            label=batch[1].to("cuda") 
            with torch.no_grad():
                lm_loss,logit = self.forward(inputs,label)
                # 调用这个模型. 重写了反前向传播模型.
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                # 和defect detection任务不一样，这个的输出就是softmax值，而非sigmoid值
                labels.append(label.cpu().numpy())
        logits=np.concatenate(logits,0)
        labels=np.concatenate(labels,0)

        probs = logits
        pred_labels = [0 if first_softmax  > threshold else 1 for first_softmax in logits[:,0]]
        # 如果logits中的一个元素，其一个softmax值 > threshold, 则说明其label为0，反之为1

        return probs, pred_labels


def retain_specific_tokens(inputs_ids_mask, inputs_ids, tokenizer):
    assert inputs_ids_mask.shape == inputs_ids.shape
    boolean_var = torch.zeros_like(inputs_ids, device=inputs_ids.device, dtype=torch.bool)
    for spec_id in tokenizer.all_special_ids:
        boolean_var |= (inputs_ids == spec_id)
    inputs_ids_mask[boolean_var] = inputs_ids[boolean_var]
    return inputs_ids_mask


class RandomSmooth(object):
    ABSTAIN = ABSTAIN
    def __init__(self, num_of_classes, transform_method, max_k=1000, default_mode=True):
        self.num_of_classes = num_of_classes
        self.default_mode = default_mode
        self.max_k = max_k
        self.transform_method = transform_method

    def transform(self, x, tokenizer):
        dim_of_a_sample = x.shape[1] // 2
        x_transform = self.transform_method.transform(x[:, :dim_of_a_sample])
        x_transform = torch.hstack([x_transform,x[:, dim_of_a_sample:]])
        return retain_specific_tokens(x_transform, x, tokenizer)

    @staticmethod
    def lc_bound(k, n, alpha):
        return proportion_confint(k, n, alpha=2 * alpha, method="beta")[0]

    @staticmethod
    def calc_radius(scores_of_true, size, keep):
        count = scores_of_true.shape[0]
        done = torch.zeros(count, dtype=torch.uint8)
        radii = torch.zeros(count, dtype=torch.long)
        radius = 0
        lhs = (1.5 - scores_of_true).squeeze(1)
        while (done.sum() < count):
            rhs = math.factorial(size - radius) * math.factorial(size - keep) / (
                    math.factorial(size) * math.factorial(size - keep - radius))
            done[torch.tensor(lhs >= rhs)] = 1
            radii[torch.tensor(lhs < rhs)] = radius
            radius += 1
        return radii


class RandomSmooth4LLM(nn.Module, RandomSmooth):
    def __init__(self, encoder, config, tokenizer,
                 transform_method: RandomTransformer, args=None):
        nn.Module.__init__(self)
        RandomSmooth.__init__(self,
                              config.num_labels,
                              transform_method,
                              args.k_random,
                              args.default_mode,
                              )
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0


    def forward(self, input_ids=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        outputs = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

    def get_results(self, dataset, batch_size, threshold=0.5, alpha=0.05):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        eval_loss = 0.0
        self.eval()
        logits = []
        labels = []
        for sample_idx in range(self.args.n_sampling):
            for batch in eval_dataloader:
                inputs = batch[0].to("cuda")
                label = batch[1].to("cuda")
                with torch.no_grad():
                    inputs_mask = self.transform(inputs, self.tokenizer)
                    lm_loss, logit = self.forward(inputs_mask, label)
                    eval_loss += lm_loss.mean().item()
                    logits.append(logit.cpu().numpy())
                if sample_idx == 0:
                    labels.append(label.cpu().numpy())

        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]
        pred_labels = np.array(pred_labels).reshape([self.args.n_sampling, len(dataset)]).transpose([1, 0])
        pred_labels_onehot = np.eye(2)[pred_labels.astype(int)].sum(axis=-2)
        pred_labels = np.argmax(pred_labels_onehot, axis=-1)

        # abstain
        top2 = pred_labels_onehot.argsort(axis=-1)[:, ::-1][:, :2]
        count1 = pred_labels_onehot[range(len(dataset)), top2[:, 0]].astype(int)
        count2 = pred_labels_onehot[range(len(dataset)), top2[:, 1]].astype(int)
        abstain_flag = binom_test(count1, count1 + count2, prop=0.5) > alpha
        pred_labels[abstain_flag] = self.ABSTAIN

        probs = np.array(probs).reshape([self.args.n_sampling, len(dataset), -1]).transpose([1, 0, 2]).mean(axis=-2)
        labels = np.concatenate(labels, 0)

        return probs, pred_labels


class RandomDelSmooth(object):
    ABSTAIN = ABSTAIN
    def __init__(self, num_of_classes, transform_method, max_k=1000, default_mode=True):
        self.num_of_classes = num_of_classes
        self.transform_method = transform_method
        self.default_mode = default_mode
        self.max_k = max_k

    def transform(self, x, tokenizer):
        dim_of_a_sample = x.shape[1] // 2
        x_transform = self.transform_method.transform(x[:, :dim_of_a_sample]) # the first example
        x_transform = torchmalware.models.utils.window_pad(x_transform, dim_of_a_sample,
                                                                  dim=1,
                                                                  value=self.transform_method.pad_value)
        x_transform_ = torch.hstack([x_transform, x[:, dim_of_a_sample:]])
        return retain_specific_tokens(x_transform_, x, tokenizer)

    @staticmethod
    def lc_bound(k, n, alpha):
        return proportion_confint(k, n, alpha=2 * alpha, method="beta")[0]

    def calc_radius(self, p_lower):
        return self.transform_method.calc_radius(p_lower)

class RandomDelSmooth4LLM(nn.Module, RandomDelSmooth):
    def __init__(self, encoder, config, tokenizer,
                 transform_method, args=None):
        nn.Module.__init__(self)
        RandomDelSmooth.__init__(self,
                              config.num_labels,
                              transform_method,
                              args.k_random,
                              args.default_mode
                              )
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, input_ids=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        outputs = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

    def get_results(self, dataset, batch_size, threshold=0.5, alpha=0.05):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        eval_loss = 0.0
        self.eval()
        logits = []
        labels = []
        for sample_idx in range(self.args.n_sampling):
            for batch in eval_dataloader:
                inputs = batch[0].to("cuda")
                label = batch[1].to("cuda")
                with torch.no_grad():
                    inputs_mask = self.transform(inputs, self.tokenizer)
                    lm_loss, logit = self.forward(inputs_mask, label)
                    eval_loss += lm_loss.mean().item()
                    logits.append(logit.cpu().numpy())
                if sample_idx == 0:
                    labels.append(label.cpu().numpy())

        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]
        pred_labels = np.array(pred_labels).reshape([self.args.n_sampling, len(dataset)]).transpose([1, 0])
        pred_labels_onehot = np.eye(2)[pred_labels.astype(int)].sum(axis=-2)
        pred_labels = np.argmax(pred_labels_onehot, axis=-1)

        # abstain
        top2 = pred_labels_onehot.argsort(axis=-1)[:, ::-1][:, :2]
        count1 = pred_labels_onehot[range(len(dataset)), top2[:, 0]].astype(int)
        count2 = pred_labels_onehot[range(len(dataset)), top2[:, 1]].astype(int)
        abstain_flag = binom_test(count1, count1 + count2, prop=0.5) > alpha
        pred_labels[abstain_flag] = self.ABSTAIN

        probs = np.array(probs).reshape([self.args.n_sampling, len(dataset), -1]).transpose([1, 0, 2]).mean(axis=-2)
        labels = np.concatenate(labels, 0)

        return probs, pred_labels


class HashSmooth(HashSmoothBase):
    ABSTAIN = ABSTAIN

    def __init__(self, num_of_classes, transform_method, max_k=1000, default_mode=True):
        super(HashSmooth, self).__init__(num_of_classes, transform_method, max_k, None, None, default_mode)
        self.transform_method = transform_method

    def transform(self, x, tokenizer):
        dim_of_a_sample = x.shape[1] // 2
        x_transform = self.transform_method.transform(x[:, :dim_of_a_sample], self.k_hashcode)  # the first example
        x_transform_ = torch.hstack([x_transform, x[:, dim_of_a_sample:]])
        return retain_specific_tokens(x_transform_, x, tokenizer)

    def calc_radius(self, probas, second_probas, k_hashcode=None, max_radius=None, n_grid=100):
        return self._calc_radius(probas, second_probas, k_hashcode, max_radius, n_grid)


class HashSmooth4LLM(nn.Module, HashSmooth):
    def __init__(self, encoder, config, tokenizer,
                 transform_method, args=None):
        nn.Module.__init__(self)
        HashSmooth.__init__(self,
                            config.num_labels,
                            transform_method,
                            args.k_random,
                            args.default_mode
                            )
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, input_ids=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        outputs = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

    def get_results(self, dataset, batch_size, threshold=0.5, alpha=0.05):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        eval_loss = 0.0
        self.eval()
        logits = []
        labels = []
        for sample_idx in range(self.args.n_sampling):
            for batch in eval_dataloader:
                inputs = batch[0].to("cuda")
                label = batch[1].to("cuda")
                with torch.no_grad():
                    inputs_mask = self.transform(inputs, self.tokenizer)
                    lm_loss, logit = self.forward(inputs_mask, label)
                    eval_loss += lm_loss.mean().item()
                    logits.append(logit.cpu().numpy())
                if sample_idx == 0:
                    labels.append(label.cpu().numpy())

        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]
        pred_labels = np.array(pred_labels).reshape([self.args.n_sampling, len(dataset)]).transpose([1, 0])
        pred_labels_onehot = np.eye(2)[pred_labels.astype(int)].sum(axis=-2)
        pred_labels = np.argmax(pred_labels_onehot, axis=-1)

        # abstain
        top2 = pred_labels_onehot.argsort(axis=-1)[:, ::-1][:, :2]
        count1 = pred_labels_onehot[range(len(dataset)), top2[:, 0]].astype(int)
        count2 = pred_labels_onehot[range(len(dataset)), top2[:, 1]].astype(int)
        abstain_flag = binom_test(count1, count1 + count2, prop=0.5) > alpha
        pred_labels[abstain_flag] = self.ABSTAIN

        probs = np.array(probs).reshape([self.args.n_sampling, len(dataset), -1]).transpose([1, 0, 2]).mean(axis=-2)
        labels = np.concatenate(labels, 0)

        return probs, pred_labels


