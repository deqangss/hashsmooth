import torch
import torch.nn as nn
import torch
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
    
        
    def forward(self, inputs_ids_1,position_idx_1,attn_mask_1,inputs_ids_2,position_idx_2,attn_mask_2,labels=None): 
        bs,l=inputs_ids_1.size()
        inputs_ids=torch.cat((inputs_ids_1.unsqueeze(1),inputs_ids_2.unsqueeze(1)),1).view(bs*2,l)
        position_idx=torch.cat((position_idx_1.unsqueeze(1),position_idx_2.unsqueeze(1)),1).view(bs*2,l)
        attn_mask=torch.cat((attn_mask_1.unsqueeze(1),attn_mask_2.unsqueeze(1)),1).view(bs*2,l,l)

        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0]
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

        self.eval()
        logits=[] 
        labels=[]
        for batch in eval_dataloader:
            (inputs_ids_1,position_idx_1,attn_mask_1,
            inputs_ids_2,position_idx_2,attn_mask_2,
            label)=[x.to("cuda")  for x in batch]
            with torch.no_grad():
                logit = self.forward(inputs_ids_1,position_idx_1,attn_mask_1,inputs_ids_2,position_idx_2,attn_mask_2)
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
        x_transform = self.transform_method.transform(x)
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


    def forward(self, inputs_ids_1,position_idx_1,attn_mask_1,inputs_ids_2,position_idx_2,attn_mask_2,labels=None):
        bs, l = inputs_ids_1.size()
        inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs * 2, l)
        position_idx = torch.cat((position_idx_1.unsqueeze(1), position_idx_2.unsqueeze(1)), 1).view(bs * 2, l)
        attn_mask = torch.cat((attn_mask_1.unsqueeze(1), attn_mask_2.unsqueeze(1)), 1).view(bs * 2, l, l)

        # embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        outputs = \
        self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[0]
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
                (inputs_ids_1, position_idx_1, attn_mask_1,
                 inputs_ids_2, position_idx_2, attn_mask_2,
                 label) = [x.to("cuda") for x in batch]
                with torch.no_grad():
                    inputs_ids_1_mask = self.transform(inputs_ids_1, self.tokenizer)
                    logit = self.forward(inputs_ids_1_mask,
                                         position_idx_1,
                                         attn_mask_1,
                                         inputs_ids_2,
                                         position_idx_2,
                                         attn_mask_2)
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
    def __init__(self, num_of_classes, transform_method, max_k=1000, code_length=384, data_flow_length=128, default_mode=True):
        self.num_of_classes = num_of_classes
        self.transform_method = transform_method
        self.default_mode = default_mode
        self.max_k = max_k
        self.code_length = code_length
        self.data_flow_length = data_flow_length

    def transform(self, x, tokenizer):
        inputs_ids_1_mask1 = self.transform_method.transform(x[:, :self.code_length])
        inputs_ids_1_mask1 = torchmalware.models.utils.window_pad(inputs_ids_1_mask1, self.code_length, dim=1,
                                                                  value=self.transform_method.pad_value)
        inputs_ids_1_mask2 = self.transform_method.transform(
            x[:, self.code_length:self.code_length + self.data_flow_length])
        inputs_ids_1_mask2 = torchmalware.models.utils.window_pad(inputs_ids_1_mask2, self.data_flow_length, dim=1,
                                                                  value=self.transform_method.pad_value)
        inputs_ids_1_mask = torch.hstack([inputs_ids_1_mask1, inputs_ids_1_mask2])
        return retain_specific_tokens(inputs_ids_1_mask, x, tokenizer)

    @staticmethod
    def lc_bound(k, n, alpha):
        return proportion_confint(k, n, alpha=2 * alpha, method="beta")[0]

    @staticmethod
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
                                 args.code_length,
                                 args.data_flow_length,
                                 args.default_mode
                              )
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2, attn_mask_2,
                labels=None):
        bs, l = inputs_ids_1.size()
        inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs * 2, l)
        position_idx = torch.cat((position_idx_1.unsqueeze(1), position_idx_2.unsqueeze(1)), 1).view(bs * 2, l)
        attn_mask = torch.cat((attn_mask_1.unsqueeze(1), attn_mask_2.unsqueeze(1)), 1).view(bs * 2, l, l)

        # embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        outputs = \
            self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[
                0]
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
                (inputs_ids_1, position_idx_1, attn_mask_1,
                 inputs_ids_2, position_idx_2, attn_mask_2,
                 label) = [x.to("cuda") for x in batch]
                with torch.no_grad():
                    inputs_ids_1_mask = self.transform(inputs_ids_1, self.tokenizer)
                    logit = self.forward(inputs_ids_1_mask,
                                         position_idx_1,
                                         attn_mask_1,
                                         inputs_ids_2,
                                         position_idx_2,
                                         attn_mask_2)
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

    def __init__(self, num_of_classes, transform_method, max_k=1000, code_length=384, data_flow_length=128, default_mode=True):
        super(HashSmooth, self).__init__(num_of_classes, transform_method, max_k, None, None, default_mode)
        self.transform_method = transform_method
        self.code_length = code_length
        self.data_flow_length = data_flow_length

    def transform(self, x, tokenizer):
        sub_k1 = int(self.k_hashcode * (self.code_length / float(self.code_length + self.data_flow_length)))
        inputs_ids_1_mask1 = self.transform_method.transform(x[:, :self.code_length], sub_k1)[:, :self.code_length]
        sub_k2 = self.k_hashcode - sub_k1
        if sub_k2 <= 0:
            return retain_specific_tokens(inputs_ids_1_mask1, x, tokenizer)
        inputs_ids_1_mask2 = self.transform_method.transform(
            x[:, self.code_length:self.code_length + self.data_flow_length], sub_k2)[:, :self.data_flow_length]
        inputs_ids_1_mask = torch.hstack([inputs_ids_1_mask1, inputs_ids_1_mask2])
        return retain_specific_tokens(inputs_ids_1_mask, x, tokenizer)

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
                            args.code_length,
                            args.data_flow_length,
                            args.default_mode
                            )
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, inputs_ids_1,position_idx_1,attn_mask_1,inputs_ids_2,position_idx_2,attn_mask_2,labels=None):
        bs, l = inputs_ids_1.size()
        inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs * 2, l)
        position_idx = torch.cat((position_idx_1.unsqueeze(1), position_idx_2.unsqueeze(1)), 1).view(bs * 2, l)
        attn_mask = torch.cat((attn_mask_1.unsqueeze(1), attn_mask_2.unsqueeze(1)), 1).view(bs * 2, l, l)

        # embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        outputs = \
            self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[
                0]
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
                (inputs_ids_1, position_idx_1, attn_mask_1,
                 inputs_ids_2, position_idx_2, attn_mask_2,
                 label) = [x.to("cuda") for x in batch]
                with torch.no_grad():
                    inputs_ids_1_mask = self.transform_method.transform(inputs_ids_1, self.tokenizer)
                    logit = self.forward(inputs_ids_1_mask,
                                         position_idx_1,
                                         attn_mask_1,
                                         inputs_ids_2,
                                         position_idx_2,
                                         attn_mask_2)
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