# coding=utf-8
# @Time    : 2020/7/8
# @Author  : Zhou Yang
# @Email   : zyang@smu.edu.sg
# @File    : attack.py
'''For attacking CodeBERT models'''
import json
import sys
import os
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('../../../')
sys.path.append('../../../python_parser')

import csv
import argparse
import warnings
import pickle
import copy
from functools import partial
from datetime import datetime
import torch
import multiprocessing
import time
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.append('../../../python_parser')
sys.path.append('../../../../hashsmooth')
sys.path.append('../../../../randomsmooth')
sys.path.append('../../../../torchware')

from model import Model
from utils import set_seed
from utils import Recorder
from run_smooth import TextDataset
from attacker import ALERT_Attacker
from transformers import RobertaForMaskedLM
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2Model, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from model import RandomSmooth4LLM, RandomDelSmooth4LLM, HashSmooth4LLM, binom_test

from randomsmooth.random_tran import RandomTransformer
from torchmalware.random_del_wrapper import RandomDeleter
from hashsmooth import EditLSHTransformerTorch
from hashsmooth.utils_hash import lower_confidence_interval, upper_confidence_interval

from transformers import logging
logging.set_verbosity_error()
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning
warnings.filterwarnings("ignore")

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2Model, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}
logger = logging.get_logger(__name__)


def get_code_pairs(file_path):
    postfix=file_path.split('/')[-1].split('.txt')[0]
    folder = '/'.join(file_path.split('/')[:-1]) # 得到文件目录
    code_pairs_file_path = os.path.join(folder, 'cached_{}.pkl'.format(
                                    postfix))
    with open(code_pairs_file_path, 'rb') as f:
        code_pairs = pickle.load(f)
    return code_pairs


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--csv_store_path", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--use_ga", action='store_true',
                        help="Whether to GA-Attack.")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
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
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--smooth', type=str, default='random', choices=['random', 'hash', 'randdel', 'none'],
                        help="random smoothing method.")
    parser.add_argument('--k_random', type=int, default=64, help="number of random codes.")
    parser.add_argument('--n_sampling', type=int, default=100, help="number of sampling.")
    parser.add_argument('--n_estimation', type=int, default=10000,
                        help="number of sampling for certification estimation.")
    parser.add_argument('--kmer', type=int, default=2, help="number of adjacent tokens.")
    parser.add_argument('--n_examples', type=int, default=200,
                        help="number of examples for certification or attacking.")
    parser.add_argument('--default_mode', action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--alpha", default=0.05, type=float,
                        help="confidence interval.")

    args = parser.parse_args()

    device = torch.device("cuda")
    args.device = device

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    if args.smooth == 'none':
        model = Model(model, config, tokenizer, args)
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    elif args.smooth == "random":
        input_transformer = RandomTransformer(args.k_random,
                                              mask_id=tokenizer.pad_token_id,
                                              reuse_noise=True,
                                              seed=args.seed)
        model = RandomSmooth4LLM(model, config, tokenizer, input_transformer, args=args)
        checkpoint_prefix = 'checkpoint-best-f1/model_{}_{}.bin'.format(args.smooth, int(args.k_random))
        model.get_results = partial(model.get_results, alpha=args.alpha)
    elif args.smooth == 'randdel':
        input_transformer = RandomDeleter(p_del=1. - (float(args.k_random) / tokenizer.model_max_length),
                                          mask_id=tokenizer.pad_token_id,
                                          reuse_noise=True,
                                          seed=args.seed)
        model = RandomDelSmooth4LLM(model, config, tokenizer, input_transformer, args=args)
        checkpoint_prefix = 'checkpoint-best-f1/model_{}_{}.bin'.format(args.smooth, int(args.k_random))
        model.get_results = partial(model.get_results, alpha=args.alpha)
    elif args.smooth == 'hash':
        input_transformer = EditLSHTransformerTorch(tokenizer.max_len_single_sentence,  # where are 2 tokens?
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
        checkpoint_prefix = 'checkpoint-best-f1/model_{}_{}.bin'.format(args.smooth, int(args.k_random))
        model.get_results = partial(model.get_results, alpha=args.alpha)
    else:
        raise NotImplementedError

    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    print("{} - reload model from {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), output_dir))

    ## Load MLM model
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)

    cpu_cont = 16

    # Load Dataset
    ## Load Dataset
    pool = multiprocessing.Pool(cpu_cont)
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file, pool = pool)
    ## Load code pairs
    source_codes = get_code_pairs(args.eval_data_file)
    if args.do_certify:
        source_codes = source_codes[:args.n_examples]

    postfix = args.eval_data_file.split('/')[-1].split('.txt')[0].split("_")
    folder = '/'.join(args.eval_data_file.split('/')[:-1])  # 得到文件目录
    subs_path = os.path.join(folder, 'test_subs_{}_{}.jsonl'.format(
        postfix[-2], postfix[-1]))
    generated_substitutions = []
    with open(subs_path) as f:
        for line in f:
            js = json.loads(line.strip())
            generated_substitutions.append(js["substitutes"])
    if args.do_certify:
        generated_substitutions = generated_substitutions[:args.n_examples]
    assert len(source_codes) == len(eval_dataset) == len(generated_substitutions), "{}-{}-{}".format(len(source_codes),
                                                                                         len(eval_dataset),
                                                                                         len(generated_substitutions))

    success_attack = 0
    total_cnt = 0

    recoder = Recorder(args.csv_store_path)
    attacker = ALERT_Attacker(args, model, tokenizer, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)
    start_time = time.time()
    query_times = 0
    features = []
    new_features = []
    status = []
    for index, example in enumerate(eval_dataset):
        print("Index: ", index)
        example_start_time = time.time()
        code_pair = source_codes[index]
        logits, preds = model.get_results([example], args.eval_batch_size)
        orig_label = preds[0]
        true_label = example[1].item()
        # if not orig_label == true_label:
        #     continue

        substitutes = generated_substitutions[index]
        first_code = code_pair[2]
        identifiers = list(substitutes.keys())
        # if len(identifiers) == 0:
        #     continue
        total_cnt += 1

        print(identifiers)
        code, prog_length, feature, new_feature, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.greedy_attack(
            example, substitutes, code_pair)
        attack_type = "Greedy"
        if is_success == -1 and args.use_ga:
            code, prog_length, feature, new_feature, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.ga_attack(
                example, substitutes, code, initial_replace=replaced_words)
            attack_type = "GA"

        example_end_time = (time.time() - example_start_time) / 60
        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
        print("Query times in this attack: ", model.query - query_times)

        score_info = ''
        if names_to_importance_score is not None:
            for key in names_to_importance_score.keys():
                score_info += key + ':' + str(names_to_importance_score[key]) + ','

        replace_info = ''
        if replaced_words is not None:
            for key in replaced_words.keys():
                replace_info += key + ':' + replaced_words[key] + ','

        features.append(feature)
        new_features.append(new_feature)
        status.append(is_success)

        if is_success == 1:
            success_attack += 1
            recoder.write(index, code, feature, new_feature, prog_length,
                          adv_code, true_label, orig_label, temp_label,
                          is_success, variable_names, score_info, nb_changed_var, nb_changed_pos, replace_info,
                          attack_type, model.query - query_times, example_end_time)
        else:
            recoder.write(index, None, feature, new_feature, prog_length,
                          None, true_label, orig_label, temp_label,
                          is_success, variable_names, score_info, nb_changed_var, nb_changed_pos, replace_info,
                          attack_type, model.query - query_times, example_end_time)
        query_times = model.query
        print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
    print("Final success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))

if __name__ == '__main__':
    main()