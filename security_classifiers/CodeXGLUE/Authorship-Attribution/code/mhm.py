import torch
import sys
import os

sys.path.append('../../../')
sys.path.append('../../../python_parser')
retval = os.getcwd()

from functools import partial
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.append('../../../python_parser')
sys.path.append('../../../../hashsmooth')
sys.path.append('../../../../randomsmooth')
sys.path.append('../../../../torchware')

import csv
import json
import argparse
import warnings
import torch
import numpy as np
from model import Model
from utils import set_seed
from utils import Recorder
from run import TextDataset
from utils import CodeDataset
from run_parser import get_identifiers, get_example
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaTokenizer, RobertaModel)
from attacker import MHM_Attacker
from attacker import convert_examples_to_features

from model import RandomSmooth4LLM, RandomDelSmooth4LLM, HashSmooth4LLM, binom_test

from randomsmooth.random_tran import RandomTransformer
from torchmalware.random_del_wrapper import RandomDeleter
from hashsmooth import EditLSHTransformerTorch
from hashsmooth.utils_hash import lower_confidence_interval, upper_confidence_interval

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning) # Only report warning\

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
}

from utils import build_vocab
            
def main():
    
    import json
    import pickle
    import time
    import os
    
    # import tree as Tree
    # from dataset import Dataset, POJ104_SEQ
    # from lstm_classifier import LSTMEncoder, LSTMClassifier
    
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
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--csv_store_path", default=None, type=str,
                        help="results")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--is_original_mhm", action='store_true',
                        help="whether to use the original mhm")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--number_labels", type=int,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_certify", action='store_true',
                        help="Whether to run certification on the dev set.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")

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


    args.device = torch.device("cuda")
    # Set seed
    set_seed(args.seed)

    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to('cuda') 

    args.start_epoch = 0
    args.start_step = 0

    ## Load Target Model
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last') # 读取model的路径
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        # 如果路径存在且有内容，则从checkpoint load模型
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())


    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels= args.number_labels 
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
    print ("MODEL LOADED!")


    ## Load Dataset
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    file_type = args.eval_data_file.split('/')[-1].split('.')[0] # valid
    folder = '/'.join(args.eval_data_file.split('/')[:-1]) # 得到文件目录
    codes_file_path = os.path.join(folder, '{}_subs.jsonl'.format(
                                file_type))
    print(codes_file_path)
    source_codes = []
    substs = []
    with open(codes_file_path) as rf:
        for line in rf:
            item = json.loads(line.strip())
            source_codes.append(item["code"].replace("\\n", "\n").replace('\"','"'))
            substs.append(item["substitutes"])
    assert(len(source_codes) == len(eval_dataset) == len(substs))

    code_tokens = []
    for index, code in enumerate(source_codes):
        code_tokens.append(get_identifiers(code, "python")[1])

    id2token, token2id = build_vocab(code_tokens, 5000)

    recoder = Recorder(args.csv_store_path)
    attacker = MHM_Attacker(args, model, codebert_mlm, tokenizer_mlm, token2id, id2token)
    
    # token2id: dict,key是变量名, value是id
    # id2token: list,每个元素是变量名

    print ("ATTACKER BUILT!")
    
    adv = {"tokens": [], "raw_tokens": [], "ori_raw": [],
           'ori_tokens': [], "label": [], }
    n_succ = 0.0
    total_cnt = 0
    query_times = 0
    all_start_time = time.time()
    for index, example in enumerate(eval_dataset):
        code = source_codes[index]
        subs = substs[index]

        orig_prob, orig_label = model.get_results([example], args.eval_batch_size)
        orig_prob = orig_prob[0]
        orig_label = orig_label[0]
        ground_truth = example[1].item()
        start_time = time.time()
        if orig_label != ground_truth:
            _res = {'succ': None, 'tokens': code, 'raw_tokens': code}
        else:
            # 这里需要进行修改.
            if args.is_original_mhm:
                _res = attacker.mcmc_random(tokenizer, code,
                                    _label=ground_truth, _n_candi=30,
                                    _max_iter=30, _prob_threshold=1, subs = subs)
            else:
                _res = attacker.mcmc(tokenizer, code,
                                    _label=ground_truth, _n_candi=30,
                                    _max_iter=30, _prob_threshold=1, subs = subs)
        
        if _res['succ'] is None:
            pass
        if _res['succ'] == True:
            print ("EXAMPLE "+str(index)+" SUCCEEDED!")
            n_succ += 1
            adv['tokens'].append(_res['tokens'])
            adv['raw_tokens'].append(_res['raw_tokens'])
        else:
            print ("EXAMPLE "+str(index)+" FAILED.")
        total_cnt += 1
        print ("  time cost = %.2f min" % ((time.time()-start_time)/60))
        time_cost = (time.time()-start_time)/60
        print ("  ALL EXAMPLE time cost = %.2f min" % ((time.time()-all_start_time)/60))
        print ("  curr succ rate = "+str(n_succ/total_cnt))
        print("Query times in this attack: ", model.query - query_times)
        print("All Query times: ", model.query)
        # recoder.writemhm(index, code, code, _res['tokens'], _res["prog_length"], _res['tokens'], ground_truth, orig_label, _res["new_pred"], _res["is_success"], _res["old_uid"], _res["score_info"], _res["nb_changed_var"], _res["nb_changed_pos"], _res["replace_info"], _res["attack_type"], model.query - query_times, time_cost)
        if _res['succ'] == True:
            recoder.writemhm(index, code, code, _res['tokens'],
                         _res["prog_length"], " ".join(_res['tokens']), ground_truth, orig_label, _res["new_pred"], _res["is_success"], _res["old_uid"], _res["score_info"], _res["nb_changed_var"], _res["nb_changed_pos"], _res["replace_info"], _res["attack_type"], model.query - query_times, "0")

        else:
            recoder.writemhm(index, code, code, _res['tokens'],
                          None, " ".join(_res['tokens']), ground_truth, orig_label, None, 0, None, None, None, None, None, None, model.query - query_times, None)
        query_times = model.query

if __name__ == "__main__":
    main()