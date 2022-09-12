import sys, json
import tqdm
sys.path.append('../')
import torch
import os
import numpy as np
import argparse
import src
from pathlib import Path
from config import Configurable
from src import encoder, model, framework
import warnings
warnings.filterwarnings("ignore")

np.random.seed(1219)
torch.manual_seed(1219)
torch.cuda.manual_seed_all(1219)
np.random.seed(1219)
torch.backends.cudnn.deterministic = True
argparser = argparse.ArgumentParser()
argparser.add_argument('--config_file', default='../configs/default_llc.cfg')
argparser.add_argument('--model_name', default='Unnamed')
argparser.add_argument('--val_file')
argparser.add_argument('--train_file')
argparser.add_argument('--model_path')
argparser.add_argument('--batch_size', type = int)
argparser.add_argument('--scale', type = int)
argparser.add_argument('--damp', type = float)
argparser.add_argument('--outdir')
argparser.add_argument('--gpu')
args, extra_args = argparser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

config = Configurable(args.config_file, extra_args)

rel2id = json.load(open(config.rel2id_file))
word2id = json.load(open(config.word2id_file))
word2vec = np.load(config.word2vec_file)
sentence_encoder = src.encoder.CNNEncoder(
    token2id=word2id,
    max_length=config.max_length,
    word_size=config.word_size,
    position_size=config.position_size,
    hidden_size=config.hidden_size,
    blank_padding=True,
    kernel_size=config.kernel_size,
    padding_size=config.padding_size,
    word2vec=word2vec,
    dropout=config.dropout,
    #mask_entity = True,
    fix_embedding = True
)


model = src.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

framework = src.framework.IFCalc(
    train_path = args.train_file,
    test_path = args.val_file,
    model = model,
    save_model_path = args.model_path,
    batch_size=args.batch_size)


framework.process_step1(
    s_test_outdir=args.outdir,
    recursion_depth=31,
    r_averaging=2,
    damp=args.damp,
    scale=args.scale)

