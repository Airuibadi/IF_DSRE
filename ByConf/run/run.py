import os, sys
import copy
import json
import numpy as np
sys.path.append('../')
import src
from src import encoder, model, framework
import argparse
from config import Configurable
import warnings
warnings.filterwarnings("ignore")

argparser = argparse.ArgumentParser()
argparser.add_argument('--config_file', default='../configs/default_ppc.cfg')
args, extra_args = argparser.parse_known_args()
target_rel = args.config_file[-7:-4]
config = Configurable(args.config_file, extra_args, show=True)

rel2id = json.load(open(config.rel2id_file))
word2id = json.load(open(config.word2id_file))
word2vec = np.load(config.word2vec_file)
assert word2vec.shape[1] == config.word_size

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
    mask_entity = True,
    fix_embedding = True
)
model = src.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
model.cuda()

MTF = src.framework.MeanTeacherDenoise(model=model,
                                    clean_pos_path = config.clean_pos_file,
                                    noisy_pos_path = config.noisy_pos_file,
                                    clean_NA_path = config.clean_NA_file,
                                    noisy_NA_path = config.noisy_NA_file,
                                    dev_path = config.val_file,
                                    test_path = config.test_file,
                                    target_rel_name = target_rel,
                                    target_rel_path = config.rel2id_file,
                                    config_file = args.config_file) 
MTF.process_conf()
