import os
import sys
import argparse

from configparser import SafeConfigParser

sys.path.append('..')

class Configurable:

    def __init__(self, config_file, extra_args, show = False):
        config = SafeConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = { k[2:] : v for k, v in zip(extra_args[0::2], extra_args[1::2]) }
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
                    print(section, k, v)
        self._config = config
        print("Loaded config file successful.")
        if show :
            for section in config.sections():
                for k, v in config.items(section):
                    print(k, v)
        

    @property
    def data_dir(self):
        return self._config.get('Data','data_dir')

    @property
    def target_data_file(self):
        return self._config.get('Data','target_data_file')

    @property
    def clean_pos_file(self):
        return self._config.get('Data','clean_pos_file')

    @property
    def noisy_pos_file(self):
        return self._config.get('Data','noisy_pos_file')

    @property
    def clean_NA_file(self):
        return self._config.get('Data','clean_NA_file')

    @property
    def noisy_NA_file(self):
        return self._config.get('Data','noisy_NA_file')

    @property
    def test_file(self):
        return self._config.get('Data','test_file')
        
    @property
    def val_file(self):
        return self._config.get('Data','val_file')
    @property
    def rel2id_file(self):
        return self._config.get('Data','rel2id_file')

    @property
    def pretrained_embeddings_file(self):
        return self._config.get('Data','pretrained_embeddings_file')

    @property
    def word2id_file(self):
        return self._config.get('Data','word2id_file')

    @property
    def word2vec_file(self):
        return self._config.get('Data','word2vec_file')

    @property
    def max_length(self):
        return self._config.getint('Model','max_length')

    @property
    def word_size(self):
        return self._config.getint('Model','word_size')

    @property
    def position_size(self):
        return self._config.getint('Model','position_size')

    @property
    def hidden_size(self):
        return self._config.getint('Model','hidden_size')

    @property
    def kernel_size(self):
        return self._config.getint('Model','kernel_size')

    @property
    def padding_size(self):
        return self._config.getint('Model','padding_size')

    @property
    def dropout(self):
        return self._config.getfloat('Model','dropout')

