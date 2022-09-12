import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from pathlib import Path
from .data_loader import SentenceRELoader
from .utils import  display_progress
from .influence_function import s_test, grad_z
import numpy as np
from random import sample
np.random.seed(1219)
torch.manual_seed(1219)
torch.cuda.manual_seed_all(1219)
torch.backends.cudnn.deterministic = True
class IFSentenceRE_check(nn.Module):
    def __init__(self, 
                 model,
                 train_path, 
                 batch_size) :
    
        super().__init__()
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True)
        # Model
        self.model = model
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
                # Cuda
        if torch.cuda.is_available():
            self.cuda()


    def calc_s_test_single(self, ins_data, damp=0.01, scale=25, recursion_depth=100, r=1):
        s_test_vec_list = []
        for i in range(r):
            s_test_vec_list.append(s_test(ins_data, self.model, self.train_loader, self.criterion,
                                         damp=damp, scale=scale,recursion_depth=recursion_depth))
        s_test_vec = s_test_vec_list[0]
        for i in range(1, r):
            s_test_vec += s_test_vec_list[i]
        s_test_vec = [i / r for i in s_test_vec]
        return s_test_vec

    def calc_grad_z_single(self, ins_data) :
        grad_z_vec = grad_z(ins_data, self.model, self.criterion)
        return grad_z_vec
    
    def grid_search(self, scale_start=25, scale_end=500, damp_start=0.01, damp_end=0.2) :
    #def grid_search(self, scale_start=200, scale_end=500, damp_start=0.15, damp_end=0.2) :
        #print(len(self.train_loader.dataset))
        search_counter = 0
        total_count = 100
        train_sample = sample([i for i in range(len(self.train_loader.dataset))], min(20, len(self.train_loader.dataset)))
        for damp in np.arange(damp_start, 0.06, 0.01) :
            for scale in np.arange(scale_start, 250, 25) :
                flag = True
                for k, idx in enumerate(train_sample) :
                    display_progress("In search(%f, %d) validation %d/%d"%(damp, scale, k, len(train_sample)), search_counter, total_count)
                    ins_data = self.train_loader.dataset[idx]
                    s_test_vec = self.calc_s_test_single(ins_data, damp=damp, scale=scale, r=1)
                    grad_z_vec = self.calc_grad_z_single(ins_data)
                    influence = self.calc_influence_function(s_test_vec, grad_z_vec)
                    if influence < 0 :
                        flag = False
                        break
                if flag :
                    return scale, damp
                search_counter += 1
        for damp in np.arange(0.06, 0.11, 0.01) :
            for scale in np.arange(250, 450, 50) :
                flag = True
                for k, idx in enumerate(train_sample) :
                    ins_data = self.train_loader.dataset[idx]
                    display_progress("In search(%f, %d) validation %d/%d"%(damp, scale, k, len(train_sample)), search_counter, total_count)
                    s_test_vec = self.calc_s_test_single(ins_data, damp=damp, scale=scale, r=1)
                    grad_z_vec = self.calc_grad_z_single(ins_data)
                    influence = self.calc_influence_function(s_test_vec, grad_z_vec)
                    if influence < 0 :
                        flag = False
                        break
                if flag :
                    return scale, damp
                search_counter += 1
        return 500, 0.12
    def calc_influence_function(self, s_test, grad_z):
        tmp_influence = sum(
            [
                torch.sum(k * j).data.cpu().numpy()
                for k, j in zip(grad_z, s_test)
            ])
        return tmp_influence

    def load_model(self, path, model_name):
        model_file = Path(path+'/'+model_name)
        if model_file.exists() :
            state_dict = torch.load(open(model_file, "rb"),map_location = lambda storage, loc: storage)
            self.model.load_state_dict(state_dict)
            print("Load model:{} success".format(model_name))
        else :
            raise FileNotFoundError
    def process(self) :
        a, b = self.grid_search()
        return a, b

