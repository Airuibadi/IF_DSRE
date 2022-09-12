import sys, os
import shutil
from copy import deepcopy
import json
import random
import subprocess, time
import numpy as np
import torch 
from .sentence_re import SentenceRE
from .data_loader import SentenceRELoader
from .IF_hyperparameter_identifier import IFSentenceRE_check
from collections import defaultdict

class MeanTeacherDenoise() :
    def __init__(self,
                model,
                clean_pos_path,
                noisy_pos_path,
                clean_NA_path,
                noisy_NA_path,
                dev_path,
                test_path,
                target_rel_name,
                target_rel_path,
                config_file,
                loop_times = 30) :
        super().__init__()
        self.model = model 
        self.teacher_model = deepcopy(self.model)
        self.teacher_model = deepcopy(self.model)


        self.dev_path = dev_path
        self.test_path = test_path
        with open(clean_pos_path, 'r') as f :
            self.clean_pos_set = [line for line in f]
        with open(noisy_pos_path, 'r') as f :
            self.noisy_pos_set = [line for line in f]
        with open(clean_NA_path, 'r') as f :
            self.clean_NA_set = [line for line in f]
        with open(noisy_NA_path,'r') as f :
            self.noisy_NA_set = [line for line in f]
        self.loop_times = loop_times
        relation = json.load(open(target_rel_path))
        for key in relation :
            if relation[key] != '0' :
                target_rel = key
        self.target_rel = target_rel.replace('/','_')
        self.workspace_path = "./workspace"+self.target_rel
        self.name = target_rel_name
        self.config_file = config_file
        for name, para in model.named_parameters() :
            if para.requires_grad :
                print(para.shape)
        if not os.path.exists(self.workspace_path) :
            os.mkdir(self.workspace_path)
        else :
            shutil.rmtree(self.workspace_path)
            os.mkdir(self.workspace_path)

        self.selected_cleans = []
        self.selected_noisys = []
        self.trustable_count = defaultdict(int)
        self.trustable_result = {}

    def process_conf(self) :
        #===========init==============
        Round0_train_data_path = self.train_data_construction(self.clean_pos_set, [], [], 0)
        self.overfitting_parameters(train_data_path = Round0_train_data_path, target_model = self.teacher_model)
        self.Round0_noisy_data_path = self.indexlization(self.noisy_pos_set, file_suffix = self.name+"0noisy_train.txt")
        instances = []
        with open(self.Round0_noisy_data_path) as f :
            for line in f :
                instances.append(eval(line))
        tmp_test_loader = SentenceRELoader(
                self.Round0_noisy_data_path,
                self.teacher_model.rel2id,
                self.teacher_model.sentence_encoder.tokenize,
                160,
                False)
        result = []
        for batch in tmp_test_loader :
            for i in range(len(batch)) :
                batch[i] = batch[i].cuda()
            out = self.teacher_model(*batch[1:])
            out = torch.nn.functional.softmax(out, 1)
            result += [out[i][1].item() for i in range(len(out))]
        a = np.argsort(result)[::-1]
        b = np.sort(result)[::-1]
        select_clean = []
        for i in range(len(a)//10) :
            select_clean.append(instances[a[i]]['id'])
        #print("===========select_clean:%d============="%len(select_clean))
        #==== in loop =====
        for loop_iter in range(2, 50) :
            print("---------%d----------"%loop_iter)
            trusted, leftout  = self.selected_data_construction(select_clean, [])
            print(len(trusted))
            print(len(leftout))
            train_data_path = self.train_data_construction(self.clean_pos_set, trusted, [], loop_iter)
            noisy_data_path = self.tmp_data_saving(data=leftout, suffix=self.name+str(loop_iter)+"noisy_train.txt")
            self.overfitting_parameters(train_data_path=train_data_path, target_model=self.teacher_model)
            instances = []
            with open(noisy_data_path) as f :
                for line in f :
                    instances.append(eval(line))
            tmp_test_loader = SentenceRELoader(
                noisy_data_path,
                self.teacher_model.rel2id,
                self.teacher_model.sentence_encoder.tokenize,
                160,
                False
            )
            result = []
            for batch in tmp_test_loader :
                for i in range(len(batch)) :
                    batch[i] = batch[i].cuda()
                out = self.teacher_model(*batch[1:])
                out = torch.nn.functional.softmax(out, 1)
                result += [out[i][1].item() for i in range(len(out))]
            a = np.argsort(result)[::-1]
            select_clean = []
            for i in range(len(a)//10) :
                select_clean.append(instances[a[i]]['id'])
            #print("===========select_clean:%d============="%len(select_clean))
            #===========Saving Result============
            saving_id = []
            for key in self.trustable_count :
                if self.trustable_count[key] >= 3 :
                    saving_id.append(key)
            print(len(saving_id))
            self.trustable_result.update({loop_iter:saving_id})    
            json.dump(self.trustable_result, open(self.name+"ByConf_res.json", 'w'))
    def indexlization(self, target_set, file_suffix) :
        ided_set = []
        '''deplicate instances'''
        for ins in target_set :
            if ins not in ided_set :
                ided_set.append(ins)
        ided_set = [eval(i) for i in ided_set]
        for idx, ins in enumerate(ided_set) :
            ins['id']  = idx
        ided_set = [str(ins)+'\n' for ins in ided_set]
        return self.tmp_data_saving(ided_set, suffix = file_suffix)

    def train_data_construction(self, origin_pos, candidate_pos, candidate_NA, idx, num = -1) :
        '''
        pos1 = random.sample(candidate_pos, min(90, len(candidate_pos)))
        pos2 = random.sample(origin_pos, min(len(origin_pos), 100-len(pos1)))
        pos2train = pos1+pos2
        '''
        pos2train = origin_pos+candidate_pos
        #na_tmp = random.sample(candidate_NA, min(10, len(candidate_NA)))
        na_tmp = []
        na_sample = random.sample(self.clean_NA_set, min(len(self.clean_NA_set),len(pos2train)-len(na_tmp)))
        na2train = na_sample + na_tmp
        print(len(pos2train+na2train))
        return self.tmp_data_saving(pos2train+na2train, suffix=self.name+str(idx)+"clean_train.txt")
        #return self.indexlization(pos2train+na2train, file_suffix = self.name+str(idx)+"clean_train.txt")
    def selected_data_construction(self, selected_clean, selected_noisy) :
        leftout = []
        trusted = []
        for idx in selected_clean :
            self.trustable_count[idx] += 1
        with open(self.Round0_noisy_data_path, 'r') as f :
            for line in f :
                if self.trustable_count[eval(line)['id']] < 3 :
                    leftout.append(line)
                else :
                    trusted.append(line)
        return trusted, leftout

    def tmp_data_saving(self, data, suffix = "") :
        with open(self.workspace_path+'/'+suffix, 'w') as f :
            for ins in data :
                f.writelines(ins)
        return self.workspace_path+'/'+suffix


    def overfitting_parameters(self, train_data_path, target_model) :
        ins_num = self.line_counter(train_data_path)
        batch_size = self.batch_size_identifier(ins_num, 10)
        ptf = SentenceRE(target_model, train_data_path, self.dev_path, self.test_path, batch_size = batch_size)
        ptf.train_model_overfitting() 

    def overfitting_parameters_with_ema(self, train_data_path, target_model, ema_model, gamma) :
        ins_num = self.line_counter(train_data_path)
        batch_size = self.batch_size_identifier(ins_num, 10)
        ptf = SentenceRE(target_model, train_data_path, self.dev_path, self.test_path, batch_size = batch_size)
        ptf.train_model_overfitting_with_ema(ema_model = ema_model, gamma = gamma) 

    def batch_size_identifier(self, size, least_epoch) :
        size_candidates = [4, 8, 16, 24, 32, 48, 64, 96, 128, 160]
        batch_size = size / least_epoch
        res = 0
        for num in size_candidates :
            if batch_size > num  :
                res = num
        if res == 0 :
            res = 4
        return res

    def line_counter(self, file) :
        cnt = 0
        with open(file, 'r') as f :
            for line in f :
                cnt += 1
        return cnt


