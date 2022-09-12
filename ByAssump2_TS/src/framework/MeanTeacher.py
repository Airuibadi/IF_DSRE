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
                loop_times = 50) :
        super().__init__()
        self.model = model 
        self.teacher_model = deepcopy(self.model)
        self.student_model = deepcopy(self.model)

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

        self.Round0_noisy_data_path = self.indexlization(target_set = self.noisy_pos_set, file_suffix = self.name+'0noisy_train.txt')
        self.selected_cleans = []
        self.selected_noisys = []
        self.trustable_count = defaultdict(int)
        self.trustable_result = {}
        #self.init_train()

    def process(self, mode = 2) :
        selected_clean, selected_noisy = self.init_train()
        self.selected_cleans.append(selected_clean)
        self.selected_noisys.append(selected_noisy)

        print("Mean teacher loop start")
        for loop_iter in range(2, self.loop_times+1) :
            print("Mean teacher loop: %d"%(loop_iter))
            if mode == 0 :
                for idx in selected_clean :
                    self.trustable_count[idx] += 1
                leftout = []
                trusted = []
                with open(self.Round0_noisy_data_path, 'r') as f :
                    for line in f :
                        if self.trustable_count[eval(line)['id']] < 3 :
                            leftout.append(line)
                        else :
                            trusted.append(line)
                noisy_data_path = self.tmp_data_saving(data=leftout, suffix=self.name+str(loop_iter)+"noisy_train.txt")
                train_data_path = self.train_data_construction(self.clean_pos_set, trusted, [], loop_iter)
                self.student_model = deepcopy(self.teacher_model)
                self.overfitting_parameters(train_data_path = train_data_path, target_model = self.student_model)
                torch.save(self.teacher_model.state_dict(), open(os.path.join(self.workspace_path, "teacher"+str(loop_iter)+".model"), "wb"))
                torch.save(self.student_model.state_dict(), open(os.path.join(self.workspace_path, "student"+str(loop_iter)+".model"), "wb"))
                ins_num = self.line_counter(file = train_data_path)
                IFcf = IFSentenceRE_check(self.student_model, train_path = train_data_path, 
                                            batch_size = self.batch_size_identifier(ins_num, 40))
                scale, damp = IFcf.process()
                self.IF_calc_step1(input_file = train_data_path, batch_size = self.batch_size_identifier(ins_num, 40),
                                        model_path=self.workspace_path+ "/student"+str(loop_iter)+".model", scale=scale, damp=damp)
                self.IF_calc_step2(input_file = train_data_path, input_file2 = noisy_data_path,
                                    model_path=self.workspace_path+ "/student"+str(loop_iter)+".model", scale=scale, damp=damp)
                selected_clean, selected_noisy = self.get_results(IF_value_matrix=self.workspace_path+"/"+self.name+str(loop_iter)+"clean_res_dir/IF_value_result.npy", 
                                clean_set_path=train_data_path, noisy_set_path=noisy_data_path)
            elif mode == 1 :
                candidate_pos, candidate_NA, leftout, trusted = self.selected_data_construction(selected_clean, selected_noisy)
                train_data_path = self.train_data_construction(self.clean_pos_set, trusted, [], loop_iter)
                noisy_data_path = self.tmp_data_saving(data=leftout, suffix=self.name+str(loop_iter)+"noisy_train.txt") 
                #self.student_model = deepcopy(self.model)
                self.overfitting_parameters(train_data_path = train_data_path, target_model = self.student_model)
                torch.save(self.teacher_model.state_dict(), open(os.path.join(self.workspace_path, "teacher"+str(loop_iter)+".model"), "wb"))
                torch.save(self.student_model.state_dict(), open(os.path.join(self.workspace_path, "student"+str(loop_iter)+".model"), "wb"))
                ins_num = self.line_counter(file = train_data_path)
                IFcf = IFSentenceRE_check(self.student_model, train_path = train_data_path, 
                                            batch_size = self.batch_size_identifier(ins_num, 40))
                scale, damp = IFcf.process()
                self.IF_calc_step1(input_file = train_data_path, batch_size = self.batch_size_identifier(ins_num, 40),
                                        model_path=self.workspace_path+ "/student"+str(loop_iter)+".model", scale=scale, damp=damp)
                self.IF_calc_step2(input_file = train_data_path, input_file2 = noisy_data_path,
                                    model_path=self.workspace_path+ "/student"+str(loop_iter)+".model", scale=scale, damp=damp)
                selected_clean, selected_noisy = self.get_results(IF_value_matrix=self.workspace_path+"/"+self.name+str(loop_iter)+"clean_res_dir/IF_value_result.npy", 
                                clean_set_path=train_data_path, noisy_set_path = noisy_data_path)
            elif mode == 2 :
                candidate_pos, candidate_NA, leftout, trusted = self.selected_data_construction(selected_clean, selected_noisy)
                train_data_path = self.train_data_construction(self.clean_pos_set, trusted, [], loop_iter)
                noisy_data_path = self.tmp_data_saving(data=leftout, suffix=self.name+str(loop_iter)+"noisy_train.txt") 
                if loop_iter < 3 :
                    self.overfitting_parameters_with_ema(train_data_path = train_data_path, target_model = self.student_model, ema_model = self.teacher_model, gamma = 0.0)
                    self.update_ema_variables(model = self.student_model, ema_model = self.teacher_model, step = loop_iter, alpha = 0.01)
                else :
                    self.overfitting_parameters_with_ema(train_data_path = train_data_path, target_model = self.student_model, ema_model = self.teacher_model, gamma = 1.0)
                    self.update_ema_variables(model = self.student_model, ema_model = self.teacher_model, step = loop_iter, alpha = 0.95)
                torch.save(self.teacher_model.state_dict(), open(os.path.join(self.workspace_path, "teacher"+str(loop_iter)+".model"), "wb"))
                torch.save(self.student_model.state_dict(), open(os.path.join(self.workspace_path, "student"+str(loop_iter)+".model"), "wb"))

                ins_num = self.line_counter(file = train_data_path)
                IFcf = IFSentenceRE_check(self.student_model, train_path = train_data_path, 
                                            batch_size = self.batch_size_identifier(ins_num, 40))
                scale, damp = IFcf.process()
                self.IF_calc_step1(input_file = train_data_path, batch_size = self.batch_size_identifier(ins_num, 40),
                                        model_path=self.workspace_path+ "/student"+str(loop_iter)+".model", scale=scale, damp=damp)
                self.IF_calc_step2(input_file = train_data_path, input_file2 = noisy_data_path,
                                    model_path=self.workspace_path+ "/student"+str(loop_iter)+".model", scale=scale, damp=damp)
                selected_clean, selected_noisy = self.get_results(IF_value_matrix=self.workspace_path+"/"+self.name+str(loop_iter)+"clean_res_dir/IF_value_result.npy", 
                                clean_set_path=train_data_path, noisy_set_path = noisy_data_path)

            self.selected_cleans.append(selected_clean)
            self.selected_noisys.append(selected_noisy)

            _dict = {0:self.selected_cleans, 1:self.selected_noisys}
            tempa_res = []
            for key in self.trustable_count :
                if self.trustable_count[key] >= 3 :
                    tempa_res.append(key)
            self.trustable_result.update({loop_iter:tempa_res})
            '''
            if loop_iter != 1 and self.trustable_result[loop_iter] != self.trustable_result[loop_iter-1] :
                for key in self.trustable_count :
                    if self.trustable_count[key] < 3 :
                        self.trustable_count[key] = 0
            '''
            json.dump(self.trustable_result, open("./trustable_"+self.name+".json", 'w'))
            #json.dump(_dict, open("./res_bpc_all_50.json", 'w'))
    

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
        pos1 = random.sample(candidate_pos, min(50, len(candidate_pos)))
        pos2 = random.sample(origin_pos, min(len(origin_pos), 100-len(pos1)))
        pos2train = pos1+pos2
        #na_tmp = random.sample(candidate_NA, min(10, len(candidate_NA)))
        na_tmp = []
        na_sample = random.sample(self.clean_NA_set, min(len(self.clean_NA_set),len(pos2train)-len(na_tmp)))
        na2train = na_sample + na_tmp
        return self.tmp_data_saving(pos2train+na2train, suffix=self.name+str(idx)+"clean_train.txt")

    def selected_data_construction(self, selected_clean, selected_noisy) :
        candidate_pos = []
        candidate_NA = []
        leftout = []
        trusted = []
        for idx in selected_clean :
            self.trustable_count[idx] += 1
        with open(self.Round0_noisy_data_path, 'r') as f :
            for line in f :
                '''
                if eval(line)['id'] in selected_noisy and eval(line)['id'] not in selected_clean :
                    candidate_NA.append(line)
                elif eval(line)['id'] in selected_clean and eval(line)['id'] not in selected_noisy :
                    candidate_pos.append(line)      
                else :
                '''
                #WARNNING!
                if self.trustable_count[eval(line)['id']] < 3 :
                    leftout.append(line)
                else :
                    trusted.append(line)
        tmp = [eval(ins) for ins in candidate_NA]
        for ins in tmp :
            ins['relation'] = 'NA'
        candidate_NA = [str(ins)+'\n' for ins in tmp]
        return candidate_pos, candidate_NA, leftout, trusted

    def tmp_data_saving(self, data, suffix = "") :
        with open(self.workspace_path+'/'+suffix, 'w') as f :
            for ins in data :
                f.writelines(ins)
        return self.workspace_path+'/'+suffix

    def init_train(self) :
        print("Initializing process start")
        
        Round0_train_data_path = self.train_data_construction(self.clean_pos_set, [], [], 0)

        self.overfitting_parameters(train_data_path = Round0_train_data_path, target_model = self.teacher_model)
        self.student_model = deepcopy(self.teacher_model)
        torch.save(self.teacher_model.state_dict(), open(os.path.join(self.workspace_path, "teacher0.model"), "wb"))
        torch.save(self.student_model.state_dict(), open(os.path.join(self.workspace_path, "student0.model"), "wb"))

        ins_num = self.line_counter(file = Round0_train_data_path)
        IFcf = IFSentenceRE_check(self.student_model, train_path = Round0_train_data_path, 
                                    batch_size = self.batch_size_identifier(ins_num, 50))
        scale, damp = IFcf.process()
        self.IF_calc_step1(input_file = Round0_train_data_path, batch_size = self.batch_size_identifier(ins_num, 50),
                                model_path=self.workspace_path+ "/student0.model", scale=scale, damp=damp)
        self.IF_calc_step2(input_file = Round0_train_data_path, input_file2 = self.Round0_noisy_data_path,
                            model_path=self.workspace_path+ "/student0.model", scale=scale, damp=damp)
        return self.get_results(IF_value_matrix=self.workspace_path+"/"+self.name+"0clean_res_dir/IF_value_result.npy", 
                            clean_set_path=Round0_train_data_path, noisy_set_path=self.Round0_noisy_data_path)

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

    def IF_calc_step1(self, input_file, batch_size, model_path, scale, damp) :
        a = subprocess.Popen(["./run3_step1.sh", input_file, '%d'%(self.line_counter(file=input_file)//8+1), '%d'%batch_size,
                                        model_path, '%d'%scale, '%f'%damp, '%s'%self.config_file], shell=False)
        while a.poll() is None:
            time.sleep(1)
        return a

    def IF_calc_step2(self, input_file, input_file2, model_path, scale, damp) :
        a = subprocess.Popen(["./run3_step2.sh", input_file, '%d'%(self.line_counter(file=input_file)//8+1), input_file2,
                            model_path, '%d'%scale, '%f'%damp, '%s'%self.config_file], shell=False)
        while a.poll() is None:
            time.sleep(1)
        return a

    def get_results(self, IF_value_matrix, clean_set_path, noisy_set_path) :
        matrix = np.load(IF_value_matrix)
        clean_data = []
        noisy_data = []
        with open(clean_set_path, 'r') as f :
            for line in f :
                clean_data.append(eval(line))
        with open(noisy_set_path, 'r') as f :
            for line in f :
                noisy_data.append(eval(line))
        
        select_num = len(noisy_data) // 10
        assert len(clean_data) == matrix.shape[1]
        assert len(noisy_data) == matrix.shape[0]

        print(matrix.shape)
        res_inf = matrix.sum(axis=1).tolist()
        assert len(res_inf) == len(noisy_data)
        harmful = np.argsort(res_inf)
        helpful = harmful[::-1]
        a = np.sort(res_inf)
        clean_pos_idx = []
        clean_neg_idx = []
        noisy_pos_idx = []
        noisy_neg_idx = []
        noisy_true_idx = []
        noisy_false_idx = []
        for idx, ins in enumerate(clean_data) :
            if ins['relation'] == 'NA' :
                clean_neg_idx.append(idx)
            else :
                clean_pos_idx.append(idx)
        for idx, ins in enumerate(noisy_data) :
            if ins['relation'] == 'NA' :
                noisy_neg_idx.append(idx)
            else :
                noisy_pos_idx.append(idx)
        influences = []
        def sigmoid(x):
            return 1/(1+np.exp(-x))
        for picked_line in clean_pos_idx :
            influences.append(matrix[:, picked_line].tolist())
        influences = np.array(influences)
        print(clean_pos_idx)
        print(influences.shape)
        print(len(clean_pos_idx))
        res_inf = influences.sum(axis=0).tolist()
        harmful = np.argsort(res_inf)
        helpful = harmful[::-1]
        print(len(res_inf))
        selected_noisy = []
        best_close = 10000000
        select_num_pos = len(noisy_data) // 10
        selected_clean = []
        for magic_num in np.arange(-0.01, 0.01, 0.000001) :
            selected_clean_tmp = []
            for idx in range(matrix.shape[0]) :
                flag = True
                for idy in clean_pos_idx :
                    if matrix[idx][idy] < magic_num :
                        flag = False
                        break
                if flag :
                    selected_clean_tmp.append(noisy_data[idx]['id'])
            #print("selected_tmp:%d select_num_pos:%d best:%d magic_num:%f"%(len(selected_clean_tmp), select_num_pos, best_close, magic_num))
            if abs(len(selected_clean_tmp)-select_num_pos) < best_close :
                best_close = abs(len(selected_clean_tmp)-select_num_pos)
                selected_clean = selected_clean_tmp
            elif abs(len(selected_clean_tmp)-select_num_pos) > best_close :
                break
        print("magic_num:%f"%magic_num)
        print(len(selected_clean))
        print(selected_clean)
        interset = []
        for idx in selected_clean :
            if idx in selected_noisy :
                interset.append(idx)
        print(len(interset))
        return selected_clean, selected_noisy

    def update_ema_variables(self, model, ema_model, step, alpha = 0.99):
        # Use the true average until the exponential average is more correct
        #alpha = min(1 - 1 / (step + 10), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)          
