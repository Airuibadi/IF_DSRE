import os, logging, json
from tqdm import tqdm
import numpy
import torch
from torch import nn, optim
from pathlib import Path
from .data_loader import SentenceRELoader
from .utils import AverageMeter, display_progress, save_json
from .influence_function import s_test, grad_z
import numpy as np
np.random.seed(1219)
torch.manual_seed(1219)
torch.cuda.manual_seed_all(1219)
torch.backends.cudnn.deterministic = True
class IFCalc(nn.Module):
    def __init__(self, 
                 model,
                 train_path, 
                 test_path,
                 save_model_path,
                 batch_size=320):
    
        super().__init__()

        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True)
        
        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False
            )
        # Model
        self.model = model
        self.load_model(save_model_path)
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
                # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.save_model_path = save_model_path
        self.test_path = test_path

    def calc_s_test_single(self, ins_data, damp=0.01, scale=25, recursion_depth=100, r=1):
        s_test_vec_list = []
        for i in range(r):
            s_test_vec_list.append(s_test(ins_data, self.model, self.train_loader, self.criterion,
                                         damp=damp, scale=scale,recursion_depth=recursion_depth))

            ################################
            # TODO: Understand why the first[0] tensor is the largest with 1675 tensor
            #       entries while all subsequent ones only have 335 entries?
            ################################
        s_test_vec = s_test_vec_list[0]
        for i in range(1, r):
            s_test_vec += s_test_vec_list[i]
        s_test_vec = [i / r for i in s_test_vec]
        return s_test_vec

    def calc_s_test(self, save_path, recursion_depth, r_averaging, damp, scale) :
        for i in tqdm(range(0, len(self.test_loader.dataset))) :
            ins_data = self.test_loader.dataset[i]
            ins_index = i
            s_test_vec = self.calc_s_test_single(ins_data, damp, scale, 
                        recursion_depth, r_averaging)
            torch.save(s_test_vec,
                    save_path.joinpath(f"{ins_index}_recdep{recursion_depth}_r{r_averaging}.s_test"))
            torch.cuda.empty_cache()
 
    def calc_influence_function(self, train_dataset_size, test_dataset_size,
                                    s_test_dir, recursion_depth, r_averaging):
        def to_flat_tensor(tensors) :
            return torch.cat([t.flatten() for t in tensors])
        def norm(tensors) :
            return torch.norm(to_flat_tensor(tensors))
        def normalize(tensors) :
            norm = torch.norm(to_flat_tensor(tensors))
            return [t/norm for t in tensors]

        available_s_test_files = 0
        for dd in Path(s_test_dir).glob("*.s_test") :
            available_s_test_files += 1
        if available_s_test_files != test_dataset_size :
            logging.warn("Load Influence Data: number of s_test files mismatch the dataset size")
        influences = []
        division = 3000
        for div in range(0, test_dataset_size, division) :
            s_test_vecs = [torch.load(str(s_test_dir)+f"/{i}_recdep{recursion_depth}_r{r_averaging}.s_test") for i in range(div, min(div+division, test_dataset_size))]
            div_influences = []
            for l in tqdm(range(0, train_dataset_size)) :
                ins_data = self.train_loader.dataset[l]
                grad_z_vec = grad_z(ins_data, self.model, self.criterion)
                one_train_point_influence = []
                for i in range(len(s_test_vecs)):
                    tmp_influence = sum(
                            [
                                ###################################
                                # TODO: verify if computation really needs to be done
                                # on the CPU or if GPU would work, too
                                ###################################
                                torch.sum(k * j).data.cpu().numpy()
                                for k, j in zip(grad_z_vec, s_test_vecs[i])
                                ###################################
                                # Originally with [i] because each grad_z contained
                                # a list of tensors as long as e_s_test list
                                # There is one grad_z per training data sample
                                ###################################
                                ]) / train_dataset_size
                    one_train_point_influence.append(tmp_influence)
                div_influences.append(np.array(one_train_point_influence))
            influences.append(div_influences)   
            torch.cuda.empty_cache()
        influences = np.concatenate(influences, axis=1)
        np.save(str(s_test_dir)+'/IF_value_tmp.npy', influences)

    def load_model(self, path):
        model_file = Path(path)
        if model_file.exists() :
            state_dict = torch.load(open(model_file, "rb"),map_location = lambda storage, loc: storage)
            self.model.load_state_dict(state_dict)
            #print("Load model:{} success".format(path))
        else :
            raise FileNotFoundError
    def process_step1(self, s_test_outdir, recursion_depth=100, r_averaging=3, damp=0.01, scale=25) :
        self.calc_s_test(Path(s_test_outdir), recursion_depth, r_averaging, damp, scale)

    def process_step2(self, s_test_outdir, recursion_depth=100, r_averaging=3, damp=0.01, scale=25) :
        self.calc_influence_function(len(self.train_loader.dataset), len(self.test_loader.dataset), 
                                     s_test_outdir, recursion_depth, r_averaging)
