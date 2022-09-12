# Few Clean Instances Help Denoising Distant Supervision


This repository contains the PyTorch code for the paper
> Yufang Liu*, Ziyin Huang*, Yijun Wang, Changzhi Sun, Man Lan, Yuanbin Wu, Xiaofeng Mou and Ding Wang. [*Few Clean Instances Help Denoising Distant Supervision*](https://). Coling, 2022.

A model-agnostic denoise method for distant supervision relation extraction. We propose a new criterion for clean
instance selection based on influence functions. It collects sample-level evidence for recognizing good instancess 
(which is more informative than loss-level evidence). We also propose a teacher-student mechanism for controlling 
purity of intermediate results when bootstrapping the clean set.

## Setup

**Environment**: One or more multi-GPU node(s) with the following software/libraries installed:
- [python 3.7.11](https://www.python.org/downloads/)
- [PyTorch 1.9.0](https://pytorch.org/)
- [numpy 1.20.3](https://docs.scipy.org/doc/numpy/user/quickstart.html)  
- [nltk 3.7](https://www.nltk.org/install.html)

**Data preprocessing**: We split the dataset to create training data for each relation classification, 
please read the paper for more details. We provide a simple dataset ```business_person_company```(bpc) for example.

## Running
This repository contains the code for 4 different criteria, each directory contains
whole code for one criterion. Here we show the steps to run the code for Assumption 2 
with Teacher-Student style update (ByAssump2_TS).

1. To accelerate the IF calculation process, we split the dataset, run the same code on the different parts 
and collect the results together. So before running the code, specify specific gpu for each process. 
For step 1, if you have 4 gpus, and you want to run 2 process on each gpu, set device info in ```run/run3.step1.sh```as:
    ```angular2
    Devices=("0" "0" "1" "1" "2" "2" "3" "3") 
    ``` 
    and set the div number in ```src/framework/MeanTeacher.py/IF_calc_step1``` as 8(gpu_num * process):
    ```angular2
     a = subprocess.Popen(["./run3_step1.sh", input_file, '%d'%(self.line_counter(file=input_file)//8+1), '%d'%batch_size,
                                            model_path, '%d'%scale, '%f'%damp, '%s'%self.config_file], shell=False)
    ```
    same as for step 2, set device info in ```run/run3.step1.sh``` and div number in ```src/framework/MeanTeacher.py/IF_calc_step2```.

2. Running the code for relation ```business_person_company``` as 
    ```angular2
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --config_file ../configs/default_bpc.cfg
    ```
    check the shell script ```run3_step1.sh``` and ```run3_step2.sh``` have the execute permission,
    if not, permission denied error will be reported. Simple solution is ```chmod +x ${file_name}```
    for these two files.
    
## Output
After running one relation classification, a workspace directory for this relation will be generated.
It contains all the running statistics, including the selected training data for each iteration, 
student model, teacher model, and etc.   
    
## Citation
If you find this code useful in your research, please cite:

```
@inproceedings{IFDSRE,
  title={Few Clean Instances Help Denoising Distant Supervision},
  author={Yufang Liu, Ziyin Huang, Yijun Wang, Changzhi Sun, Man Lan, Yuanbin Wu, Xiaofeng Mou and Ding Wang},
  booktitle={The 29th International Conference on Computational Linguistics},
  year={2022}
}
```