#!/bin/bash
input_file=$1
div=$2
batch_size=$3
model_path=$4
scale=$5
damp=$6
config_file=$7
tmp=`echo ${input_file%_*}`
file_profix=$tmp
division=$div
tmp_dir=$file_profix"_tmp_dir" 
res_dir=$file_profix"_res_dir"
mkdir $tmp_dir
echo $division
split -l $division  $input_file -d -a 1 ./$tmp_dir/"train_"
num=0
declare -a Devices
#Devices=("0" "0" "1" "1" "2" "2" "3" "3") 
Devices=("0" "0" "1" "1") 
for file in `ls $tmp_dir`
do
    mkdir $tmp_dir'/'$file'outdir'
    python IF_calc1.py '--config_file' $config_file '--val_file' $tmp_dir'/'$file '--train_file' $input_file\
    					'--model_path' $model_path '--batch_size' $batch_size\
    					'--scale' $scale '--damp' $damp\
    					 '--outdir' $tmp_dir'/'$file'outdir' '--gpu' ${Devices[$num]} &
    ((num=num+1))
done
wait
mkdir $tmp_dir'/outdir_all'
python rename.py $tmp_dir $division $tmp_dir'/all_outdir'
for file in `ls $tmp_dir'/'*'outdir/'*'s_test'`
do
    mv $file $tmp_dir'/outdir_all'
done
rm -r $tmp_dir/*'train'*
