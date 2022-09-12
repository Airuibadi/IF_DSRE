#!/bin/bash
input_file=$1
div=$2
input_file2=$3
model_path=$4
scale=$5
damp=$6
config_file=$7
tmp=`echo ${input_file%_*}`
file_profix=$tmp
tmp_dir=$file_profix"_tmp_dir" 
res_dir=$file_profix"_res_dir"
num=0
declare -a Devices
Devices=("0" "1" "2" "3")
division=$div
split -l $division  $input_file -d -a 1 ./$tmp_dir/"train_"
for file in `ls $tmp_dir'/train'*`
do
    #echo $file
    mkdir $file'outdir'
    for ((i=0; i<$division; ++i))
    do
        ((index=num*division+i))
        mv $tmp_dir'/outdir_all/'$index'_'* $file'outdir'
    done
    python rename2.py $file'outdir'
    python IF_calc2.py '--config_file' $config_file '--val_file' $file '--train_file' $input_file2\
                        '--model_path' $model_path '--batch_size' $batch_size\
                        '--scale' $scale '--damp' $damp\
                         '--outdir' $file'outdir' '--gpu' ${Devices[$num]} &
    ((num=num+1))
done
wait
rm $tmp_dir'/outdir_all' -r
mkdir $res_dir
python combine_calc3.py $tmp_dir $res_dir

