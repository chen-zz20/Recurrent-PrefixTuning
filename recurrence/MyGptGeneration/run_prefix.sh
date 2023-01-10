#!/bin/bash
source ~/../lihe/anaconda3/bin/activate
conda activate HW3
echo "`date`"
echo "Training Start"
echo "Prefix 32"
python main.py --name gpt-prefix32        --num_epochs 30 --prefix 32 --model_config config.json        --pretrain_dir pretrain/gpt2-paras.jt        &>gpt_prefix32_train.info
echo "`date`"  ": gpt-prefix32 Done"
python main.py --name gpt-medium-prefix32 --num_epochs 30 --prefix 32 --model_config config_medium.json --pretrain_dir pretrain/gpt2-medium-paras.jt &>gpt_medium_prefix32_train.info
echo "`date`"  ": gpt-medium-prefix32 Done"
python main.py --name gpt-large-prefix32  --num_epochs 30 --prefix 32 --model_config config_large.json  --pretrain_dir pretrain/gpt2-large-paras.jt  &>gpt_large_prefix32_train.info --batch_size 16
echo "`date`"  ": gpt-large-prefix32 Done"
python main.py --name gpt-xl-prefix32     --num_epochs 30 --prefix 32 --model_config config_xl.json     --pretrain_dir pretrain/gpt2-xl-paras.jt     &>gpt_xl_prefix32_train.info --batch_size 16 
echo "`date`"  ": gpt-xl-prefix32 Done"
echo "evaluation Start"
python main.py --test gpt-prefix32        &>gpt_prefix32_out.info
echo "`date`"  ": gpt-prefix32 Done"
python main.py --test gpt-medium-prefix32 &>gpt_medium_prefix32_out.info
echo "`date`"  ": gpt-medium-prefix32 Done"
python main.py --test gpt-large-prefix32  &>gpt_large_prefix32_out.info --batch_size 16
echo "`date`"  ": gpt-large-prefix32 Done"
python main.py --test gpt-xl-prefix32     &>gpt_xl_prefix32_out.info --batch_size 16 
echo "`date`"  ": gpt-xl-prefix32 Done" 
echo "Prefix 128"
python main.py --name gpt-prefix128        --num_epochs 30 --prefix 128 --model_config config.json        --pretrain_dir pretrain/gpt2-paras.jt        &>gpt_prefix128_train.info
echo "`date`"  ": gpt-prefix128 Done"
python main.py --name gpt-medium-prefix128 --num_epochs 30 --prefix 128 --model_config config_medium.json --pretrain_dir pretrain/gpt2-medium-paras.jt &>gpt_medium_prefix128_train.info
echo "`date`"  ": gpt-medium-prefix128 Done"
python main.py --name gpt-large-prefix128  --num_epochs 30 --prefix 128 --model_config config_large.json  --pretrain_dir pretrain/gpt2-large-paras.jt  &>gpt_large_prefix128_train.info --batch_size 16
echo "`date`"  ": gpt-large-prefix128 Done"
python main.py --name gpt-xl-prefix128     --num_epochs 30 --prefix 128 --model_config config_xl.json     --pretrain_dir pretrain/gpt2-xl-paras.jt     &>gpt_xl_prefix128_train.info --batch_size 16 
echo "`date`"  ": gpt-xl-prefix128 Done"
echo "evaluation Start"
python main.py --test gpt-prefix128        &>gpt_prefix128_out.info
echo "`date`"  ": gpt-prefix128 Done"
python main.py --test gpt-medium-prefix128 &>gpt_medium_prefix128_out.info
echo "`date`"  ": gpt-medium-prefix128 Done"
python main.py --test gpt-large-prefix128  &>gpt_large_prefix128_out.info --batch_size 16
echo "`date`"  ": gpt-large-prefix128 Done"
python main.py --test gpt-xl-prefix128     &>gpt_xl_prefix128_out.info --batch_size 16 
echo "`date`"  ": gpt-xl-prefix128 Done" 
echo "FineTuning"
python main.py --name gpt-ftune32        --num_epochs 30 --model_config config.json        --pretrain_dir pretrain/gpt2-paras.jt        &>gpt_ftune32_train.info
echo "`date`"  ": gpt-ftune32 Done"
python main.py --name gpt-medium-ftune32 --num_epochs 30 --model_config config_medium.json --pretrain_dir pretrain/gpt2-medium-paras.jt &>gpt_medium_ftune32_train.info
echo "`date`"  ": gpt-medium-ftune32 Done"
python main.py --name gpt-large-ftune32  --num_epochs 30 --model_config config_large.json  --pretrain_dir pretrain/gpt2-large-paras.jt  &>gpt_large_ftune32_train.info --batch_size 16
echo "`date`"  ": gpt-large-ftune32 Done"
python main.py --name gpt-xl-ftune32     --num_epochs 30 --model_config config_xl.json     --pretrain_dir pretrain/gpt2-xl-paras.jt     &>gpt_xl_ftune32_train.info --batch_size 16 
echo "`date`"  ": gpt-xl-ftune32 Done"
echo "evaluation Start"
python main.py --test gpt-ftune32        &>gpt_ftune32_out.info
echo "`date`"  ": gpt-ftune32 Done"
python main.py --test gpt-medium-ftune32 &>gpt_medium_ftune32_out.info
echo "`date`"  ": gpt-medium-ftune32 Done"
python main.py --test gpt-large-ftune32  &>gpt_large_ftune32_out.info --batch_size 16
echo "`date`"  ": gpt-large-ftune32 Done"
python main.py --test gpt-xl-ftune32     &>gpt_xl_ftune32_out.info --batch_size 16 
echo "`date`"  ": gpt-xl-ftune32 Done"