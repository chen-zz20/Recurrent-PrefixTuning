# PrefixTuning

## 项目简介
《人工神经网络》课程大作业代码展示，尝试复现论文《Prefix-Tuning: Optimizing Continuous Prompts for Generation》并将其转化为jittor版本。

## 文件：
    .
    |—— gpt-state-dict # 存放可供jittor识别使用的state_dict模型文件
    |—— gpt2-model #下载存放的gpt2模型所在位置，因预训练模型过大，为空文件。训练时自行下载
    |—— new-transformers #存放原始的transformer文件和相应的jittor版本的transformer文件
    |—— PrefixTuning #原始论文的原始代码，来自https://github.com/XiangLi1999/PrefixTuning
    |—— recurrence #复现论文所在文件夹
        |—— PrefixTuning #复现的原始论文（pytorch版本）
        |—— MyGptGeneration #基于HW3文本生成任务的jittor实现，并附加了prefixtuning机制
        |—— result #论文复现结果
    |—— score #给e2e-metrics数据集以及webnlg数据集打分
        |—— e2e-metrics #代码来自https://github.com/tuetschek/e2e-metrics
        |—— dart #代码来自https://github.com/Yale-LILY/dart
    |—— try #复现过程中的一些失败的尝试
        |—— PrefixTuning #最初试图直接改动transformer库的版本
        |—— HW3 #试图添加进webnlg数据集、E2E数据集的手写GPT版本
    |—— change.sh #用于切换transformer版本，选择其版本是jittor版本还是pytorch版本
    |—— state-dict-convert.py #将下载的基于pytorch的model文件转化为可供jittor识别使用的state_dict文件
    |—— README.md

## 环境配置
prefixtuning 环境：
```
cd PrefixTuning/transformers/
conda create -n prefixtuning python==3.7.6 #注意python版本必须是3.7否则论文无法复现！
codna activate prefixtuning
python3.7 -m pip install -e .[torch]
conda install nltk==3.5
cd ../../score/e2e-metrics
python3.7 -m pip install -r requirements.txt
```

HW3 环境：
```
conda create -n HW3 --clone prefixtuning
conda activate HW3
python3.7 -m pip install jittor
```

切换transformer版本（原始的pytorch版本和修改一部分的jittor版本）
```
cd ./
./change.sh pytorch #切换到pytorch版本
./change.sh jittor #切换到jittor版本
```

## 文本生成任务
训练文本生成模型：
```
cd ./recurrence/MyGptGeneration/codes
python main.py --name YourModelName --model_config YourModelConfig --pretrain_dir ../../../gpt-state-dict/YourModelPretrainFile
```
文本生成：
```
cd ./recurrence/MyGptGeneration/codes
python main.py --test YourModelName
```
