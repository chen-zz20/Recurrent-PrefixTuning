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

jittor 环境：
```
conda create -n jittor --clone prefixtuning
conda activate jittor
python3.7 -m pip install jittor
```

切换transformer版本（原始的pytorch版本和修改一部分的jittor版本）
```
cd ./
./change.sh pytorch #切换到pytorch版本
./change.sh jittor #切换到jittor版本
```

## 文本生成任务
### 原始论文复现

单个模型的训练以及生成
```
conda activate prefixtuning
cd ./
./change.sh pytorch #transformer切换到pytorch版本
cd recurrence/PrefixTuning/gpt2
python train_e2e.py --mode YourDataSet --notes GPTModelSize
```

8个模型（2个数据集与4个GPT2预训练模型）训练的脚本实现
```
cd ./
./change.sh pytorch #transformer切换到pytorch版本
cd recurrence/PrefixTuning/gpt2
./run.sh
```

模型评分

E2E数据集的评分可以在执行上述命令下自动完成，webNLG数据集的评分需要手动完成。具体操作如下：
```
cp txt ./score/dart/evaluation/example/new_score.txt #把需要评分的txt文件复制到上述目录下的new_score.txt文件中
cd ./score/dart/evaluation
./run_eval_on_webnlg.sh
```
最后输出的result.txt文件即为评分文件。


### GPT的计图实现

GPT2模型转换为jittor能够读取的state-dict模型
```
conda activate jittor
cd ./
python state-dict-convert.py --name gpt2-XXX #单个文件的实现（需对4个不同大小的gpt2模型进行处理）
./state2dict.sh #或者直接用脚本对gpt2的四个不同大小的模型做预处理
```

训练文本生成模型：
```
conda activate jittor
cd ./
./change.sh jittor #将transformer版本切换为jittor版本
cd ./recurrence/MyGptGeneration/codes
python main.py --name YourModelName --pretain_model gpt2-XXX
```

文本生成：
```
conda activate jittor
cd ./
./change.sh jittor #将transformer版本切换为jittor版本
cd ./recurrence/MyGptGeneration/codes
python main.py --test YourModelName
```
