## 文件：
    .
    |—— gen_log # 存放可供jittor识别使用的state_dict模型文件
    |—— train_log #下载存放的gpt2模型所在位置，因预训练模型过大，为空文件。训练时自行下载

## 文件名说明
gpt-large(medium,large,xl): 预训练模型数据名称
prefix32： 32词prefixtuning
prefix128：128词prefixtuning
ftune32: finetuning

## 备注
受显存限制，下列情况未能完成运行：
- gpt-xl-128prefix 文本生成
- gpt-xl-ftune 训练和文本生成
