### baseline缩减版代码

[比赛链接](https://tianchi.aliyun.com/competition/entrance/532031/introduction)


根据比赛提供的[image-retrieval-baseline](https://github.com/MUGE-2021/image-retrieval-baseline)修改，
baseline的缩减版本，删掉了很多功能，代码简洁易读。与官方最大的不同是我是以原图png格式进行训练和预测。代码魔改后只支持单卡GPU训练。

想跑高分可以看这个库：[Chinese CLIP](https://github.com/OFA-Sys/Chinese-CLIP),
但是我复现不出来他的结果，我很郁闷我的零样本学习结果mean recall只有1%咋回事，于是我就重新跑结果也只有67%，可能是我代码的问题吧，毕竟很多人都复现出来了。


### 代码说明
```
├──Clip    CLIP模型实现，model_configs存放相关模型config文件
├──eval    运行extract_features提取特征，再运行make_topk_predictions获取结果文件
├──preprocess
│   │──transform_images.py 原始代码我内存不够跑不了，修改后转的png文件而不是npz
│   └──transform_openai_pretrain_weights.py  你用cn_clip就忽略这个文件吧，这是转换openai的模型为state_dict
├──training
│   │──pretrain_model 存放预训练模型
│   │──data.py 加载数据
│   │──params.py 参数，注意设置图像路径（包括图像文件夹和jsonl文件）和权重路径
│   │──scheduler.py 学习率策略，用的余弦学习率
│   └──train.py 训练代码
└──README.md
```
