### baseline缩减版代码

[比赛链接](https://tianchi.aliyun.com/competition/entrance/532031/introduction)

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
