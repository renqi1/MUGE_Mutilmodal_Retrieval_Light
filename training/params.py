import argparse


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["RN50", "RN101", "RN50x4"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name in ["ViT-B-32", "ViT-B-16"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", type=bool, default=False, help="whether to test the model on val-data",)
    parser.add_argument("--logs", type=str, default="../logs/", help="Where to store logs. Use None to avoid storing logs.")
    parser.add_argument("--name", type=str, default=None, help="Optional identifier for the experiment when storing logs. Otherwise use current time.")
    parser.add_argument("--batch-size", type=int, default=96, help="Batch size per GPU.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--warmup", type=int, default=100, help="Number of steps to warmup for.")
    parser.add_argument("--save-frequency", type=int, default=1, help="How often to save checkpoints.")
    parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="amp", help="Floating point precition.")
    parser.add_argument("--model", default="ViT-B-16", help="Name of the vision backbone to use.")
    parser.add_argument("--context-length", type=int, default=24, help="The text length.")
    parser.add_argument(
        "--resume",
        type=str,
        # default='../logs/lr=1e-06_wd=0.001_model=ViT-B-16_batchsize=110_date=2022-11-19-09-24-33/checkpoints/epoch_5.pt',
        default=None,
        help="path to latest checkpoint (default: none)"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="/home/gzz/机械硬盘/sda3/Multimodal_Retrieval/MR_train_queries.jsonl",
        help="Path to jsonl annotation file with training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="/home/gzz/机械硬盘/sda3/Multimodal_Retrieval/MR_valid_queries.jsonl",
        help="Path to jsonl annotation file with validation data",
    )
    parser.add_argument(
        "--train-img",
        type=str,
        default="/home/gzz/机械硬盘/sda3/Multimodal_Retrieval/MR_train_imgs",
        help="Path to file with training images",
    )
    parser.add_argument(
        "--val-img",
        type=str,
        default="/home/gzz/机械硬盘/sda3/Multimodal_Retrieval/MR_valid_imgs",
        help="Path to file with validation images",
    )
    parser.add_argument(
        "--clip-weight-path",
        default="./pretrain_model/clip_cn_vit-b-16.pt",
        type=str,
        help="The path of openai pretrained weight, used to initialize CLIP, should be set to None if you do not use pretrained CLIP",
    )
    # 分别加载visual模型和bert模型
    # parser.add_argument(
    #     "--visual-weight-path",
    #     default="./pretrain_model/ViT-B-16.state_dict.pt",
    #     type=str,
    #     help="The path of openai pretrained weight, used to initialize the image encoder, should be set to None if you do not use pretrained VISUAL",
    # )
    # parser.add_argument(
    #     "--bert-weight-path",
    #     default="./pretrain_model/pytorch_model.bin",
    #     type=str,
    #     help="The path of bert pretrained weight, used to initialize the text encoder, should be set to None if you do not use pretrained BERT",
    # )

    args = parser.parse_args()

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
