import os
import logging
from time import gmtime, strftime
from pathlib import Path
import json

import torch
from torch import optim
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler

from Clip.clip import load, create_model
from Clip.model import convert_weights, CLIP
from training.train import train, evaluate
from training.data import get_data
from training.params import parse_args
from training.scheduler import cosine_lr


def main(args):
    # 代码魔改后，只支持单卡GPU训练
    assert torch.cuda.is_available()

    # get the name of the experiments
    if args.name is None:
        args.name = strftime(
            f"lr={args.lr}_"
            f"wd={args.wd}_"
            f"model={args.model}_"
            f"batchsize={args.batch_size}_date=%Y-%m-%d-%H-%M-%S",
            gmtime(),
        )

    args.log_path = os.path.join(args.logs, args.name, "out.log")
    if os.path.exists(args.log_path):
        print("Error. Experiment already exists. Use --name {} to specify a new experiment.")
        return -1

    assert args.precision in ['amp', 'fp16', 'fp32']

    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    logging.basicConfig(filename=args.log_path, format='%(levelname)s: %(message)s', level=logging.INFO)

    # # 比赛方提供的basline的模型加载方式，分别加载visual模型和bert模型
    # model_config_file = Path(__file__).parent / f"model_configs/{args.model.replace('/', '-')}.json"
    # print('Loading model from', model_config_file)
    # assert os.path.exists(model_config_file)
    #
    # with open(model_config_file, 'r') as f:
    #     model_info = json.load(f)
    # model = CLIP(**model_info).cuda()
    #
    # if args.visual_weight_path is not None:
    #     assert os.path.exists(args.visual_weight_path), "Pretrained VISUAL weight not exists!"
    # if args.bert_weight_path is not None:
    #     assert os.path.exists(args.bert_weight_path), "Pretrained BERT weight not exists!"
    # load(model, clip_path=args.clip_weight_path, bert_path=args.bert_weight_path)

    # 加载 cn_clip model: https://github.com/OFA-Sys/Chinese-CLIP
    # 加载优先级：resume -> clip_weight_path -> None
    checkpoint = None
    if args.clip_weight_path is not None:
        assert os.path.exists(args.clip_weight_path), "Pretrained CLIP weight not exists!"
        if args.resume is None:
            print('loading pretrained clip model...')
            checkpoint = torch.load(args.clip_weight_path)
        else:
            print('loading resume model...')

    model = create_model(args.model, checkpoint).cuda()

    if args.precision == "amp" or args.precision == "fp32":
        for p in model.parameters():
            p.data = p.data.float()
            if p.grad:
                p.grad.data = p.grad.data.float()
    if args.precision == "fp16":
        convert_weights(model)

    data = get_data(args, args.context_length)

    exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n: not exclude(n)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        total_steps = data["train"].num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        # scheduler = None

    scaler = GradScaler() if args.precision == "amp" else None

    start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    cudnn.deterministic = False

    for epoch in range(start_epoch, args.epochs):
        train(model, data, epoch, optimizer, scaler, scheduler, args)
        if args.val and args.val_data is not None:
            # evaluate未修改，可能跑不了，要验证就自己再改改
            evaluate(model, data, epoch + 1)
        # Saving checkpoints.
        if (epoch + 1) == args.epochs or ((epoch + 1) % args.save_frequency == 0):
            torch.save(
                {
                    "epoch": epoch + 1,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"),
            )
            print('save model successfully!')


if __name__ == "__main__":
    args = parse_args()
    main(args)