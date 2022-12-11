import os
import time
import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast
import logging


def get_loss(model, images, texts, loss_img, loss_txt):
    image_features, text_features, logit_scale = model(images, texts)
    logit_scale = logit_scale.mean()

    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long().cuda(non_blocking=True)

    total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
    return total_loss


def train(model, data, epoch, optimizer, scaler, scheduler, args):

    model.train()

    dataloader = data['train']

    loss_img = nn.CrossEntropyLoss().cuda()
    loss_txt = nn.CrossEntropyLoss().cuda()

    num_batches_per_epoch = dataloader.num_batches
    prev_time = time.time()
    for i, (images, texts) in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        optimizer.zero_grad()

        images = images.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss = get_loss(model, images, texts, loss_img, loss_txt)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss = get_loss(model, images, texts, loss_img, loss_txt)
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        # Determine approximate time left one epoch
        time_left = datetime.timedelta(seconds=(num_batches_per_epoch - i) * (time.time() - prev_time))
        prev_time = time.time()

        if (i + 1) % 100 == 0:
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

        if (i % 10) == 0:
            num_samples = i * len(images)
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%) {time_left}]\t"
                f"Loss: {total_loss.item():.6f}\t"
                f"LR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {model.logit_scale.data:.3f}"
            )
            print(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%) {time_left}]\t"
                f"Loss: {total_loss.item():.6f}\t"
                f"LR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {model.logit_scale.data:.3f}"
            )


def evaluate(model, data, epoch):
    # 验证应该很费时，就没验证，还没改好，可能跑不了，自己改吧
    logging.info(f"Begin to eval epoch: {epoch}...")
    
    model.eval()

    dataloader = data['val']

    loss_img = nn.CrossEntropyLoss().cuda()
    loss_txt = nn.CrossEntropyLoss().cuda()

    cumulative_loss = 0.0
    num_elements = 0.0
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for images, texts in tqdm(dataloader):

            images = images.cuda(non_blocking=True)
            texts = texts.cuda(non_blocking=True)
            image_features, text_features, logit_scale = model(images, texts)
            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(images)).long()
            ground_truth = ground_truth.cuda(non_blocking=True)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

        metrics = {}
        loss = cumulative_loss / num_elements
        metrics.update(
            **{"val_loss": loss.item(), "epoch": epoch, "num_elements": num_elements}
        )

        logging.info(
            f"Eval Epoch: {epoch} "
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

    return metrics
