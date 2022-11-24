# -*- coding: utf-8 -*-
'''
This script extracts image and text features for evaluation. (with single-GPU)
'''

import os
import argparse
import json
import torch
from tqdm import tqdm
from Clip.clip import create_model
from data import get_eval_img_dataset, get_eval_txt_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-data', 
        type=str, 
        default="/home/gzz/机械硬盘/sda3/Multimodal_Retrieval/MR_test_imgs",
        help="If --extract-image-feats is True, specify the path of processed image file."
    )
    parser.add_argument(
        '--text-data', 
        type=str, 
        default="/home/gzz/机械硬盘/sda3/Multimodal_Retrieval/MR_test_queries.jsonl",
        help="If --extract-text-feats is True, specify the path of test query jsonl file."
    )
    parser.add_argument(
        "--clip-weight-path",
        default="../logs/lr=5e-05_wd=0.001_model=ViT-B-16_batchsize=96_date=2022-11-20-11-00-51/checkpoints/epoch_3.pt",
        type=str,
        help="The path of openai pretrained weight",
    )
    parser.add_argument(
        '--image-feat-output-path',
        type=str,
        default=None,
        help="If --extract-image-feats is True, specify the path of output image features."
    )
    parser.add_argument(
        '--text-feat-output-path',
        type=str,
        default=None,
        help="If --extract-text-feats is True, specify the path of output text features."
    )
    parser.add_argument('--extract-image-feats', default=True, help="Whether to extract image features.")
    parser.add_argument('--extract-text-feats', default=True, help="Whether to extract text features.")
    parser.add_argument("--img-batch-size", type=int, default=32, help="Image batch size.")
    parser.add_argument("--text-batch-size", type=int, default=32, help="Text batch size.")
    parser.add_argument("--context-length", type=int, default=24, help="The text length.")
    parser.add_argument("--model", default="ViT-B-16", help="Name of the vision backbone to use.")

    args = parser.parse_args()

    return args    


if __name__ == "__main__":
    args = parse_args()

    assert os.path.exists(args.clip_weight_path), "Pretrained CLIP weight not exists!"

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"{name}: {val}")

    # Initialize the model.
    checkpoint = torch.load(args.clip_weight_path, map_location='cpu')
    model = create_model(args.model, checkpoint).cuda()

    # Make inference for images
    if args.extract_image_feats:
        print("Preparing image inference dataset.")
        img_data = get_eval_img_dataset(args)
        print('Make inference for images...')
        if args.image_feat_output_path is None:
            args.image_feat_output_path = "{}.img_feat.jsonl".format(args.image_data)
        write_cnt = 0
        with open(args.image_feat_output_path, "w") as fout:
            model.eval()
            with torch.no_grad():
                for batch in tqdm(img_data):
                    image_ids, images = batch
                    images = images.cuda(non_blocking=True)
                    image_features = model(images, None)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    for image_id, image_feature in zip(image_ids, image_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"image_id": image_id, "feature": image_feature})))
                        write_cnt += 1
        print('{} image features are stored in {}'.format(write_cnt, args.image_feat_output_path))


    # Make inference for texts
    if args.extract_text_feats:
        print("Preparing text inference dataset.")
        text_data = get_eval_txt_dataset(args, max_txt_length=args.context_length)
        print('Make inference for texts...')
        if args.text_feat_output_path is None:
            args.text_feat_output_path = "{}.txt_feat.jsonl".format(args.text_data[:-6])
        write_cnt = 0
        with open(args.text_feat_output_path, "w") as fout:
            model.eval()
            with torch.no_grad():
                for batch in tqdm(text_data):
                    query_ids, texts = batch
                    texts = texts.cuda(non_blocking=True)
                    text_features = model(None, texts)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    for query_id, text_feature in zip(query_ids.tolist(), text_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"query_id": query_id, "feature": text_feature})))
                        write_cnt += 1
        print('{} query text features are stored in {}'.format(write_cnt, args.text_feat_output_path))
    
    print("Done!")