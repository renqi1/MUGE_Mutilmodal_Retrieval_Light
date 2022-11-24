# -*- coding: utf-8 -*-
'''
I modify the code, not save as npzfile because the cpu memory is not enough to do that,
this code saves the images to specific folder.
'''

import argparse
import os
from PIL import Image
import base64
from io import BytesIO
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/gzz/机械硬盘/sda3/Multimodal_Retrieval/", help="the directory which stores the image tsvfiles")
    parser.add_argument("--image_resolution", type=int, default=224, help="the resolution of transformed images, default to 224*224")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_path = os.path.join(args.data_dir, "MR_train_imgs.tsv")
    val_path = os.path.join(args.data_dir, "MR_valid_imgs.tsv")
    test_path = os.path.join(args.data_dir, "MR_test_imgs.tsv")
    if not os.path.exists(train_path[:-4]):
        os.makedirs(train_path[:-4])
    if not os.path.exists(val_path[:-4]):
        os.makedirs(val_path[:-4])
    if not os.path.exists(test_path[:-4]):
        os.makedirs(test_path[:-4])
    for path, split in zip((train_path, val_path, test_path), ("train", "valid", "test")):
        assert os.path.exists(path), "the {} filepath {} not exists!".format(split, path)
        print("begin to transform {} split".format(split))
        with open(path, "r") as fin:
            for line in tqdm(fin):
                img_id, b64 = line.strip().split("\t")
                image = Image.open(BytesIO(base64.urlsafe_b64decode(b64)))
                output_path = "{}/{}.png".format(path[:-4], img_id)
                image.save(output_path)
        print("finished transforming  {} split, the output is saved at {}".format(split, output_path[:-4]))
    print("done!")