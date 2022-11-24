import os
import logging
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from Clip.clip import tokenize

class EvalTxtDataset(Dataset):
    def __init__(self, jsonl_filename, max_txt_length=24):
        assert os.path.exists(jsonl_filename), "The annotation datafile {} not exists!".format(jsonl_filename)
        
        logging.debug(f'Loading jsonl data from {jsonl_filename}.')
        self.queries = []
        with open(jsonl_filename, "r") as fin:
            for line in fin:
                obj = json.loads(line.strip())
                query_id = obj['query_id']
                query = obj['query_text']
                self.queries.append((query_id, query))
        logging.debug(f'Finished loading jsonl data from {jsonl_filename}.')
        
        self.max_txt_length = max_txt_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_id, query = self.queries[idx]
        text = tokenize([str(query)], context_length=self.max_txt_length)[0]
        return query_id, text

class EvalImgDataset(Dataset):
    def __init__(self, img_filename):
        assert os.path.exists(img_filename), "The image datafile {} not exists!".format(img_filename)
        
        logging.debug(f'Loading image file from {img_filename}.')
        self.imgs = os.listdir(img_filename)
        self.img_ids = [x[:-4] for x in self.imgs]
        self.img_filename = img_filename

    def _read_img_from_file(self, img_id, img_filename):
        img_path = '{}/{}.png'.format(img_filename, img_id)
        img = Image.open(img_path)
        img_array = np.array(img).transpose(2, 0, 1)
        return torch.from_numpy(img_array)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image = self._read_img_from_file(img_id, self.img_filename)
        return img_id, image

def get_eval_txt_dataset(args, max_txt_length=24):
    input_filename = args.text_data
    dataset = EvalTxtDataset(input_filename, max_txt_length=max_txt_length)
    num_samples = len(dataset)
    dataloader = DataLoader(dataset, batch_size=args.text_batch_size, pin_memory=True)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return dataloader

def get_eval_img_dataset(args):
    img_filename = args.image_data
    dataset = EvalImgDataset(img_filename)
    num_samples = len(dataset)
    dataloader = DataLoader(dataset, batch_size=args.img_batch_size, pin_memory=True)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return dataloader