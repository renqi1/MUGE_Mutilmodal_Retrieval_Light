# Code modified from https://github.com/openai/CLIP

from typing import Union, List
from pathlib import Path
import os
import json
import torch
from Clip import _tokenizer
from Clip.model import CLIP, restore_model

__all__ = ["load", "tokenize"]

_MODEL_INFO = {
    "ViT-B-16": {
        "struct": "ViT-B-16@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 224
    },
    "ViT-L-14": {
        "struct": "ViT-L-14@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 224
    },
    "ViT-L-14-336": {
        "struct": "ViT-L-14-336@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 336
    },
    "ViT-H-14": {
        "struct": "ViT-H-14@RoBERTa-wwm-ext-large-chinese",
        "input_resolution": 224
    },
    "RN50": {
        "struct": "RN50@RBT3-chinese",
        "input_resolution": 224
    },
}


def load(model, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", clip_path=None, bert_path=None):
    """Load CLIP and BERT model weights
    """
    bert_state_dict = torch.load(bert_path, map_location="cpu") if bert_path else None
    clip_state_dict = torch.load(clip_path, map_location="cpu") if clip_path else None

    restore_model(model, clip_state_dict, bert_state_dict).to(device)

    if str(device) == "cpu":
        model.float()
    return model

def create_model(name, checkpoint=None):
    model_name = _MODEL_INFO[name]['struct']
    vision_model, text_model = model_name.split('@')
    # Initialize the model.
    vision_model_config_file = Path(
        __file__).parent / f"model_configs/{vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)

    text_model_config_file = Path(
        __file__).parent / f"model_configs/{text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        for k, v in json.load(ft).items():
            model_info[k] = v
    if isinstance(model_info['vision_layers'], str):
        model_info['vision_layers'] = eval(model_info['vision_layers'])
    print('Model info', model_info)
    model = CLIP(**model_info)
    if checkpoint:
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        print('missing_keys:', missing_keys, '\t', 'unexpected_keys:', unexpected_keys)
        print('load pretrain state dict successfully !')
    return model


def tokenize(texts: Union[str, List[str]], context_length: int = 24) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all baseline models use 24 as the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append([_tokenizer.vocab['[CLS]']] + _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))[:context_length - 2] + [_tokenizer.vocab['[SEP]']])

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result