import json
import os
import pdb  # noqa
import random
from collections import defaultdict
import itertools
from itertools import combinations
import functools
import pickle as pkl
 
import torch
import tqdm  # noqa
from PIL import Image
from torch.utils.data import Dataset
import time
import torchvision

from torch.utils.data import WeightedRandomSampler
from transformers import Blip2Processor, InstructBlipProcessor # , CodeLlamaTokenizer
from transformers import AutoProcessor

from shapely.geometry.polygon import Polygon

from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import yaml
import numpy as np
import h5py

import ast
import cv2
import wandb

import sys
# sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/models/LLaVA")
sys.path.append("/projectnb/ivc-ml/array/research/robotics/LLaVA")
# sys.path.append("models/LLaVA_modified/LLaVA")
try:
    from llava.mm_utils import (
        process_images,
        tokenizer_image_token,
        get_model_name_from_path,
        KeywordsStoppingCriteria,
    )
    from llava.mm_utils import expand2square
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IMAGE_PLACEHOLDER,
    )
except:
    pass
import csv


class ArkitScenes(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode
        
        omnidata = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/3d_reasoning_qas/omni3d_img2ann.json"))

        self.data = []
        for image_file in omnidata:
            # pdb.set_trace()
            for object_name, center_3d, bbox_2d, cam_k in omnidata[image_file]:
                center_3d = (np.array(center_3d)*100).astype(int)
                self.data.append((image_file, object_name, center_3d, bbox_2d, cam_k))

        self.data = self.data[:3000]

        
    def __getitem__(self, idx):
        
        image_file, object_name, center_3d, bbox_2d, cam_k = self.data[idx]

        fx = cam_k[0][0]
        fy = cam_k[1][1]
        cx = cam_k[0][2]
        cy = cam_k[1][2]

        fx = round(fx, 2)
        fy = round(fy, 2)
        cx = round(cx, 2)
        cy = round(cy, 2)


        img = Image.open(image_file)

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        prefix += "###Human: <im_start><image><im_end> \nHuman: Answer with precise 3D coordinate points."
        prefix += "Assume camera is at origin. Camera looks at positive Z, X points to right and Y points upwards. "

        cam_instrinsic_prompt = f"Camera intrinsic parameters are: focal length x: {fx}, focal length y: {fy}, center point x: {cx}, center point y: {cy}. The image resolution is: {img.size[0]} x {img.size[1]}."
        
        question = f"What is the rough 3D location of {object_name}?"
        answer = str(list(center_3d)).replace("[", "(").replace("]", ")")

        prompt = f"{prefix} {cam_instrinsic_prompt} {question} ###Assistant: \n"
        text_label = prompt + answer + " \n###"

        return [image_file,], [img,], prompt, text_label, answer, [answer,], f"arkit_3dqa"

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        new_images = []
        for image_b in images_batch:
            for image in image_b:
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                # pdb.set_trace()
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)

        pixel_values = torch.stack(new_images, dim=0)

        input_ids = []
        attention_mask = []
        for prompt in prompts:
            # pdb.set_trace()
            input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids.append(input_id)
            attention_mask.append(torch.ones_like(input_id))
        
        # pad with zeros
        # pdb.set_trace()
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        # pdb.set_trace()
        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "labels": input_ids,
            "prompts": prompts,
            "text_labels": text_labels,
            "dataset": datanames,
            "answers": answers,
            "answer_choices": answer_choices,
        }

        return return_dict


class Stanford2D3DS(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode


        
    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def collate_fn(self, batch):
        pass


class Structured3D(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode


        
    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def collate_fn(self, batch):
        pass


class Grit20M(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode


        
    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def collate_fn(self, batch):
        pass