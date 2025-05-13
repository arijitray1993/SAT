
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

try:
    sys.path.append("/projectnb/ivc-ml/array/research/robotics/LLaVA")
    # sys.path.append("models/LLaVA_modified/LLaVA")
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


def convert_byte_to_string(bytes_to_decode, max_len=None):
    if max_len is None:
        max_len = bytes_to_decode.shape[-1]
    return (bytes_to_decode.view(f"S{max_len}")[0]).decode()


def read_video_frames(video_path):
    """Reads frames from an MP4 video file."""

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def make_desc_for_task(task_name, task_templ):
    if task_name == "ObjectNavAffordance":
        task_desc = f"I need to go to the {task_templ['synsets'][0].split('.')[0]} that is best used to {task_templ['affordance'].lower().strip()} Which direction should I turn to face the object? Make sure to not run into obstacles."
    elif task_name == "PickupType":
        task_desc = f"I need to pickup {task_templ['synsets'][0].split('.')[0]}. Which direction should I turn to face the object?"
    elif task_name == "EasyObjectNavType":
        obj_name = task_templ['synsets'][0].split('.')[0]
        obj_name = obj_name.replace("_", " ")
        task_desc = f"I need to go to the {obj_name}. Which direction should I turn to face the object?" # Make sure to not run into obstacles."

    return task_desc



class SPOC_data(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        if self.args.get('debug_mode'):
            self.table = wandb.Table(columns=["Prompt", "Action Answer", "Image"])
            self.logger = wandb.init(project="spoc_debug", config=args)
        
        data_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/SPOC_data"

        tasks = os.listdir(data_path)
        action_map = {
            'm': "go straight",
            'l': "left",
            'r': "right",
            'ls': "left",
            'rs': "right",
            'b': "move backward",
            'end': "no action, goal achieved",
        }
        action_map = {
            'move_ahead': "straight",
            'rotate_left': "left",
            'rotate_right': "right",
            'rotate_left_small': "left",
            'rotate_right_small': "right",
        }
        selected_tasks = ["EasyObjectNavType"]#, "PickupType"] # "ObjectNavDescription",]
        self.action_choices = list(action_map.values())
        self.all_data = []
        split = args['split']
        if args['split'] == "valtrain":
            split = "train"
        for task in tasks:
            if task not in selected_tasks:
                continue
            train_ids = os.listdir(os.path.join(data_path, task, split))
            
            #load train h5py
            for train_id in train_ids:
                h5_file = h5py.File(os.path.join(data_path, task, split, train_id, "hdf5_sensors.hdf5"), 'r')
                
                episodes = h5_file.keys()

                for episode_id in episodes:
                    video_file = os.path.join(data_path, task, split, train_id, f"raw_navigation_camera__{episode_id}.mp4")
                    episode = h5_file[episode_id]

                    # these are the keys:
                    # 'house_index', 'hypothetical_task_success', 'last_action_is_random', 'last_action_str', 'last_action_success', 
                    # 'last_agent_location', 'minimum_l2_target_distance', 'minimum_visible_target_alignment', 'room_current_seen', 
                    # 'rooms_seen', 'task_relevant_object_bbox', 'templated_task_spec', 'visible_target_4m_count'

                    actions_taken = []
                    for action in episode['last_action_str']:
                        actions_taken.append(convert_byte_to_string(action))

                    random_action = []
                    for action in episode['last_action_is_random']:
                        random_action.append(action[0])
                    
                    last_action_success = []
                    for action in episode['last_action_success']:
                        last_action_success.append(action[0])

                    #pdb.set_trace()
                    object_bbox_cols = np.array(episode['nav_task_relevant_object_bbox']['max_cols'])
                    
                    obj_visible = [col[0]!=-1 for col in object_bbox_cols]

                    task_templ = ast.literal_eval(convert_byte_to_string(episode['templated_task_spec'][0]))
                    task_desc = make_desc_for_task(task, task_templ)
                    # pdb.set_trace()
                    
                    for index, (action, rand_action, last_action_succ, obj_vis) in enumerate(zip(actions_taken, random_action, last_action_success, obj_visible)):
                        if obj_vis:
                            if (not rand_action) and last_action_succ and index > 0:
                                if action in action_map:
                                    action_word = action_map[action]
                                else:
                                    action_word = action.replace("_", " ")
                                # pdb.set_trace()
                                if action_word == "look straight":
                                    if random.random() > 0.5:
                                        continue
                                if self.args['split'] == "val": # just for zero-shot eval.
                                    if action_word == "done":
                                        continue
                                
                                    if not obj_visible:
                                        continue
                                
                                self.all_data.append({
                                    'frame_index': index-1,
                                    'task': task_desc,
                                    'episode_id': episode_id,
                                    'video_file': video_file,
                                    'action_taken': action_word,
                                })
                    
                    
        if args['split'] == "train":
            self.all_data = self.all_data[:int(0.8*len(self.all_data))]
        elif args['split'] == "valtrain":
            self.all_data = self.all_data[int(0.8*len(self.all_data)):]
        else:
            self.all_data = self.all_data

        print("Length of SPOC data: ", len(self.all_data))
        # pdb.set_trace()

    def __getitem__(self, idx):
        entry = self.all_data[idx]
        
        video_frames = read_video_frames(entry['video_file'])

        frame_index = entry['frame_index']
        try:
            frame = video_frames[frame_index]
        except:
            video_frames = read_video_frames(entry['video_file'])
            frame = video_frames[frame_index]
        frame = Image.fromarray(frame)
        # pdb.set_trace()
        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        
        prompt = f'{prefix}###Human: <im_start><image><im_end> \nHuman: Answer in natural language. {entry["task"]} Answer the question using a single word or phrase. Choose between the following options: "left", "right", or "straight". ###Assistant: \n'
        action = entry['action_taken']
        text_label = prompt + action + " \n###"

        if self.args.get('mode') == "train":
            prompt = text_label

        if self.args.get('debug_mode'):
            self.table.add_data(prompt, action, wandb.Image(frame))
            self.logger.log({"spoc_examples": self.table})

        return ["",], [frame,], prompt, text_label, action, self.action_choices, "SPOC_easyObjNav"

    def __len__(self):
        return len(self.all_data)

    def collate_fn(self, batch):
        image_paths, images, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)
        
        new_images = []
        for image_b in images:
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

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # pdb.set_trace()
        return_dict = {
            'images': images,
            "input_ids": input_ids,
            'labels': input_ids,
            'text_labels': text_labels,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "prompts": prompts,
            'answers': answers,
            'dataset': datanames,
        }
        # pdb.set_trace()
        return return_dict