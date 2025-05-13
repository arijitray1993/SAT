import json
import os
import pdb  # noqa
import random
from collections import defaultdict
import itertools
from itertools import combinations
import functools
import pickle as pkl
import requests
import io
 
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
import math

import ast
import cv2
import wandb
from numpy.random import choice
import sys
# sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/models/LLaVA")
sys.path.append("/projectnb/ivc-ml/array/research/robotics/LLaVA")
# sys.path.append("models/LLaVA_modified/LLaVA")
#NOLINT

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

try:
    from utils.ai2thor_utils import generate_program_from_roomjson, format_program, generate_attribute_program_from_roomjson
except:
    pass

import numpy as np

from custom_datasets.embodied_ai_datasets import *
from custom_datasets.d3_datasets import *

try:
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
except:
    pass


def stich_image(images):
    total_width = sum(image.width for image in images)
    max_height = max(image.height for image in images)

    # Create a new blank image with the correct size
    new_image = Image.new('RGB', (total_width, max_height))

    # Paste images into the new image
    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.width

    return new_image

def add_red_dot_with_text(image, position, text):
    if position[0] is None:
        return image
    # Load the image
    draw = ImageDraw.Draw(image)

    # Coordinates and radius of the dot
    x, y = position
    radius = 10  # You can adjust the size of the dot

    # Draw the red dot
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='red')

    # Load a font (optional, comment out if not needed)
    try:
        font = ImageFont.truetype("arial.ttf", 8)  # Adjust font and size as needed
    except IOError:
        font = ImageFont.load_default()

    # Calculate text width and height to center it
    text_width = draw.textlength(text, font=font)
    text_x = x - text_width / 2
    text_y = y

    # Draw the text
    draw.text((text_x, text_y), text, fill='white', font=font)

    return image


def get_qa_type(question):
    question_type = "other"
    
    if "how did the camera" in question.lower() or "is the camera moving" in question.lower():
        question_type = "action_sequence"

    if ("need to go" in question.lower()):
        question_type = "goal_aim"

    if "any of the objects in the initial" in question.lower():
        question_type = "obj_movement"

    if "if i" in question.lower():
        question_type = "action_consequence"

    if 'if i move to the' in question.lower() or "for someone at the" in question.lower():
        question_type = "perspective"

    return question_type


def interleave_iterators(*iterators):
    finished = [False for x in range(len(iterators))]
    stop_cond = functools.reduce(lambda x,y:not x or not y,finished)
    while stop_cond:
        for i,it in enumerate(iterators):
            try:
                yield next(it)
            except StopIteration:
                finished[i] = True
        stop_cond = functools.reduce(lambda x,y:not x or not y,finished)

def format_prompts(images, question, answer_choices, answer, model_choice, mode):
    
    correct_answer = answer

    if len(answer_choices)>1:   
        ans_choice_order = answer_choices
        ans_choice_order = ['"'+ans+'"' for ans in ans_choice_order]
        random.shuffle(ans_choice_order)
        answer_choices_format = " or ".join(ans_choice_order)
    else:
        answer_choices_format = ""

    if model_choice=="llava":
        image_prompt_format = "<image>"*len(images)
        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        
        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. "

        if answer_choices_format != "":
            prompt += f"Choose between the following options: {answer_choices_format}. ###Assistant: \n"
        else:
            prompt += f"###Assistant: \n"

        text_label = prompt + correct_answer + " \n###"

        if mode == "train":
            prompt += f"{correct_answer} \n###"  
    elif model_choice == "llava_ov":
        # very weird prompt format. double check. 

        """
            <|im_start|>user <image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant \nThere is a red stop sign in the image.<|im_end|><|im_start|>user <image>\nWhat about this image? How many cats do you see?<|im_end|><|im_start|>assistant\n'
        """
        image_prompt_format = "<|im_start|>user  <image>\n <|im_end|>"*len(img)
        image_prompt_format = image_prompt_format[:-len("<|im_end|>")]

        if answer_choices_format != "":
            question = f"{question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}. "

        if self.args['mode'] == "train":
            prompt = f"{image_prompt_format}{question} <|im_end|><|im_start|>assistant: \n {answer} <|im_end|>"
            text_label = prompt
        else:
            prompt = f"{image_prompt_format}{question} <|im_end|><|im_start|>assistant: \n"
            text_label = prompt + answer + " <|im_end|>"
    
    return prompt, text_label


def get_inputs_for_model(imgs, prompts, tokenizer=None, image_processor=None, model_choice=None):

    if model_choice == "llava_ov":
        inputs = image_processor(text=prompts, images=imgs, return_tensors='pt').to(torch.float16)
        return inputs['pixel_values'], inputs['input_ids'], inputs['attention_mask'], inputs['image_sizes']

    if model_choice == "llava":
        new_images = []
        for image_b in imgs:
            for image in image_b:
                image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
                # pdb.set_trace()
                image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)

        pixel_values = torch.stack(new_images, dim=0)

        input_ids = []
        attention_mask = []
        for prompt in prompts:
            # pdb.set_trace()
            input_id = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids.append(input_id)
            attention_mask.append(torch.ones_like(input_id))
        
        # pad with zeros
        # pdb.set_trace()
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return pixel_values, input_ids, attention_mask


class CustomMix(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode
        
        if args.get("instructBLIP") or args.get("BLIP2"):
            self.batch_decode = self.image_processor.batch_decode

        self.all_mix = []
        self.all_lens = []
        self.weights = []

        mix_datas = args.get("mix_datas")

        if "llavaIT" in mix_datas:
            self.llava_data = LLaVAInstructTune(args, tokenizer, image_processor)
            self.all_mix.append(self.llava_data)
            self.weights.append(mix_datas["llavaIT"])
            self.all_lens.append(len(self.llava_data))

        if "SAT" in mix_datas:
            self.procthor_data = SAT(args, tokenizer, image_processor)
            self.all_mix.append(self.procthor_data)
            self.weights.append(mix_datas["SAT"])
            self.all_lens.append(len(self.procthor_data))
        
        if "VSR_VRD25D" in mix_datas:
            self.vsr_vrd25d = VSR_VRD25D(args, tokenizer, image_processor)
            self.all_mix.append(self.vsr_vrd25d)
            self.weights.append(mix_datas["VSR_VRD25D"])
            self.all_lens.append(len(self.vsr_vrd25d))
        
        if 'robopoint' in mix_datas:
            self.robopoint = RoboPointDataset(args, tokenizer, image_processor)
            self.all_mix.append(self.robopoint)
            self.weights.append(mix_datas["robopoint"])
            self.all_lens.append(len(self.robopoint))

        if 'procthor_cot' in mix_datas:
            self.procthor_cot = ProcTHOR_COT(args, tokenizer, image_processor)
            self.all_mix.append(self.procthor_cot)
            self.weights.append(mix_datas["procthor_cot"])
            self.all_lens.append(len(self.procthor_cot))
        
        print("combined data ...")

        print("Total number of data points: ", sum(self.all_lens))
    
    def __getitem__(self, idx):
        
        data = random.choices(population=self.all_mix, k=1, weights=self.weights)[0]
        return data[idx%len(data)]

    def __len__(self):
        return max(self.all_lens)
    
    def collate_fn(self, batch):
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        if self.args.get("llava_ov"):
            
            pixel_values, input_ids, attention_mask, image_sizes = get_inputs_for_model(images_batch, prompts, None, self.image_processor, model_choice="llava_ov")
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "image_sizes": image_sizes,
            }
        else:
            pixel_values, input_ids, attention_mask = get_inputs_for_model(images_batch, prompts, self.tokenizer, self.image_processor, model_choice="llava")
            
            return_dict = {
                'image_paths': image_paths,
                "input_ids": input_ids[:, :800],
                "attention_mask": attention_mask[:, :800],
                'pixel_values': pixel_values,
                "labels": input_ids[:, :800],
                "prompts": prompts,
                "text_labels": text_labels,
                "datanames": datanames,
                "answers": answers,
            }

        return return_dict


class VSR_VRD25D(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode
        
        if args.get("split") == "train":
            data_files = {"train": "train.jsonl",}
        else:
            data_files = {"test": "test.jsonl"}
        dataset = load_dataset("cambridgeltl/vsr_random", data_files=data_files)
        self.coco_path = "/projectnb/ivc-ml/array/data/COCO/images/"

        vrd_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/vrd_qa_data.json"

        self.data = []
        if args.get("split") == "train":
            dataset = dataset["train"]
        else:
            dataset = dataset["test"]
        for entry in dataset:
            image_path = entry["image_link"]
            image_path = image_path.split("/")[-2:]
            image_path = os.path.join(self.coco_path, *image_path)

            caption = entry["caption"].lower()
            relation = entry["relation"].lower()
            label = entry["label"]

            entities = caption.split(relation)

            subject, object = entities[0], entities[1]
            subject = subject.strip().lower().replace("is", "").replace("the", "").replace("are", "")
            object = object.strip().lower().replace("is", "").replace("the", "").replace("are", "")

            question = f"Is {subject} {relation} the {object}?"
            answer = "yes" if label == 1 else "no"
            wrong_answer = "no" if label == 1 else "yes"

            self.data.append((image_path, question, [answer, wrong_answer]))
            print("number of data points: ", len(self.data))

        if args.get("split") == "train":
            vrd25data = json.load(open(vrd_path))
            v25_data = []
            for img, qa_entries in vrd25data:
                for question, answers in qa_entries:
                    v25_data.append((img, question, answers))
            self.data += random.sample(v25_data, min(len(v25_data),170000))
            random.shuffle(self.data)
    
            print("Total number of data points in VRD25D: ", len(v25_data))
        
        
        if args.get("split") != "train": 
            self.data = self.data[:args['num_data_points']]

        print("Total number of data points in VSR_VRD25D: ", len(self.data))
    
    def __getitem__(self, idx):
        im_file, question, answer = self.data[idx]
        
        correct_answer = answer[0]

        ans_choice_order = ['"'+ans+'"' for ans in answer]
        random.shuffle(ans_choice_order)
        answer_choices_format = " or ".join(ans_choice_order) 

        #if im_file is not None:
        img = [Image.open(im_file).convert("RGB"),]

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        if self.args['mode'] == "train":
            prompt = f"{prefix}###Human: <im_start><image><im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}. ###Assistant: \n {correct_answer} \n###"
            text_labels = prompt
        else:
            prompt = f"{prefix}###Human: <im_start><image><im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}. ###Assistant: \n "
            text_labels = prompt + correct_answer + " \n###"        

        
        return [im_file,], img, prompt, text_labels, correct_answer, answer, "vsr25d_spatial"
    
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

class GQASpatial_OG_QA(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        gqa_qa = json.load(open("/projectnb/ivc-ml/array/data/GQA/val_balanced_questions.json"))
        gqa_im_path = "/projectnb/ivc-ml/array/data/GQA/images/"

        qa_data = []
        for qaid in gqa_qa:
            entry = gqa_qa[qaid]
            if entry['types']['semantic'] == 'rel':
                img_id = entry["imageId"]
                question = entry["question"]
                answer = entry["answer"]
                image_path = os.path.join(gqa_im_path, f"{img_id}.jpg")

                qa_data.append((image_path, question, answer))
        
        self.data = qa_data[:args.get("num_data_points", 10000)]
    
    def __getitem__(self, idx):
        im_file, question, answer = self.data[idx]

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        prompt = f"{prefix}###Human: <im_start><image><im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. ###Assistant: \n "
        text_labels = prompt + answer + " \n###" 

        img = [Image.open(im_file).convert("RGB"),]       

        return [im_file,], img, prompt, text_labels, answer, [answer,], "gqa_spatial_ogqa"


    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        pixel_values, input_ids, attention_mask =  get_inputs_for_model(images_batch, prompts, self.tokenizer, self.image_processor, model_choice="llava")
        
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


class LLaVAInstructTune(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode
        
        if args.get("llava_ov"):
            self.batch_decode = self.image_processor.batch_decode

        json_data = json.load(open(args['llava_json_path']))
        
        self.image_path = args['llava_image_path']
        
        self.data = []

        for entry in tqdm.tqdm(random.sample(json_data, args.get('num_data_points'))):
            
            im_path = entry.get('image')

            # pdb.set_trace()
            if im_path is None:
                continue

            if not os.path.exists(os.path.join(self.image_path, im_path)):
                continue
            
            if len(entry['conversations'])%2!=0:
                continue

            for question, answer in zip(entry['conversations'][::2], entry['conversations'][1::2]):
                self.data.append((os.path.join(self.image_path, im_path), question['value'], answer['value']))

        
        if args.get("split") == "train":
            self.data = self.data[:int(len(self.data)*0.9)]
        else:
            self.data = self.data[int(len(self.data)*0.9):]
        
        print("Total number of data points in instructtune: ", len(self.data))

    def __getitem__(self, idx):
        im_file, question, answer = self.data[idx]
        
        #if im_file is not None:
        img = [Image.open(im_file).convert("RGB"),]
        #else:
        #    img = Image.new("RGB", (224, 224), (255, 255, 255)) 

        if "<image>" in question:
            question = question.replace("<image>", "")

        prompt, text_labels = format_prompts(img, question, ["",], answer, model_choice=self.args.get("model_choice"), mode=self.args['mode'])
        
        return [im_file,], img, prompt, text_labels, answer, [answer,], "llava_instructtune"
        
    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        image_paths, imgs, captions, prompts, text_labels, program_texts, house_jsons, objs_present = zip(*batch)
        
        if self.args.get("llava_ov"):
            pixel_values, input_ids, attention_mask, image_sizes =  get_inputs_for_model(imgs, prompts, None, self.image_processor, model_choice="llava_ov")
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "image_sizes": image_sizes,
            }
        else:
            pixel_values, input_ids, attention_mask =  get_inputs_for_model(imgs, prompts, self.tokenizer, self.image_processor, model_choice="llava")
            # pdb.set_trace()
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
            }

        return return_dict


class RealSATDynamic(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        data = json.load(open("/projectnb/ivc-ml/array/data/SAT/realDynamic/SATDynamicReal.json"))

        # make a copy of the dataset where the answer choices are reversed and add to self.data
        new_data = []
        for entry in data:
            new_data.append((entry[0], entry[1], entry[3], entry[2], entry[2], entry[4]))
            new_data.append((entry[0], entry[1], entry[2], entry[3], entry[2], entry[4]))
        
        self.data = new_data

        print("Total number of data points in RealSATDynamic with circ eval: ", len(self.data))
    
    def get_prefix(self, datatype):
        if datatype == "ego_movement":
            prefix = "The first image is from the beginning of the video and the second image is from the end. "
        else:
            prefix = ""
        return prefix

    
    def __getitem__(self, idx):
        images, question, answer, distractor, correct_answer, datatype = self.data[idx]

        answer_choices = [answer, distractor]

        image_prompt_format = "<image>"*len(["im" for im in images if im!=""])

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

        question_specific_prefix = self.get_prefix(datatype)

        question = f"{question_specific_prefix} {question}"

        final_prompt, final_label = format_prompts(images, question, answer_choices, answer, model_choice=self.args.get("model_choice"), mode=self.args['mode'])

        imgs = [Image.open(im_file).convert("RGB") for im_file in images if im_file!=""]

        return images, imgs, prompt, text_label, correct_answer, answer_choices, "realsat_"+datatype

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        pixel_values, input_ids, attention_mask =  get_inputs_for_model(images_batch, prompts, self.tokenizer, self.image_processor, model_choice="llava")

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

class ProcTHOR_COT(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        self.data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/3d_reasoning_qas/CoT_reasoning/sat_train_cot_reasoning_qas.json"))

        spatial_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_spatial_qas_v2_train.json'    
        spatial_data = json.load(open(spatial_qa_json_path))    
        static_data = []
        for house_ind, cam_pos, cam_rot, qa_entries in spatial_data:
            static_data.extend([(question, im_order, "", answers) for question, im_order, answers in qa_entries])
        
        self.data.extend(static_data)


        print("Total number of data points: ", len(self.data))
    
    def __getitem__(self, idx):
        question, im_order, cot_reason, answers = self.data[idx]

        correct_answer = answers[0]
        answer_choices = answers.copy()
        random.shuffle(answer_choices)
        
        answer_choices_format = " or ".join([f'"{ans}"' for ans in answer_choices])

        image_prompt_format = "<image>"*len(im_order)

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions"

        if cot_reason == "":
            prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}.###Assistant: ANSWER: \n {correct_answer} \n"
        else:
            prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format} ###Assistant: Let me first reason. REASON: {cot_reason} Hence, here is my final answer. ANSWER: \n {correct_answer} \n###"
        text_label = prompt

        imgs = [Image.open(im_file).convert("RGB") for im_file in im_order if im_file!=""]

        return im_order, imgs, prompt, text_label, correct_answer, answer_choices, "procthor_cot"
    
    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        pixel_values, input_ids, attention_mask =  get_inputs_for_model(images_batch, prompts, self.tokenizer, self.image_processor, model_choice="llava")

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

class SAT(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        dataset = load_dataset("array/SAT", batch_size=1)

        self.data = dataset[args['split']]

    def __getitem__(self, idx):
        entry = self.data[idx]

        images = entry['image_bytes'] # this is a list of images. Some questions are on one image, and some on 2 images

        image_paths = ["",]*len(images) # image paths are empty since encoded in entry.

        question = entry['question']
        answer_choices = entry['answers']
        correct_answer = entry['correct_answer']

        qa_type = entry['question_type']
        
        corrected_answer_choices = []
        for answer in answer_choices:
            if "in the first frame" in answer: # a small bug, todo fix in data gemeration later.
                answer = answer.replace("in the first frame", "")
            corrected_answer_choices.append(answer)
        answer_choices = corrected_answer_choices

        ans_choice_order = answer_choices
        ans_choice_order = ['"'+ans+'"' for ans in ans_choice_order]
        random.shuffle(ans_choice_order)
        answer_choices_format = " or ".join(ans_choice_order)

        final_prompt, final_label = format_prompts(images, question, answer_choices, answer, model_choice=self.args.get("model_choice"), mode=self.args['mode'])

        
        return image_paths, images, final_prompt, final_label, correct_answer, answer_choices, qa_type
    
    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        pixel_values, input_ids, attention_mask =  get_inputs_for_model(images_batch, prompts, self.tokenizer, self.image_processor, model_choice="llava")

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


class RoboPointDataset(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode        

        llava_format_data = json.load(open("/projectnb/ivc-ml/array/data/Robopoint/robopoint_1432k.json"))
        
        self.image_path = "/projectnb/ivc-ml/array/data/Robopoint/images"
        self.data = []
        for entry in tqdm.tqdm(llava_format_data):
            im_path = entry.get('image')

            # pdb.set_trace()
            if im_path is None:
                continue

            if not os.path.exists(os.path.join(self.image_path, im_path)):
                continue
            
            if len(entry['conversations'])%2!=0:
                continue

            for question, answer in zip(entry['conversations'][::2], entry['conversations'][1::2]):
                self.data.append(([os.path.join(self.image_path, im_path),], question['value'], answer['value']))

        print("length of robopoint data: ", len(self.data))

    def __getitem__(self, idx):
        image_order, question, correct_answer = self.data[idx]

        image_prompt_format = "<image>"*len(image_order)

        question = question.replace("<image>", "")
        
        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        if self.args['mode'] == "train":
            prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. ###Assistant: \n {correct_answer} \n###"
            text_label = prompt
        else:
            prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. ###Assistant: \n" 
            text_label = prompt + correct_answer + " \n###"

        images = [Image.open(img).convert("RGB") for img in image_order]

        return image_order, images, prompt, text_label, correct_answer, [correct_answer,], f"robopoint"
    
    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        pixel_values, input_ids, attention_mask =  get_inputs_for_model(images_batch, prompts, self.tokenizer, self.image_processor, model_choice="llava")
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


#### all image qa real datasets

class GQA(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.batch_decode = self.tokenizer.batch_decode
        
        gqa_path = "/projectnb/ivc-ml/array/data/GQA/" 

        if args['split'] == "valtrain":
            data = json.load(open(os.path.join(gqa_path, "val_all_questions.json")))
        elif args['split'] == "val":
            data = json.load(open(os.path.join(gqa_path, "val_all_questions.json")))
        elif args['split'] == "train":
            data = json.load(open(os.path.join(gqa_path, "train_all_questions.json")))

        self.data = []
        for qid in data:
            question = data[qid]['question']
            answer = data[qid]['answer']
            image_id = data[qid]['imageId']
            im_file = os.path.join(gqa_path, "images", f"{image_id}.jpg")
            self.data.append((question, answer, im_file))

        self.data = self.data[:args['num_data_points']]

        print("Split: ", args['split'])
        print("Total number of data points ", len(self.data))


    def __getitem__(self, idx):
        question, answer, im_file = self.data[idx]

        image = Image.open(im_file).convert("RGB")

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

        image_prompt_format = "<image>"

        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. ###Assistant: \n "
        text_label = prompt + answer

        return [], [image,], prompt, text_label, answer, [answer, ], "GQA"

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        
        _, images, prompts, text_labels, answers, _, datanames = zip(*batch)

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
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)

        # pdb.set_trace()
        return_dict = {
            "input_ids": input_ids,
            'labels': input_ids,
            'text_labels': text_labels,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "prompts": prompts,
            'answers': answers,
            "dataset": datanames
        }
        # pdb.set_trace()
        return return_dict


class VQAV2(Dataset):
    def __init__(self, args, tokenizer, image_processor):

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.batch_decode = self.tokenizer.batch_decode

        self.vqa_path = "/projectnb/ivc-ml/array/data/VQA/VQAV2"

        vqa_anno = json.load(open(os.path.join(self.vqa_path, "v2_mscoco_val2014_annotations.json")))
        vqa_ques = json.load(open(os.path.join(self.vqa_path, "v2_OpenEnded_mscoco_val2014_questions.json")))

        self.data = []
        for anno_entry, ques_entry in zip(vqa_anno['annotations'], vqa_ques['questions']):
            assert anno_entry['question_id'] == ques_entry['question_id']
            question = ques_entry['question']
            answer = anno_entry['multiple_choice_answer']
            image_id = anno_entry['image_id']
            im_file = os.path.join(self.vqa_path, "val2014", f"COCO_val2014_{str(image_id).zfill(12)}.jpg")
            self.data.append((question, answer, im_file))
        
        self.data = self.data[:args['num_data_points']]

    def __getitem__(self, idx):
        question, answer, im_file = self.data[idx]

        image = Image.open(im_file).convert("RGB")

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

        image_prompt_format = "<image>"

        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. ###Assistant: \n "

        text_label = prompt + answer

        return [], [image,], prompt, text_label, answer, [answer, ], "vqav2"

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        _, images, prompts, text_labels, answers, _, datanames = zip(*batch)

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
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)

        # pdb.set_trace()
        return_dict = {
            "input_ids": input_ids,
            'labels': input_ids,
            'text_labels': text_labels,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "prompts": prompts,
            'answers': answers,
            "dataset": datanames
        }
        # pdb.set_trace()
        return return_dict
        
class OKVQA(Dataset):
    def __init__(self, args, tokenizer, image_processor):

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.batch_decode = self.tokenizer.batch_decode

        self.okvqa_path = "/projectnb/ivc-ml/array/data/VQA/OKVQA"

        okvqa_anno = json.load(open(os.path.join(self.okvqa_path, "mscoco_val2014_annotations.json")))
        okvqa_ques = json.load(open(os.path.join(self.okvqa_path, "OpenEnded_mscoco_val2014_questions.json")))

        self.data = []
        for anno_entry, ques_entry in zip(okvqa_anno['annotations'], okvqa_ques['questions']):
            assert anno_entry['question_id'] == ques_entry['question_id']
            question = ques_entry['question']
            answer = anno_entry['answers'][0]['answer']
            image_id = anno_entry['image_id']
            im_file = os.path.join(self.okvqa_path, "val2014", f"COCO_val2014_{str(image_id).zfill(12)}.jpg")
            self.data.append((question, answer, im_file))
        
        self.data = self.data[:args['num_data_points']]

    def __getitem__(self, idx):
        question, answer, im_file = self.data[idx]

        image = Image.open(im_file).convert("RGB")

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

        image_prompt_format = "<image>"

        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. ###Assistant: \n "

        text_label = prompt + answer

        return [], [image,], prompt, text_label, answer, [answer, ], "okvqa"

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        _, images, prompts, text_labels, answers, _,  datanames = zip(*batch)

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
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)


        # pdb.set_trace()
        return_dict = {
            "input_ids": input_ids,
            'labels': input_ids,
            'text_labels': text_labels,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "prompts": prompts,
            'answers': answers,
            'dataset': datanames
        }
        # pdb.set_trace()
        return return_dict


class AllVQA(Dataset):
    def __init__(self, args, tokenizer, image_processor):

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.batch_decode = self.tokenizer.batch_decode

        self.gqa = GQA(args, tokenizer, image_processor)
        self.vqav2 = VQAV2(args, tokenizer, image_processor)
        self.okvqa = OKVQA(args, tokenizer, image_processor)
        
    
    def __getitem__(self, idx):
        if idx < len(self.gqa):
            return self.gqa[idx]
        elif idx < len(self.gqa) + len(self.vqav2):
            return self.vqav2[idx - len(self.gqa)]
        else:
            return self.okvqa[idx - len(self.gqa) - len(self.vqav2)]
    
    def __len__(self):
        return len(self.gqa) + len(self.vqav2) + len(self.okvqa)
    
    def collate_fn(self, batch):
        return self.gqa.collate_fn(batch)

class MMBench(Dataset):
    def __init__(self, args):
        
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def collate_fn(self, batch):
        pass

class SeedBench(Dataset):
    pass


class CVBench(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        cv_bench = load_dataset("nyu-visionx/CV-Bench")
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        if args.get("instructBLIP") or args.get("BLIP2"):
            self.batch_decode = self.image_processor.batch_decode

        self.data = cv_bench['test'].shuffle(seed=42)

        # random.shuffle(self.data)
        self.data = self.data[:args['num_data_points']]
        
        self.choice_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7} 

    def __getitem__(self, idx):
        '''
        {'idx': 0,
        'type': '2D',
        'task': 'Count',
        'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=256x256>,
        'question': 'How many organs are in the image?',
        'choices': ['3', '2', '1', '0'],
        'answer': '(C)',
        'prompt': 'How many organs are in the image? Select from the following choices.\n(A) 3\n(B) 2\n(C) 1\n(D) 0',
        'filename': 'img/2D/count/ade20k_10.png',
        'source': 'ADE20K',
        'source_dataset': 'ADE20K Validation Set',
        'source_filename': 'ADE_val_00000248.jpg',
        'target_class': None,
        'target_size': None,
        'bbox': None}
        '''
        image = self.data['image'][idx]
        question = self.data['question'][idx]

        choices = self.data['choices'][idx]
        choice_format = ", ".join(choices[:-1]) + ", or "+choices[-1]


        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        image_prompt_format = "<image>"

        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Choose between the following options: {choice_format}.###Assistant: \n "

        answer = self.data['answer'][idx]
        answer = answer.replace("(", "").replace(")", "")
        answer = choices[self.choice_to_number[answer]]

        type_task = self.data['type'][idx] + "_" + self.data['task'][idx]
        
        

        if self.args.get("instructBLIP") or self.args.get("BLIP2"):
            if self.args.get("zero_shot_mode"):
                prompt = f"{question} Choose between the following options: {choice_format}?"
            else:
                prompt = f"Question: {question} Answer the question using a single word or phrase. Choose between the following options: {choice_format}. Answer: "
        elif self.args.get("llava_ov"):
            
            """
            <|im_start|>user <image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant \nThere is a red stop sign in the image.<|im_end|><|im_start|>user <image>\nWhat about this image? How many cats do you see?<|im_end|><|im_start|>assistant\n'
            """
            image_prompt_format = "<|im_start|>user  <image>\n <|im_end|>"*1
            image_prompt_format = image_prompt_format[:-len("<|im_end|>")]
            
            prompt = f"{image_prompt_format}{question} Choose between the following options: {choice_format} <|im_end|><|im_start|>assistant: \n"
            text_label = prompt + answer
        else:
            if self.args.get("zero_shot_mode"):
                prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {choice_format}.###Assistant: \n "
            elif self.args.get("zero_shot_choice_mode"):
                prompt = self.data['prompt'][idx]
                prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: {prompt} ###Assistant: \n "
                answer = self.data['answer'][idx]
            text_label = prompt + answer
        

        
        # pdb.set_trace()
        return [], [image,], prompt, text_label, answer, [answer,], f"cvbench_{type_task}"
        
    def __len__(self):
        return len(self.data['prompt'])
        
    def collate_fn(self, batch):
        img_files, images, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        if self.args.get("llava_ov"):
            pixel_values, input_ids, attention_mask, image_sizes =  get_inputs_for_model(images, prompts, None, self.image_processor, model_choice="llava_ov")
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "image_sizes": image_sizes,
                "dataset": datanames,
                "answers": answers,
            }
        else:
            pixel_values, input_ids, attention_mask =  get_inputs_for_model(images, prompts, self.tokenizer, self.image_processor, model_choice="llava")
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
            }

        return return_dict


class BLINK(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode
        
        if args.get("instructBLIP") or args.get("BLIP2"):
            self.batch_decode = self.image_processor.batch_decode

        self.args = args

        dataset_name = 'BLINK-Benchmark/BLINK'

        SUBTASK_NAME = ['Multi-view_Reasoning', 'Relative_Depth', 'Spatial_Relation'] # , 'Object_Localization',]
        #SUBTASK_NAME = ['Relative_Depth', 'Spatial_Relation'] # , 'Object_Localization',]

        self.data = []
        for subtask in SUBTASK_NAME:
            count = 0
            for entry in load_dataset(dataset_name, subtask)['val']:
                self.data.append((entry, subtask))
                count += 1
                if count >= args['num_data_points']/3:
                    break

        self.choice_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7} 


    def __getitem__(self, idx):
        entry, subtask = self.data[idx]
        question = entry['prompt'].split("?")[0]+"?"
        
        
        # question = question.replace("The images are frames from a video. The video is shooting a static scene. The camera is either moving clockwise (left) or counter-clockwise (right) around the object.", "")
        
        answer = entry['answer']
        answer = answer.replace("(", "").replace(")", "")
        answer = entry['choices'][self.choice_to_number[answer]]

        if "The video is shooting a static scene. The camera is either moving clockwise" in question:
            answer_choices = ["moved "+x for x in entry['choices']]
            answer = "moved "+answer
            choice_format = ", ".join(answer_choices[:-1]) + ", or "+answer_choices[-1]

            question = question.replace("The video is shooting a static scene. The camera is either moving clockwise (left) or counter-clockwise (right) around the object.", "")
        else:    
            choice_format = ", ".join(entry['choices'][:-1]) + ", or "+entry['choices'][-1]

        images = []
        image_1 = entry['image_1']
        images.append(image_1)
        if entry['image_2'] is not None:
            image_2 = entry['image_2']
            images.append(image_2)

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        image_prompt_format = "<image>"*len(images)

        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Choose between the following options: {choice_format}.###Assistant: \n "

        if self.args.get("zero_shot_mode"):
            prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {choice_format}.###Assistant: \n "
        elif self.args.get("zero_shot_choice_mode"):
            prompt = entry['prompt']
            prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: {prompt} ###Assistant: \n "
            answer = entry['answer']

        text_label = prompt + answer

        if self.args.get("instructBLIP") or self.args.get("BLIP2"):
            if self.args.get("zero_shot_mode"):
                prompt = f"{question} Choose between the following options: {choice_format}?"
            else:
                prompt = f"Question: {question} Answer the question using a single word or phrase. Choose between the following options: {choice_format}. Answer: "
            text_label = prompt + answer
        elif self.args.get("llava_ov"):
            
            """
            <|im_start|>user <image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant \nThere is a red stop sign in the image.<|im_end|><|im_start|>user <image>\nWhat about this image? How many cats do you see?<|im_end|><|im_start|>assistant\n'
            """
            image_prompt_format = "<|im_start|>user  <image>\n <|im_end|>"*len(images)
            image_prompt_format = image_prompt_format[:-len("<|im_end|>")]
            
            prompt = f"{image_prompt_format}{question} Choose between the following options: {choice_format} <|im_end|><|im_start|>assistant: \n"
            text_label = prompt + answer
        

        # pdb.set_trace()
        return [], images, prompt, text_label, answer, [answer,],  "BLINK_"+subtask

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        img_files, images, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        if self.args.get("llava_ov"):
            pixel_values, input_ids, attention_mask, image_sizes =  get_inputs_for_model(images, prompts, None, self.image_processor, model_choice="llava_ov")
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "image_sizes": image_sizes,
                "dataset": datanames,
                "answers": answers,
            }
        else:
            pixel_values, input_ids, attention_mask =  get_inputs_for_model(images, prompts, self.tokenizer, self.image_processor, model_choice="llava")
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
            }

        return return_dict


class AllMLMBench_new(Dataset):
    def __init__(self, args, tokenizer, image_processor):

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        self.all_data = []

        if "BLINK" in args['datasets']:
            self.all_data.append(BLINK(args, tokenizer, image_processor))
            
        if "CVBench" in args['datasets']:
            self.all_data.append(CVBench(args, tokenizer, image_processor))

        if "SAT_real" in args['datasets']:
            self.all_data.append(RealSATDynamic(args, tokenizer, image_processor))
        
        if "SAT_synthetic" in args['datasets']:
            self.all_data.append(ProcTHOR_reasoning(args, tokenizer, image_processor))
        
        if "VSR" in args['datasets']:
            vsr_args = args.copy()
            vsr_args['num_data_points'] = 3000
            self.all_data.append(VSR_VRD25D(vsr_args, tokenizer, image_processor))
        
        if "GQASp" in args['datasets']:
            gqa_args = args.copy()
            gqa_args['num_data_points'] = 3000
            self.all_data.append(GQASpatial_OG_QA(gqa_args, tokenizer, image_processor))
        
        if "GQA" in args['datasets']:
            gqa_args = args.copy()
            gqa_args['num_data_points'] = 3000
            self.all_data.append(GQA(gqa_args, tokenizer, image_processor))

        if "VQAV2" in args['datasets']:
            gqa_args = args.copy()
            gqa_args['num_data_points'] = 3000
            self.all_data.append(VQAV2(gqa_args, tokenizer, image_processor))
        
        if "MME" in args['datasets']:
            self.all_data.append(MME(args, tokenizer, image_processor))
        
        if "POPE" in args['datasets']:
            self.all_data.append(POPE(args, tokenizer, image_processor))

    def __getitem__(self, idx):
        for data in self.all_data:
            if idx < len(data):
                return data[idx]
            idx -= len(data)

    def __len__(self):
        total_len = 0
        for data in self.all_data:
            total_len += len(data)
        return total_len

    def collate_fn(self, batch):
        return self.all_data[0].collate_fn(batch)


