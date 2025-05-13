import base64
import requests
import tqdm
import json
import os
import pdb
from collections import defaultdict
import random
from PIL import Image
import cv2
import re
import tqdm
import torch
import warnings

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import sys
sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/")
sys.path.append("../../")
from dataloaders import AllMLMBench_new
import copy

import os


if __name__=="__main__":

    # Load the dataset
    args = {
        'split': "val",
        'mode': "val",
        'num_data_points': 5000,
        'complex_only': True,
        'add_complex': True,
        "prompt_mode": "text_choice",
        "complex_only": True,
        "add_complex": True, 
        "add_perspective": True,
        'datasets': [
            "SAT_real",
#            "BLINK",
#            "CVBench",
#            "SAT_synthetic",
#            "VSR",
#            "GQASp",
        ]
    }

    ### load the model
    # pretrained = "ellisbrown/ft-llava-video-Qwen2-7B-2025_02_23_vsi_prop_scale_3q_trainset_25k"
    # pretrained = "ellisbrown/ft-llava-video-Qwen2-7B-2025_03_03_vsi_prop_scale_rgb_25k"
    # pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
    '''
    model_list = [
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_rel_direction_hard_mc',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_rel_direction_easy_oe',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_rel_distance_mc',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_rel_distance_oe',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_appearance_order_oe',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_rel_direction_hard_oe',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_room_size_est_mc',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_rel_direction_medium_oe',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_size_est_oe',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_count_mc',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_abs_distance_mc',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_rel_direction_medium_mc',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_count_oe',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_rel_direction_easy_mc',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_abs_distance_oe',
        'ellisbrown/ft-llava-video-Qwen2-7B-2025_02_21__vid_vsi_rgb_vsi_obj_appearance_order_mc',
    ]
    '''
    '''
    model_list = [
        "ellisbrown/ft-llava-video-Qwen2-7B-2025_02_23_vsi_prop_scale_3q_trainset_25k",
        "ellisbrown/ft-llava-video-Qwen2-7B-2025_03_03_vsi_prop_scale_rgb_25k",

    ]
    '''
    model_list = [
#        "ellisbrown/ft-llava-video-Qwen2-7B-2025_02_11__vid_ai2_SAT_50k"
#        "lmms-lab/LLaVA-Video-7B-Qwen2",
        "ellisbrown/ft-llava-video-Qwen2-7B-2025_03_22_sat_mixin_SAT_150k",
        "ellisbrown/ft-llava-video-Qwen2-7B-2025_03_22_sat_SAT_50k",
        "ellisbrown/ft-llava-video-Qwen2-7B-2025_03_22_sat_mixin_SAT_75k",
        "ellisbrown/ft-llava-video-Qwen2-7B-2025_03_22_sat_SAT_10k",
        "ellisbrown/ft-llava-video-Qwen2-7B-2025_03_22_sat_SAT_100k",
        "ellisbrown/ft-llava-video-Qwen2-7B-2025_03_22_sat_SAT_172k",
        "ellisbrown/ft-llava-video-Qwen2-7B-2025_03_24_sat_static_SAT_10k",
        "ellisbrown/ft-llava-video-Qwen2-7B-2025_03_24_sat_static_SAT_50k",
        "ellisbrown/ft-llava-video-Qwen2-7B-2025_03_24_sat_static_SAT_100k",
        "ellisbrown/ft-llava-video-Qwen2-7B-2025_03_24_sat_static_SAT_127k", 
#        "ellisbrown/ft-llava-video-Qwen2-7B-2025_03_22_sat_mixin_SAT_300k"       
    ]
    
    for pretrained in model_list:
        model_name_save = pretrained.split("/")[-1]
        run_name = model_name_save + "_".join(args['datasets'])
    
        dataset = AllMLMBench_new(args, tokenizer=None, image_processor=None)
        model_name = "llava_qwen"
        device = "cuda"
        device_map = "cuda:0"
        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map, attn_implementation="eager")  # Add any other thing you want to pass in llava_model_args
        model.eval()


        all_responses = []
        for ind, entry in enumerate(tqdm.tqdm(dataset)):

            try:
                im_files, images, prompt, text_label, correct_answer, answer_choices, datatype = entry
            except:
                break

            prompt = prompt.split("Human: Answer in natural language.")[-1]

            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
            
            question = DEFAULT_IMAGE_TOKEN + f"\n {prompt}"
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            
            images = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
            images = images.unsqueeze(0)

            # pdb.set_trace()
            cont = model.generate(
                input_ids,
                images=images,
                modalities= ["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            response_text = text_outputs

            # pdb.set_trace()
            # print(response_text)
            # Save the response
            all_responses.append({
                "prompt": prompt,
                "text_label": text_label,
                "response": response_text,
                "answer": correct_answer,
                "dataset": datatype
            })
            # pdb.set_trace()

            # Save the responses
            #if ind % 100 == 0:
            #    with open(f"responses/{pretrained.split('/')[-1]}_MLMbench_response.json", "w") as f:
            #        json.dump(all_responses, f)

            if ind>=20000:
                break
        with open(f"responses/{run_name}_MLMbench_response.json", "w") as f:
            json.dump(all_responses, f)
    