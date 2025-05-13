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
import ast
import yaml
import numpy as np

import sys
sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/")
sys.path.append('../../')
from dataloaders import ProcTHOR_image_camposition_marked



if __name__=="__main__":

    # Load the dataset

    args  = {
        'incontext_nomark_GPT': True,
        'split': "train",
        'mode': "val",
        'include_children': False,
        'use_angle': True,
        'use_attributes': True,
        'use_incontext': True,
        'use_no_mark_baseline': True,
        'normalize_rotation': True,
        'recon_qa_mode': True
    }

    dataset = ProcTHOR_image_camposition_marked(args, tokenizer=None, image_processor=None)
    asset_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json"))
    asset_desc_to_class = {}
    for asset_id in asset_desc:
        entries = asset_desc[asset_id]
        for im_file, obj_class, caption in entries:
            asset_desc_to_class[caption] = obj_class


    all_responses = []
    all_recon_qas = []
    for ind, entry in enumerate(tqdm.tqdm(iter(dataset))):
        # pdb.set_trace()
        image_path, img, caption, prompt, text_labels, program_text, house_json, objs_present = entry

        qa_s = []
        #for obj in objs_present:
        #    qa_s.append([
        #      f"Is there a {obj} in the image?",
        #      "yes",
        #      ["yes", "no"],
        #      "obj_class"
        #    ])

        program_text = program_text.replace("(", "[")
        program_text = program_text.replace(")", "]")
        
        cfg_dict = yaml.load(program_text, Loader=yaml.FullLoader)

        polygon = cfg_dict['polygon']
        # pdb.set_trace()
        max_coordinate = np.max(polygon)
        min_coordinate = np.min(polygon)
        max_dimension = max_coordinate - min_coordinate
        
        obj_locs = defaultdict(list)
        i=0
        while(True):
            if f'obj_{i}' in cfg_dict:
                if f'obj_{i}' not in objs_present:
                    i+=1
                    continue                
                obj = cfg_dict[f'obj_{i}']
                obj_pos = obj.split("at location ")[-1].split(" with")[0]
                obj_pos = ast.literal_eval(obj_pos)

                obj_desc = obj.split(" at location ")[0]
                obj_class = asset_desc_to_class[obj_desc]
                
                obj_locs[obj_desc].append(obj_pos)
            else:
                break
            i+=1

                
        incorrect_delta = random.choice(range(int(0.2*max_dimension), int(0.4*max_dimension)))
        for obj_desc, locs in obj_locs.items():
            correct_loc = random.choice(locs)

            incorrect_locs = []
            
            incorrect_locs.append([int(correct_loc[0] + incorrect_delta), correct_loc[1], int(correct_loc[2] - incorrect_delta)])
            incorrect_locs.append([int(correct_loc[0] - incorrect_delta), correct_loc[1], int(correct_loc[2] + incorrect_delta)])
            incorrect_locs.append([int(correct_loc[0]), int(correct_loc[1]+incorrect_delta), int(correct_loc[2] + incorrect_delta)])
            incorrect_locs.append([int(correct_loc[0]), correct_loc[1], int(correct_loc[2] - incorrect_delta)])
            incorrect_locs.append([int(correct_loc[0]+ incorrect_delta), correct_loc[1], int(correct_loc[2])])
            incorrect_locs.append([correct_loc[0], int(correct_loc[1] - incorrect_delta), int(correct_loc[2])])

            final_incoorect_locs = []
            for incorrec_loc in incorrect_locs:
                addincor = True
                for obj_pos_entry in locs:
                    if np.allclose(obj_pos_entry, incorrec_loc, atol=0.01):
                        addincor = False
                        break
                if addincor:
                    final_incoorect_locs.append(incorrec_loc)

            if len(final_incoorect_locs) < 3:
                continue
            incorrect_loc1, incorrect_loc2, incorrect_loc3 = random.sample(final_incoorect_locs, 3)
            qa_s.append([
                f"If there is a {obj_desc} in the image, what is the likely location in (x, y, z) of {obj_desc}?",
                correct_loc,
                ['"not present in the image"', '"'+str(correct_loc)+'"', '"'+str(incorrect_loc1)+'"', '"'+str(incorrect_loc2)+'"', '"'+str(incorrect_loc3)+'"'],
                "3D_loc"
            ])

        # pdb.set_trace()
        prompt = prompt.split("Using this information")[0]

        prompt = prompt + " Based on this, answer the following question."

        for qa_entry in qa_s:
            ques, ans, choices, qa_type = qa_entry

            random.shuffle(choices)

            ques = ques + " Choose between the following five choices: " + " or ".join(choices)
            
            if qa_type == "obj_class":
                full_prompt = f"{ques}"
            else:
                full_prompt = f"{prompt} {ques}. Just output one of the choices without any other explanation or text."

            print(full_prompt)
            all_recon_qas.append({
                "prompts": full_prompt,
                "image_path": image_path,
                "answers": str(ans),
                "answer_choices": choices
            })

            
    with open("all_recon_qas_nomark_train.json", "w") as f:
        json.dump(all_recon_qas, f)