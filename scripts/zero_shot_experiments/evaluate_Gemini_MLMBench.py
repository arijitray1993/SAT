import base64
from io import BytesIO
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
import numpy as np

import sys
sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/")
sys.path.append('../../')
from dataloaders import AllMLMBench_new


import google.generativeai as genai
import os

api_key_file = "/projectnb/ivc-ml/array/research/robotics/gemini"
with open(api_key_file, "r") as f:
  api_key = f.read().strip()


genai.configure(api_key=api_key)



def get_caption(images, prompt, model):
  # Choose a Gemini model.

  response = model.generate_content([prompt, *images])

  return response



if __name__=="__main__":

    # Load the dataset
    dataset_args = {
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
#            "BLINK",
#            "CVBench",
#            "SAT_synthetic",
            "SAT_real",
#            "VSR",
#            "GQASp"
        ]
    }


    #model_name = "gemini-1.5-pro"
    model_name = "gemini-1.5-flash"
    #model_name = "gemini-2.0-flash"
    use_cot = ""

    run_name = model_name + use_cot + "_".join(dataset_args['datasets'])

    dataset = AllMLMBench_new(dataset_args, tokenizer=None, image_processor=None)

    model = genai.GenerativeModel(model_name=model_name)

    all_responses = []
    
    count = 0
    for entry in tqdm.tqdm(dataset):
        try:
          imfiles, images, prompt, text_label, correct_answer, answer_choices, datatype = entry
        except:
          break
        
        if len(images) > 1:
            # make the images 512 x 512, saves money
            images = [image.resize((512, 512)) for image in images]
      
        prompt = prompt.split("Human: Answer in natural language.")[-1].split("###Assistant")[0]
        prompt += " Please answer just one of the options and no other text."

        if use_cot:
          prompt += "Think step by step and then answer one of the options."
        # pdb.set_trace()
        try:
          response = get_caption(images, prompt, model)
          response_text = response.text
        except:
          print("skipping")
          response_text = "n/a"
        
        print("Prompt: ", prompt)
        print(response_text)

        # Save the response
        all_responses.append({
          "prompt": prompt,
          "text_label": text_label,
          "image_path": "",
          "response": response_text,
          "answer": correct_answer,
          "answer_choices": [correct_answer,],
          "dataset": datatype
        })
        # pdb.set_trace()

        # Save the responses
        with open(f"responses/Gemini_responses_{run_name}.json", "w") as f:
            json.dump(all_responses, f)