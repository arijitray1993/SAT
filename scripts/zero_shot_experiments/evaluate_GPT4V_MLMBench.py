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


# OpenAI API Key
api_key_file = "/projectnb/ivc-ml/array/research/robotics/openai"
with open(api_key_file, "r") as f:
  api_key = f.read().strip()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def encode_PILImage(image):
  buffered = BytesIO()
  image.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue())
  return img_str.decode('utf-8');''[]

def get_caption(imagePIL, prompt, api_key):
  # Getting the base64 string
  base64_image = encode_PILImage(imagePIL)

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  payload = {
    "model": "gpt-4o",
    "messages": [
    {
        "role": "user",
        "content": [
        {
            "type": "text",
            "text": prompt
        },
        {
            "type": "image_url",
            "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        ]
    }
    ],
    "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  return response


if __name__=="__main__":

    # Load the dataset
    dataset_args = {
        'split': "val",
        'mode': "val",
        'num_data_points': 5000,
        'complex_only': True,
        'add_complex': True,
        'datasets': [
            "SAT_real",
            "VSR",
            "GQASp"
        ]
    }
    run_name = "_".join(dataset_args['datasets'])

    dataset = AllMLMBench_new(dataset_args, tokenizer=None, image_processor=None)
    
    all_responses = []
    for entry in tqdm.tqdm(dataset):
        imfiles, images, prompt, text_label, correct_answer, answer_choices, datatype = entry

        if len(images)>1:
          print("concatenating images")
          # join the images into one
          opencv_image_a = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
          opencv_image_b = cv2.cvtColor(np.array(images[1]), cv2.COLOR_RGB2BGR)
          image = cv2.hconcat([opencv_image_a, opencv_image_b])
          image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
          images = [image,]

        prompt = prompt.split("Human: Answer in natural language.")[-1].split("###Assistant")[0]
        prompt += " Please answer just one of the options and no other text."
        pdb.set_trace()
        response = get_caption(images[0], prompt, api_key)

        response_text = response.json()['choices'][0]['message']['content']
        
        print("Prompt: ", prompt)
        print(response_text)

        # pdb.set_trace()

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
        with open(f"responses/GPT4V_responses_{run_name}.json", "w") as f:
            json.dump(all_responses, f)