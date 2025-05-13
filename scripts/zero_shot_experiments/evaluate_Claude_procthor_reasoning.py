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
import anthropic

import sys
sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/")
sys.path.append('../../')
from dataloaders import ProcTHOR_reasoning
import httpx

api_key_file = "/projectnb/ivc-ml/array/research/robotics/claude"
with open(api_key_file, "r") as f:
  api_key = f.read().strip() 



# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_caption(image_path, prompt):
    encoded_ims = []
    image1_path = image_path[0]
    image1_media_type = "image/jpeg"
    image1_data = encode_image(image1_path)

    if len(image_path)>1:
        image2_path = image_path[1]
        image2_media_type = "image/jpeg"
        image2_data = encode_image(image2_path)
    
    if len(image_path)==1:
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "First image:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image1_media_type,
                                "data": image1_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
    else:
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "First image:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image1_media_type,
                                "data": image1_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Second image:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image2_media_type,
                                "data": image2_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
    return message


if __name__=="__main__":

    # Load the dataset
    args= {
        "split": "val",
        "mode": "val",
        "prompt_mode": "text_choice",
        "complex_only": True,
        "add_complex": True, 
        "add_perspective": True
    } 

    dataset = ProcTHOR_reasoning(args, tokenizer=None, image_processor=None)

    client = anthropic.Anthropic()

    all_responses = []
    for ind, entry in enumerate(tqdm.tqdm(dataset)):
        image_paths, images, prompt, text_label, correct_answer, answer_choices, datatype = entry

        prompt = prompt.split("Human: Answer in natural language.")[-1].split("###Assistant")[0]

        response = get_caption(image_paths, prompt)

        # pdb.set_trace()

        response_text = response.content[0].text
        # print(response_text)
        # Save the response
        all_responses.append({
          "prompt": prompt,
          "text_label": text_label,
          "image_path": image_paths,
          "response": response_text,
          "answer": correct_answer,
          "answer_choices": answer_choices,
          "dataset": datatype
        })
        # pdb.set_trace()

        # Save the responses
        if ind % 100 == 0:
            with open("claude_complexreasoning_response.json", "w") as f:
                json.dump(all_responses, f)

        if ind>=4000:
            break
      