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

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

import sys
sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/")
sys.path.append('../../')
from dataloaders import ProcTHOR_reasoning

import os



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

    
    # load the processor
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )


    all_responses = []
    for ind, entry in enumerate(tqdm.tqdm(dataset)):
        image_paths, images, prompt, text_label, correct_answer, answer_choices, datatype = entry

        prompt = prompt.split("Human: Answer in natural language.")[-1].split("###Assistant")[0]

        # process the image and text
        inputs = processor.process(
            images=images,
            text=prompt
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # pdb.set_trace()

        response_text = generated_text
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
          with open("responses/molmo_complexreasoning_response.json", "w") as f:
              json.dump(all_responses, f)

        if ind>=4000:
            break
      