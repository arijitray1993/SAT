from collections import defaultdict
import json
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import os
import random
import tqdm
import shutil
from PIL import ImageDraw, ImageFont

def add_red_dot_with_text(image, position, text):
    # Load the image
    draw = ImageDraw.Draw(image)

    # Coordinates and radius of the dot
    x, y = position
    radius = 15  # You can adjust the size of the dot

    # Draw the red dot
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='red')

    # Load a font (optional, comment out if not needed)
    #try:
    font = ImageFont.truetype("LiberationSans-Bold.ttf", 13)  # Adjust font and size as needed
    #except IOError:
    #    font = ImageFont.load_default()

    # Calculate text width and height to center it
    text_width = draw.textlength(text, font=font)
    text_x = x - text_width / 2
    text_y = y

    # Draw the text
    draw.text((text_x, text_y), text, fill='white', font=font)

    return image

def add_box_dot_with_color(image, box, color):
    # box coordinate is in x1, y1, x2, y2 format
    draw = ImageDraw.Draw(image)

    x1, y1, x2, y2 = box

    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    return image


def generateQAfromAnn(obj_data, obj_count, image, imid, gqa_im_path, mark_AB=False):

    qa_pairs = []
    if len(obj_data) < 3:
        return qa_pairs
    run_condition = True
    max_tries = 100
    while run_condition:
        max_tries -= 1
        if max_tries < 0:
            break
        obj1, obj2, obj3 = random.sample(obj_data, 3)
        if obj1[0] == obj2[0] or obj1[0] == obj3[0] or obj2[0] == obj3[0]:
            continue
        # if one object bbox inside other object bbox then skip
        obj1_bbox = [obj1[1], obj1[2], obj1[1] + obj1[3], obj1[2] + obj1[4]]
        obj2_bbox = [obj2[1], obj2[2], obj2[1] + obj2[3], obj2[2] + obj2[4]]
        obj3_bbox = [obj3[1], obj3[2], obj3[1] + obj3[3], obj3[2] + obj3[4]]

        obj1_in_obj2 = obj1_bbox[0] > obj2_bbox[0] and obj1_bbox[1] > obj2_bbox[1] and obj1_bbox[2] < obj2_bbox[2] and obj1_bbox[3] < obj2_bbox[3]
        obj2_in_obj1 = obj2_bbox[0] > obj1_bbox[0] and obj2_bbox[1] > obj1_bbox[1] and obj2_bbox[2] < obj1_bbox[2] and obj2_bbox[3] < obj1_bbox[3]
        obj1_in_obj3 = obj1_bbox[0] > obj3_bbox[0] and obj1_bbox[1] > obj3_bbox[1] and obj1_bbox[2] < obj3_bbox[2] and obj1_bbox[3] < obj3_bbox[3]
        obj3_in_obj1 = obj3_bbox[0] > obj1_bbox[0] and obj3_bbox[1] > obj1_bbox[1] and obj3_bbox[2] < obj1_bbox[2] and obj3_bbox[3] < obj1_bbox[3]
        obj2_in_obj3 = obj2_bbox[0] > obj3_bbox[0] and obj2_bbox[1] > obj3_bbox[1] and obj2_bbox[2] < obj3_bbox[2] and obj2_bbox[3] < obj3_bbox[3]
        obj3_in_obj2 = obj3_bbox[0] > obj2_bbox[0] and obj3_bbox[1] > obj2_bbox[1] and obj3_bbox[2] < obj2_bbox[2] and obj3_bbox[3] < obj2_bbox[3]

        if obj1_in_obj2 or obj2_in_obj1 or obj1_in_obj3 or obj3_in_obj1 or obj2_in_obj3 or obj3_in_obj2:
            continue
        
        run_condition = False

    if max_tries <= 0:
        return qa_pairs

    obj1_desc = obj1[0]
    obj2_desc = obj2[0]
    obj3_desc = obj3[0]

    if obj_count[obj1_desc] > -1:
        if mark_AB:
            marked_img = add_red_dot_with_text(image, (obj1[1] + obj1[3]//2, obj1[2] + obj1[4]//2), "A")
            obj1_desc = obj1_desc + " (marked by point A)"
        else:
            marked_img = add_box_dot_with_color(image, (obj1[1], obj1[2], obj1[1] + obj1[3], obj1[2] + obj1[4]), "red")
            obj1_desc = obj1_desc + " (marked by red box)"
    
    if obj_count[obj2_desc] > -1:
        if mark_AB:
            marked_img = add_red_dot_with_text(image, (obj2[1] + obj2[3]//2, obj2[2] + obj2[4]//2), "B")
            obj2_desc = obj2_desc + " (marked by point B)"
        else:
            marked_img = add_box_dot_with_color(image, (obj2[1], obj2[2], obj2[1] + obj2[3], obj2[2] + obj2[4]), "blue")
            obj2_desc = obj2_desc + " (marked by blue box)"
    
    if obj_count[obj3_desc] > -1:
        if mark_AB:
            marked_img = add_red_dot_with_text(image, (obj3[1] + obj3[3]//2, obj3[2] + obj3[4]//2), "C")
            obj3_desc = obj3_desc + " (marked by point C)"
        else:
            marked_img = add_box_dot_with_color(image, (obj3[1], obj3[2], obj3[1] + obj3[3], obj3[2] + obj3[4]), "green")
            obj3_desc = obj3_desc + " (marked by green box)"

    marked_im_path = f"{gqa_im_path}/{imid}_marked.jpg"
    marked_img.save(marked_im_path)

    # left right
    if obj1[1] < obj2[1]:
        question = "Is the {} to the left or right of the {}?".format(obj1_desc, obj2_desc)
        answer_choices = ["left", "right"]

        qa_pairs.append([question, answer_choices])

        question = "Considering the relative positions, where is the {} with respect to the {}?".format(obj1_desc, obj2_desc)
        answer_choices = ["left", "right"]
        qa_pairs.append([question, answer_choices])

    elif obj1[1] > obj2[1]:
        question = "Is the {} to the left or right of the {}?".format(obj1_desc, obj2_desc)
        answer_choices = ["right", "left"]

        qa_pairs.append([question, answer_choices])

        question = "Considering the relative positions, where is the {} with respect to the {}?".format(obj1_desc, obj2_desc)
        answer_choices = ["right", "left"]
        qa_pairs.append([question, answer_choices])

    
    if obj1[1] < obj3[1]:
        question = "Considering the relative positions, is the {} to the left or right of the {}?".format(obj1_desc, obj3_desc)
        answer_choices = ["left", "right"]

        qa_pairs.append([question, answer_choices])
        
    elif obj1[1] > obj3[1]:
        question = "Considering the relative positions, is the {} to the left or right of the {}?".format(obj1_desc, obj3_desc)
        answer_choices = ["right", "left"]

        qa_pairs.append([question, answer_choices])
        


    # above below
    if obj1[2] < obj2[2]:
        question = "Is the {} above or below the {}?".format(obj1_desc, obj2_desc)
        answer_choices = ["above", "below"]

        qa_pairs.append([question, answer_choices])

        question = "Considering the relative positions, where is the {} with respect to the {}?".format(obj1_desc, obj2_desc)
        answer_choices = ["above", "below"]
        qa_pairs.append([question, answer_choices])

    elif obj1[2] > obj2[2]:
        question = "Is the {} above or below the {}?".format(obj1_desc, obj2_desc)
        answer_choices = ["below", "above"]

        qa_pairs.append([question, answer_choices])

        question = "Considering the relative positions, where is the {} with respect to the {}?".format(obj1_desc, obj2_desc)
        answer_choices = ["below", "above"]
        qa_pairs.append([question, answer_choices])

    if obj1[2] < obj3[2]:
        question = "Considering the relative positions, is the {} above or below the {}?".format(obj1_desc, obj3_desc)
        answer_choices = ["above", "below"]

        qa_pairs.append([question, answer_choices])
    elif obj1[2] > obj3[2]:
        question = "Considering the relative positions, is the {} above or below the {}?".format(obj1_desc, obj3_desc)
        answer_choices = ["below", "above"]

        qa_pairs.append([question, answer_choices])


    # behind infront
    if obj1[5] < obj2[5]:
        question = "Is the {} further away or in front of the {}?".format(obj1_desc, obj2_desc)
        answer_choices = ["in front", "further away"]

        qa_pairs.append([question, answer_choices])

        question = "Is {} behind {}?".format(obj1_desc, obj2_desc)
        answer_choices = ["no", "yes"]

        qa_pairs.append([question, answer_choices])

        question = "Is {} in front of {}?".format(obj1_desc, obj2_desc)
        answer_choices = ["yes", "no"]

        qa_pairs.append([question, answer_choices])

        if mark_AB:
            question = "Which point is closer to the camera taking this photo, point A or point B?"
            answer_choices = ["A", "B"]
            qa_pairs.append([question, answer_choices])
        else:
            question = "Which object is closer to the camera taking this photo, the {} or the {}?".format(obj1_desc, obj2_desc)
            answer_choices = [obj1_desc, obj2_desc]
            qa_pairs.append([question, answer_choices])


    elif obj1[5] > obj2[5]:
        question = "Is the {} further away or in front of the {}?".format(obj1_desc, obj2_desc)
        answer_choices = ["further away", "in front"]

        qa_pairs.append([question, answer_choices])

        question = "Is {} behind {}?".format(obj1_desc, obj2_desc)
        answer_choices = ["yes", "no"]

        qa_pairs.append([question, answer_choices])

        question = "Is {} in front of {}?".format(obj1_desc, obj2_desc)
        answer_choices = ["no", "yes"]

        qa_pairs.append([question, answer_choices])

        if mark_AB:
            question = "Which point is closer to the camera taking this photo, point A or point B?"
            answer_choices = ["B", "A"]
            qa_pairs.append([question, answer_choices])
        else:
            question = "Which object is closer to the camera taking this photo, the {} or the {}?".format(obj1_desc, obj2_desc)
            answer_choices = [obj2_desc, obj1_desc]
            qa_pairs.append([question, answer_choices])

    # relative distance
    # assume depth is z coordinate and get 3d coordinate for object
    obj1_3d_loc = [obj1[1], obj1[2], obj1[5]]
    obj2_3d_loc = [obj2[1], obj2[2], obj2[5]]
    obj3_3d_loc = [obj3[1], obj3[2], obj3[5]]

    obj12_distance = np.linalg.norm(np.array(obj1_3d_loc) - np.array(obj2_3d_loc))
    obj13_distance = np.linalg.norm(np.array(obj1_3d_loc) - np.array(obj3_3d_loc))

    if obj12_distance < obj13_distance:
        question = "Estimate the real world distances between objects in the image. Which object is closer to the {}, the {} or the {}?".format(obj1_desc, obj2_desc, obj3_desc)
        answer  = [obj2_desc, obj3_desc]
        qa_pairs.append([question, answer])
    
    elif obj12_distance > obj13_distance:
        question = "Estimate the real world distances between objects in the image. Which object is closer to the {}, the {} or the {}?".format(obj1_desc, obj2_desc, obj3_desc)
        answer  = [obj3_desc, obj2_desc]
        qa_pairs.append([question, answer])
    
    return qa_pairs



if __name__=="__main__":

    split = "train"
    scenegraph_data_path = f'/projectnb/ivc-ml/array/data/GQA/train_sceneGraphs.json'
    qa_json_path = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GQA_spatial_qas_{split}.json'
    html_path = "/net/cs-nfs/home/grad2/array/public_html/research/r2d3/multi_qa_ims/GQA_spatial_qas.html"
    vis = True
    stats = False
    generate = True
    load_progress = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the GQA data
    gqa_data = json.load(open(scenegraph_data_path))
    gqa_im_path = "/projectnb/ivc-ml/array/data/GQA/images"

    # depth model
    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)

    
    if generate:
        im_qas_pairs = []
        if load_progress:
            im_qas_pairs = json.load(open(qa_json_path))
        
        data_count = 0
        for ind, imid in enumerate(tqdm.tqdm(gqa_data)):
            if ind < 19500:
                continue
            qa_pairs = []
            mark_AB = random.random() < 0.5

            image_file = os.path.join(gqa_im_path, f"{imid}.jpg")

            width = gqa_data[imid]['width']
            height = gqa_data[imid]['height']
            

            image = Image.open(image_file).convert("RGB")
        
            # prepare image for the model
            inputs = image_processor(images=image, return_tensors="pt")

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            # interpolate to original size
            predicted_depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            obj_data = []
            obj_count = defaultdict(int)
            for obj_key in gqa_data[imid]['objects']:
                obj_entry = gqa_data[imid]['objects'][obj_key]
                obj_name = obj_entry['name']
                if obj_name.lower() in ['wall', 'floor', 'ceiling', 'water', 'road']: # these are usually noisy in terms of bbox
                    continue
                
                attributes = obj_entry['attributes']
                obj_desc = " ".join(attributes) + " " + obj_name

                size = obj_entry['h'] * obj_entry['w']
                norm_size = size / (width * height)

                obj_count[obj_name] += 1

                if norm_size < 0.1: 
                    continue

                # get depth at the x, y location
                x = obj_entry['x']
                y = obj_entry['y']

                if x < 0 or x >= predicted_depth.shape[3] or y < 0 or y >= predicted_depth.shape[2]:
                    continue
                depth = predicted_depth[0, 0, y, x].item()
                
                obj_data.append([obj_desc, obj_entry['x'], obj_entry['y'], obj_entry['w'], obj_entry['h'], depth])

            # use x,y,depth to make left right, above, below, behind, infront questions
            qa_pairs = generateQAfromAnn(obj_data, obj_count, image, imid, gqa_im_path, mark_AB)
            
            if len(qa_pairs) > 0:
                im_qas_pairs.append([image_file, qa_pairs])
                data_count += 1
                if data_count %10 == 0:
                    json.dump(im_qas_pairs, open(qa_json_path, 'w'))

        
        json.dump(im_qas_pairs, open(qa_json_path, 'w'))

    if vis:
        all_im_qas = json.load(open(qa_json_path, "r"))
        print("Num samples: ", len(all_im_qas))
        # view in html
        html_str = f"<html><head></head><body>"
        public_im_folder = "/net/cs-nfs/home/grad2/array/public_html/research/r2d3/multi_qa_ims/gqa_spatial/"
        if not os.path.exists(public_im_folder):
            os.makedirs(public_im_folder)
        for im_file, qa_pairs in random.sample(im_qas_pairs, 50):
            public_path_im = os.path.join(public_im_folder, im_file.split("/")[-1]) 
            shutil.copyfile(im_file.replace(".jpg", "_marked.jpg"), public_path_im)

            for sample_count, qa_pair in enumerate(qa_pairs):
                question, answer_choices = qa_pair
                html_str += f"<p>{question}</p>"
                html_im_url = "https://cs-people.bu.edu/array/"+public_path_im.split("/net/cs-nfs/home/grad2/array/public_html/")[-1]
                html_str += f"<img src='{html_im_url}' style='width: 300px; height: 300px;'>"
                for ans in answer_choices:
                    html_str += f"<p>{ans}</p>"
                html_str += "<hr>"
        html_str += "</body></html>"
        with open(html_path, "w") as f:
            f.write(html_str)
