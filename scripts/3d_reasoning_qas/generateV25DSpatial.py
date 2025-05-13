from collections import defaultdict
import json
import torch
import numpy as np
from PIL import Image
import os
import random
import tqdm
import shutil
from PIL import ImageDraw, ImageFont
from generateGQASpatial import add_red_dot_with_text, add_box_dot_with_color
import csv
import tqdm
import pdb


def generateQAfromAnn(obj_data, obj_count, image, imid, gqa_im_path, mark_AB=False):

    qa_pairs = []
    if len(obj_data) < 2:
        return qa_pairs, ""
    run_condition = True
    max_tries = 100
    im_width, im_height = image.size
    while run_condition:
        max_tries -= 1
        if max_tries < 0:
            break
        obj1, obj2 = random.sample(obj_data, 2)
        if obj1[0] == obj2[0]:
            if random.random() > 0.5:
                continue
        # if one object bbox inside other object bbox then skip
        obj1_bbox = [obj1[1], obj1[2], obj1[1] + obj1[3], obj1[2] + obj1[4]]
        obj2_bbox = [obj2[1], obj2[2], obj2[1] + obj2[3], obj2[2] + obj2[4]]

        obj1_in_obj2 = obj1_bbox[0] > obj2_bbox[0] and obj1_bbox[1] > obj2_bbox[1] and obj1_bbox[2] < obj2_bbox[2] and obj1_bbox[3] < obj2_bbox[3]
        obj2_in_obj1 = obj2_bbox[0] > obj1_bbox[0] and obj2_bbox[1] > obj1_bbox[1] and obj2_bbox[2] < obj1_bbox[2] and obj2_bbox[3] < obj1_bbox[3]
        

        if obj1_in_obj2 or obj2_in_obj1:
            continue
        
        run_condition = False

    if max_tries <= 0:
        return qa_pairs, ""

    obj1_desc = obj1[0]
    obj2_desc = obj2[0]

    obj1_x = obj1[1]*im_width
    obj1_y = obj1[2]*im_height
    obj1_w = obj1[3]*im_width
    obj1_h = obj1[4]*im_height

    obj2_x = obj2[1]*im_width
    obj2_y = obj2[2]*im_height
    obj2_w = obj2[3]*im_width
    obj2_h = obj2[4]*im_height

    obj1 = (obj1_desc, obj1_x, obj1_y, obj1_w, obj1_h, obj1[5])
    obj2 = (obj2_desc, obj2_x, obj2_y, obj2_w, obj2_h, obj2[5])

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
    
    return qa_pairs, marked_im_path



if __name__=="__main__":
    vis=True
    generate=True
    qa_json_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/vrd_qa_data.json"
    html_path = "/net/cs-nfs/home/grad2/array/public_html/research/r2d3/multi_qa_ims/vrd_qa_data.html"

    # make ann data to feed into the generateQA function

    # load the ann data
    vrd_path = "/projectnb/ivc-ml/array/data/VRD25/data"
    vrd_image_path = "/projectnb/ivc-ml/array/data/VRD25/images"
    
    if generate:
        # read object class descriptiosn csv as list
        obj_desc = {}
        with open(os.path.join(vrd_path, "oidv6-class-descriptions.csv"), 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                obj_desc[row[0]] = row[1]


        # read object csv file as list
        object_data = []
        with open(os.path.join(vrd_path, "within_image_objects_train.csv"), 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                object_data.append(row)

        vrd_data = []
        with open(os.path.join(vrd_path, "within_image_vrd_train.csv"), 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                vrd_data.append(row)

        imid2obj = defaultdict(dict)
        for entry in object_data[1:]:
            # image_id object_id	entity	xmin	xmax	ymin	ymax are entries
            obj_name= obj_desc[entry[2]]
            object_id = entry[1]
            imid2obj[entry[0]][object_id] = [obj_name, entry[3], entry[4], entry[5], entry[6]]
            

        obj_data= defaultdict(list)
        all_obj_count = defaultdict(lambda: defaultdict(int))
        for entry in vrd_data:
            # image_id_1 object_id_1	image_id_2	object_id_2	distance	occlusion	raw_distance	raw_occlusion

            image_path = os.path.join(vrd_image_path, entry[0]+".jpg")
            if not os.path.exists(image_path):
                continue

            obj_details = imid2obj[entry[0]][entry[1]]

            obj_desc = obj_details[0]
            obj_pos_x = float(obj_details[1])
            obj_pos_y = float(obj_details[3])
            obj_height = float(obj_details[4]) - float(obj_details[3])
            obj_width = float(obj_details[2]) - float(obj_details[1])


            obj_data[image_path].append((obj_desc, obj_pos_x, obj_pos_y, obj_width, obj_height, 5))

            obj_details_2 = imid2obj[entry[0]][entry[3]]
            obj_desc_2 = obj_details_2[0]
            obj_pos_x_2 = float(obj_details_2[1])
            obj_pos_y_2 = float(obj_details_2[3])
            obj_height_2 = float(obj_details_2[4]) - float(obj_details_2[3])
            obj_width_2 = float(obj_details_2[2]) - float(obj_details_2[1])

            depth_2 = int(entry[4])
            if depth_2 in [-1, 0]:
                continue

            if depth_2 == 1:
                depth_2 = 10
            elif depth_2 == 2:
                depth_2 = 0
            elif depth_2 == 3:
                depth_2 = 5
            
            obj_data[image_path].append((obj_desc_2, obj_pos_x_2, obj_pos_y_2, obj_width_2, obj_height_2, depth_2))
            
            all_obj_count[image_path][obj_desc] += 1

        # pdb.set_trace()
        
        # delete all images in image path which are not used in obj_data
        # image_files = os.listdir(vrd_image_path)
        #for image_file in image_files:
        #    if os.path.isdir(os.path.join(vrd_image_path, image_file)):
        #        continue
        #    image_path = os.path.join(vrd_image_path, image_file)
        #    if image_path not in obj_data:
        #        os.remove(image_path)

        # pdb.set_trace()
        # generate the QA
        im_qas_pairs = []
        data_count = 0
        for image_path, obj_list in tqdm.tqdm(obj_data.items()):
            obj_count = all_obj_count[image_path]
            image = Image.open(image_path)
            imid = image_path.split("/")[-1].split(".")[0]
            mark_AB = random.choice([True, False])

            im_path_dir = "/".join(image_path.split("/")[:-1])
            qa_pairs, marked_im_path = generateQAfromAnn(obj_list, obj_count, image, imid, im_path_dir, mark_AB=mark_AB)

            
            if len(qa_pairs) > 0:
                im_qas_pairs.append([marked_im_path, qa_pairs])
                # pdb.set_trace()
                if data_count %100 == 0:
                    json.dump(im_qas_pairs, open(qa_json_path, 'w'))
                
                data_count += 1
                if data_count > 20000:
                    break


    if vis:
        im_qas_pairs = json.load(open(qa_json_path, "r"))
        print("Num samples: ", len(im_qas_pairs))
        # view in html
        html_str = f"<html><head></head><body>"
        public_im_folder = "/net/cs-nfs/home/grad2/array/public_html/research/r2d3/multi_qa_ims/vrd25/"
        if not os.path.exists(public_im_folder):
            os.makedirs(public_im_folder)
        for im_file, qa_pairs in random.sample(im_qas_pairs, 50):
            public_path_im = os.path.join(public_im_folder, im_file.split("/")[-1]) 
            shutil.copyfile(im_file, public_path_im)

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