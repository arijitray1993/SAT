from collections import defaultdict
import json
import prior
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from PIL import Image
import random
from pprint import pprint
import pdb
import math
from PIL import ImageDraw, ImageFont

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import ast
import tqdm
import copy
import os
import numpy as np
import yaml
from ai2thor.platform import CloudRendering
import shutil
import sys
sys.path.append("../../")
from utils.ai2thor_utils import generate_program_from_roomjson, generate_room_programs_from_house_json, make_house_from_cfg

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


def get_current_state(controller):
    nav_visible_objects = controller.step("GetVisibleObjects", maxDistance=5).metadata["actionReturn"]
    nav_all_visible_objects = controller.step("GetVisibleObjects", maxDistance=20).metadata["actionReturn"]
    nav_visible_objects = [obj for obj in nav_visible_objects if objid2assetid[obj]!=""] # these are the visible object asset ids in the scene
    
    bboxes = controller.last_event.instance_detections2D
    vis_obj_to_size = {}
    for obj_id in bboxes:
        vis_obj_to_size[obj_id] = (bboxes[obj_id][2] - bboxes[obj_id][0])*(bboxes[obj_id][3] - bboxes[obj_id][1])


    objid2info = {}
    objdesc2cnt = defaultdict(int)
    for obj_entry in controller.last_event.metadata['objects']:
        obj_name = obj_entry['name']
        obj_type = obj_entry['objectType']
        asset_id = obj_entry['assetId']
        obj_id = obj_entry['objectId']
        
        distance = obj_entry['distance']
        pos = np.array([obj_entry['position']['x'], obj_entry['position']['y'], obj_entry['position']['z']])
        rotation = obj_entry['rotation']
        desc = assetid2desc.get(asset_id, obj_type)
        moveable = obj_entry['moveable'] or obj_entry['pickupable']
        
        asset_size_xy = vis_obj_to_size.get(obj_entry['objectId'], 0)
        asset_pos_box = bboxes.get(obj_entry['objectId'], None)
        if asset_pos_box is not None:
            asset_pos_xy = [(asset_pos_box[0]+asset_pos_box[2])/2, (asset_pos_box[1]+asset_pos_box[3])/2]
        else:
            continue

        parent = obj_entry.get('parentReceptacles')
        if parent is not None:
            if len(parent) > 0:
                parent = parent[-1]
                
        
        is_receptacle = obj_entry['receptacle']
        objid2info[obj_id] = (obj_name, obj_type, distance, pos, rotation, desc, moveable, parent, asset_size_xy, is_receptacle, asset_pos_xy, asset_pos_box)
        if obj_id in nav_all_visible_objects:
            objdesc2cnt[obj_type] += 1

    
    visible_objs = []
    for objid in nav_visible_objects:
        if objid in objid2info:
            if objid2info[objid][8]>2500:
                visible_objs.append(objid)

    # pdb.set_trace()
    return nav_visible_objects, objid2info, objdesc2cnt, visible_objs


if __name__ == "__main__":
    
    split="train"
    if split == "train":
        qa_im_path = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/multi_qa_images/spatial_new_v2/'
        qa_json_path = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_spatial_qas_new_split_v2_split2.json'
    else:
        qa_im_path = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/multi_qa_images/spatial_new_{split}_v2/'
        qa_json_path = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_spatial_qas_new_{split}_v2.json'
    
    html_vis_file = '/net/cs-nfs/home/grad2/array/public_html/research/r2d3/multi_qa_ims/3d_spatial_qas_new_v2.html'
    generate=True
    vis=False
    

    if generate:
        asset_id_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json", "r"))

        if not os.path.exists(qa_im_path):
            os.makedirs(qa_im_path)
        #assetid2desc = {}
        #for image_file, asset_name, object_class, caption in asset_id_desc:
        #    assetid2desc[asset_name] = caption

        assetid2desc = {}
        for asset in asset_id_desc:
            entries = asset_id_desc[asset]
            captions = []
            for im, obj, desc in entries:
                desc = desc.strip().lower().replace(".", "")
                captions.append(desc)
            assetid2desc[asset] = random.choice(captions)

        dataset = prior.load_dataset("procthor-10k")
        all_im_qas = []
        # all_im_qas = json.load(open(qa_json_path, "r"))

        for house_ind, house in enumerate(tqdm.tqdm(dataset[split])):
            if house_ind<5000:
                continue

            house_json = house
            try:
                controller = Controller(scene=house, width=300, height=300, quality="Low", platform=CloudRendering) # quality="Ultra", renderInstanceSegmentation=True, visibilityDistance=30)
            except:
                print("Cannot render environment, continuing")
                # pdb.set_trace()
                continue

            # get camera position with max objects visible
            try:
                event = controller.step(action="GetReachablePositions")
            except:
                print("Cannot get reachable positions, continuing")
                controller.stop()
                continue
            reachable_positions = event.metadata["actionReturn"]

            if len(reachable_positions)<10:
                print("No reachable positions, continuing")
                controller.stop()
                continue

            random_positions = []
            for cam_pos in random.sample(reachable_positions, 10):
                
                cam_rot = random.choice(range(360))
                try:
                    controller.step(action="Teleport", position=cam_pos, rotation=cam_rot)
                except:
                    print("Cannot teleport, continuing")
                    continue

                nav_visible_objects = controller.step(
                    "GetVisibleObjects",
                    maxDistance=5,
                ).metadata["actionReturn"]

                random_positions.append((cam_pos, cam_rot, len(nav_visible_objects)))
            
            if len(random_positions)==0:
                print("No objects visible, continuing")
                controller.stop()
                continue

            random_positions = sorted(random_positions, key=lambda x: x[2], reverse=True)
            
            controller.stop()

            if not os.path.exists(qa_im_path + f"{house_ind}"):
                os.makedirs(qa_im_path + f"{house_ind}")

        
            sample_count = 0
            for cam_pos, cam_rot, _ in random_positions[:1]:
                qa_pair_choices = []
                try:
                    controller = Controller(scene=house_json, width=512, height=512, quality="Ultra", renderInstanceSegmentation=True, platform=CloudRendering)
                except:
                    print("Cannot render environment, continuing")
                    # pdb.set_trace()
                    continue
                try:
                    controller.step(action="Teleport", position=cam_pos, rotation=cam_rot)
                except:
                    print("Cannot teleport, continuing")
                    controller.stop()
                    continue

                objid2assetid = {}
                for obj in controller.last_event.metadata['objects']:
                    objid2assetid[obj['objectId']] = obj['assetId']

                img_view = Image.fromarray(controller.last_event.frame)

                new_path_init = qa_im_path + f"{house_ind}/{sample_count}_0.jpg"
                img_view.save(new_path_init)
                new_path_init_marked = qa_im_path + f"{house_ind}/{sample_count}_0_marked.jpg"

                # get visible objects
                nav_visible_obj_assets, objid2info, objdesc2cnt, visible_obj_assets = get_current_state(controller)
                
                # choose random object
                if len(visible_obj_assets) < 3:
                    print("No moveable objects, continuing")
                    controller.stop()
                    continue
                
                obj1, another_obj, another_obj_2 = random.sample(visible_obj_assets, 3)

                mark_AB = random.random() < 0.5

                # mark obj1 in the image
                obj1_pos_xy = objid2info[obj1][10]

                if mark_AB:
                    marked_img = add_red_dot_with_text(img_view, (obj1_pos_xy[0], obj1_pos_xy[1]), "A")
                else:
                    marked_img = add_box_dot_with_color(img_view, objid2info[obj1][11], "red")

                # pdb.set_trace()
                obj1_cnt = objdesc2cnt[objid2info[obj1][1]]
                obj_name = objid2info[obj1][0]
                obj_desc = objid2info[obj1][5]
                obj1_type = objid2info[obj1][1]

            
                # mark another obj in the image
                another_obj_pos_xy = objid2info[another_obj][10]
                if another_obj_pos_xy is None:
                    controller.stop()
                    continue

                if mark_AB:
                    marked_img = add_red_dot_with_text(marked_img, (another_obj_pos_xy[0], another_obj_pos_xy[1]), "B")
                else:
                    marked_img = add_box_dot_with_color(marked_img, objid2info[another_obj][11], "blue")
                
                another_obj_desc = objid2info[another_obj][5]
                another_obj_cnt = objdesc2cnt[objid2info[another_obj][1]]
                another_obj_pos = objid2info[another_obj][3]

                # mark another obj 2 in the image
                another_obj_2_pos_xy = objid2info[another_obj_2][10]
                if another_obj_2_pos_xy is None:
                    controller.stop()
                    continue

                if mark_AB:
                    marked_img = add_red_dot_with_text(marked_img, (another_obj_2_pos_xy[0], another_obj_2_pos_xy[1]), "C")
                else:
                    marked_img = add_box_dot_with_color(marked_img, objid2info[another_obj_2][11], "green")
                
                another_obj_2_desc = objid2info[another_obj_2][5]
                another_obj_2_cnt = objdesc2cnt[objid2info[another_obj_2][1]]
                another_obj_2_pos = objid2info[another_obj_2][3]

                if random.random() < 0.5:
                    if mark_AB:
                        obj_desc = f"{obj_desc} (marked A)"
                        another_obj_desc = f"{another_obj_desc} (marked B)"
                        another_obj_2_desc = f"{another_obj_2_desc} (marked C)"
                    else:
                        obj_desc = f"{obj_desc} (highlighted by a red box)"
                        another_obj_desc = f"{another_obj_desc} (highlighted by a blue box)"
                        another_obj_2_desc = f"{another_obj_2_desc} (highlighted by a green box)"
                else:
                    if obj_desc == another_obj_desc:
                        if mark_AB:
                            obj_desc = f"{obj_desc} (marked A)"
                            another_obj_desc = f"{another_obj_desc} (marked B)"
                        else:
                            obj_desc = f"{obj_desc} (highlighted by a red box)"
                            another_obj_desc = f"{another_obj_desc} (highlighted by a blue box)"
                    
                    if obj_desc == another_obj_2_desc:
                        if mark_AB:
                            obj_desc = f"{obj_desc} (marked A)"
                            another_obj_2_desc = f"{another_obj_2_desc} (marked C)"
                        else:
                            obj_desc = f"{obj_desc} (highlighted by a red box)"
                            another_obj_2_desc = f"{another_obj_2_desc} (highlighted by a green box)"
                    
                    if another_obj_desc == another_obj_2_desc:
                        if mark_AB:
                            another_obj_desc = f"{another_obj_desc} (marked B)"
                            another_obj_2_desc = f"{another_obj_2_desc} (marked C)"
                        else:
                            another_obj_desc = f"{another_obj_desc} (highlighted by a blue box)"
                            another_obj_2_desc = f"{another_obj_2_desc} (highlighted by a green box)"


                # this is to normalize the camera to 0 angle. 
                new_pos = cam_pos
                new_rot = cam_rot
                
                xz_rotation_matrix = np.array([[np.cos(math.radians(new_rot)), -np.sin(math.radians(new_rot))], [np.sin(math.radians(new_rot)), np.cos(math.radians(new_rot))]])


                # if two another objects are too close to each other, continue
                if np.linalg.norm(another_obj_pos - another_obj_2_pos) < 0.5:
                    controller.stop()
                    continue
                    
                if np.linalg.norm(another_obj_pos - objid2info[obj1][3]) < 0.5:
                    controller.stop()
                    continue    
                

                #### count questions ####
                question = f"How many {obj1_type}s are visible in the scene?"
                answer = f"{obj1_cnt}"
                incorrect_answer = f"{obj1_cnt - random.choice(range(1, obj1_cnt+1))}"
                incorrect_answer_2 = f"{obj1_cnt + 3}"
                incorrect_answer_3 = f"{obj1_cnt + 1}"
                incorrect_answer_4 = f"{obj1_cnt + 2}"
                
                answer_choices = (answer, incorrect_answer, incorrect_answer_2, incorrect_answer_3, incorrect_answer_4)
                image_order = (new_path_init,)
                qa_pair_choices.append((question, image_order, answer_choices))

                question = f"How many {objid2info[another_obj][1]}s are visible in the scene?"
                answer = f"{another_obj_cnt}"
                incorrect_answer = f"{obj1_cnt - random.choice(range(1, obj1_cnt+1))}"
                incorrect_answer_2 = f"{obj1_cnt + 3}"
                incorrect_answer_3 = f"{obj1_cnt + 1}"
                incorrect_answer_4 = f"{obj1_cnt + 2}"

                answer_choices = (answer, incorrect_answer, incorrect_answer_2, incorrect_answer_3, incorrect_answer_4)
                image_order = (new_path_init,)
                qa_pair_choices.append((question, image_order, answer_choices))

            
                question = f"How many {objid2info[another_obj_2][1]}s are visible in the scene?"
                answer = f"{another_obj_2_cnt}"
                incorrect_answer = f"{obj1_cnt - random.choice(range(1, obj1_cnt+1))}"
                incorrect_answer_2 = f"{obj1_cnt + 3}"
                incorrect_answer_3 = f"{obj1_cnt + 1}"
                incorrect_answer_4 = f"{obj1_cnt + 2}"

                answer_choices = (answer, incorrect_answer, incorrect_answer_2, incorrect_answer_3, incorrect_answer_4)
                image_order = (new_path_init,)
                qa_pair_choices.append((question, image_order, answer_choices))


                ##### simple which object is behind the other object questions#####

                # obj 1  and another obj distance
                obj1_distance = objid2info[obj1][2]
                another_obj_distance = objid2info[another_obj][2]
                another_obj2_distance = objid2info[another_obj_2][2]

                if obj1_distance > another_obj_distance:
                    if mark_AB:
                        question = f"Which point is closer to the camera taking this photo, point A  or point B?"
                        answer = "B"
                        incorrect_answer = "A"
                    else:
                        question = f"Which object is closer to the camera taking this photo, {obj_desc} or {another_obj_desc}?"
                        answer = f"{another_obj_desc}"
                        incorrect_answer = f"{obj_desc}"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                    
                    if mark_AB:
                        if random.random() < 0.5:
                            question = f"Is {obj_desc} behind {another_obj_desc}?"
                            answer_choices = ("yes", "no")
                        else:
                            question = f"Is {obj_desc} in front of {another_obj_desc}?"
                            answer_choices = ("no", "yes")
                    else:
                        question = f"Is {obj_desc} further away or in front of {another_obj_desc}?"
                        answer_choices = ("further away", "in front")
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))

                
                elif obj1_distance < another_obj_distance:
                    if mark_AB:
                        question = f"Which point is closer to the camera taking this photo, point A  or point B?"
                        answer = "A"
                        incorrect_answer = "B"
                    else:
                        question = f"Which object is closer to the camera taking this photo, {obj_desc} or {another_obj_desc}?"
                        answer = f"{obj_desc}"
                        incorrect_answer = f"{another_obj_desc}"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))

                    if mark_AB:
                        if random.random() < 0.5:
                            question = f"Is {obj_desc} behind {another_obj_desc}?"
                            answer_choices = ("no", "yes")
                        else:
                            question = f"Is {obj_desc} in front of {another_obj_desc}?"
                            answer_choices = ("yes", "no")
                    else:
                        question = f"Is {obj_desc} further away or in front of {another_obj_desc}?"
                        answer_choices = ("in front", "further away")
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                

                if another_obj_distance > another_obj2_distance:
                    if mark_AB:
                        question = f"Which point is closer to the camera taking this photo, point B  or point C?"
                        answer = "C"
                        incorrect_answer = "B"
                    else:
                        question = f"Which object is closer to the camera taking this photo, {another_obj_desc} or {another_obj_2_desc}?"
                        answer = f"{another_obj_2_desc}"
                        incorrect_answer = f"{another_obj_desc}"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                elif another_obj_distance < another_obj2_distance:
                    if mark_AB:
                        question = f"Which point is closer to the camera taking this photo, point B  or point C?"
                        answer = "B"
                        incorrect_answer = "C"
                    else:
                        question = f"Which object is closer to the camera taking this photo, {another_obj_desc} or {another_obj_2_desc}?"
                        answer = f"{another_obj_desc}"
                        incorrect_answer = f"{another_obj_2_desc}"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                
                

                # rotate using rotation matrix to figure out whether another obj is left to right to camera
                obj1_rotated_pos = np.dot(xz_rotation_matrix, np.array([objid2info[obj1][3][0], objid2info[obj1][3][2]]))
                another_obj_rotated_pos = np.dot(xz_rotation_matrix, np.array([another_obj_pos[0], another_obj_pos[2]]))
                another_obj_2_rotated_pos = np.dot(xz_rotation_matrix, np.array([another_obj_2_pos[0], another_obj_2_pos[2]]))

                # between A and B
                if another_obj_rotated_pos[0] > obj1_rotated_pos[0]:
                    direction = "right"
                    wrong_direction = "left"
                    if mark_AB:
                        question = f"Where is {obj_desc} with respect to {another_obj_desc}?"
                    else:
                        question = f"Considering the relative positions, where is {obj_desc} with respect to {another_obj_desc}?"
                    answer_choices = (
                        f"{wrong_direction}", # wrong direction is the correct one here, fix later
                        f"{direction}",
                    )
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))

                    if mark_AB:
                        if random.random() < 0.5:
                            question = f"Is {obj_desc} to the left of {another_obj_desc}?"
                            answer_choices = ("yes", "no")
                            image_order = (new_path_init_marked,)
                            qa_pair_choices.append((question, image_order, answer_choices))
                        else:
                            question = f"Is {obj_desc} to the right of {another_obj_desc}?"
                            answer_choices = ("no", "yes")
                            image_order = (new_path_init_marked,)
                            qa_pair_choices.append((question, image_order, answer_choices))
                    
                elif another_obj_rotated_pos[0] < obj1_rotated_pos[0]:
                    direction = "left"
                    wrong_direction = "right"
                    if mark_AB:
                        question = f"Considering the relative positions, where is {obj_desc} with respect to {another_obj_desc}?"
                    else:
                        question = f"Considering the relative positions, where is {obj_desc} with respect to {another_obj_desc}?"
                    answer_choices = (
                        f"{wrong_direction}", # wrong direction is the correct one here, fix later
                        f"{direction}",
                    )
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                

                # between B and C
                if another_obj_2_rotated_pos[0] > another_obj_rotated_pos[0]:
                    direction = "right"
                    wrong_direction = "left"
                    if mark_AB:
                        question = f"Considering the relative positions, where is {another_obj_desc} with respect to {another_obj_2_desc}?"
                    else:
                        question = f"Considering the relative positions, where is {another_obj_desc} with respect to {another_obj_2_desc}?"
                    answer_choices = (
                        f"{wrong_direction}", # wrong direction is the correct one here, fix later
                        f"{direction}",
                    )
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                elif another_obj_2_rotated_pos[0] < another_obj_rotated_pos[0]:
                    direction = "left"
                    wrong_direction = "right"
                    if mark_AB:
                        question = f"Considering the relative positions, is {another_obj_desc} to the left or right of {another_obj_2_desc}?"
                    else:
                        question = f"Considering the relative positions, where is {another_obj_desc} with respect to {another_obj_2_desc}?"
                    answer_choices = (
                        f"{wrong_direction}", # wrong direction is the correct one here, fix later
                        f"{direction}",
                    )
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))

                    if mark_AB:
                        if random.random() < 0.5:
                            question = f"Is {another_obj_desc} to the right of {another_obj_2_desc}?"
                            answer_choices = ("yes", "no")
                            image_order = (new_path_init_marked,)
                            qa_pair_choices.append((question, image_order, answer_choices))
                        else:
                            question = f"Is {another_obj_desc} to the left of {another_obj_2_desc}?"
                            answer_choices = ("no", "yes")
                            image_order = (new_path_init_marked,)
                            qa_pair_choices.append((question, image_order, answer_choices))
                

                # between A and C
                if another_obj_2_rotated_pos[0] > obj1_rotated_pos[0]:
                    direction = "right"
                    wrong_direction = "left"
                    if mark_AB:
                        question = f"Considering the relative positions, is {obj_desc} to the left or right of {another_obj_2_desc}?"
                    else:
                        question = f"Considering the relative positions, where is {obj_desc} with respect to {another_obj_2_desc}?"
                    answer_choices = (
                        f"{wrong_direction}", # wrong direction is the correct one here, fix later
                        f"{direction}",
                    )
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                elif another_obj_2_rotated_pos[0] < obj1_rotated_pos[0]:
                    direction = "left"
                    wrong_direction = "right"
                    if mark_AB:
                        question = f"Considering the relative positions, is {obj_desc} to the left or right of {another_obj_2_desc}?"
                    else:
                        question = f"Considering the relative positions, where is {obj_desc} with respect to {another_obj_2_desc}?"
                    answer_choices = (
                        f"{wrong_direction}", # wrong direction is the correct one here, fix later
                        f"{direction}",
                    )
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))

                # touching questions:
                # chack parents of the objects
                # todo



                ### above below questions

                # between B and A
                if another_obj_rotated_pos[1] > obj1_rotated_pos[1]:
                    direction_ab = "above"
                    wrong_direction_ab = "below"
                    if mark_AB:
                        question = f"Considering the relative positions, is {another_obj_desc} above or below {obj_desc}?"
                    else:
                        question = f"Considering the relative positions, where is {another_obj_desc} with respect to {obj_desc}?"
                        
                    answer = f"{direction_ab}"
                    incorrect_answer = f"{wrong_direction_ab}"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                elif another_obj_rotated_pos[1] < obj1_rotated_pos[1]:
                    direction_ab = "below"
                    wrong_direction_ab = "above"
                    if mark_AB:
                        question = f"Considering the relative positions, is {another_obj_desc} above or below {obj_desc}?"
                    else:
                        question = f"Considering the relative positions, where is {another_obj_desc} with respect to {obj_desc}?"
                        
                    answer = f"{direction_ab}"
                    incorrect_answer = f"{wrong_direction_ab}"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                

                # between C and B
                if another_obj_2_rotated_pos[1] > another_obj_rotated_pos[1]:
                    direction_bc = "above"
                    wrong_direction_bc = "below"
                    if mark_AB:
                        question = f"Considering the relative positions, is {another_obj_2_desc} above or below {another_obj_desc}?"
                    else:
                        question = f"Considering the relative positions, where is {another_obj_2_desc} with respect to {another_obj_desc}?"
                    
                    answer = f"{direction_bc}"
                    incorrect_answer = f"{wrong_direction_bc}"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                elif another_obj_2_rotated_pos[1] < another_obj_rotated_pos[1]:
                    direction_bc = "below"
                    wrong_direction_bc = "above"
                    if mark_AB:
                        question = f"Considering the relative positions, is {another_obj_2_desc} above or below {another_obj_desc}?"
                    else:
                        question = f"Considering the relative positions, where is {another_obj_2_desc} with respect to {another_obj_desc}?"
                    
                    answer = f"{direction_bc}"
                    incorrect_answer = f"{wrong_direction_bc}"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                
                

                # between C and A
                if another_obj_2_rotated_pos[1] > obj1_rotated_pos[1]:
                    direction_ac = "above"
                    wrong_direction_ac = "below"
                    if mark_AB:
                        question = f"Considering the relative positions, is {another_obj_2_desc} above or below {obj_desc}?"
                    else:
                        question = f"Considering the relative positions, is {another_obj_2_desc} above or below {obj_desc}?"
                    
                    answer = f"{direction_ac}"
                    incorrect_answer = f"{wrong_direction_ac}"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                elif another_obj_2_rotated_pos[1] < obj1_rotated_pos[1]:
                    direction_ac = "below"
                    wrong_direction_ac = "above"
                    if mark_AB:
                        question = f"Considering the relative positions, is {another_obj_2_desc} above or below {obj_desc}?"
                    else:
                        question = f"Considering the relative positions, is {another_obj_2_desc} above or below {obj_desc}?"
                    
                    answer = f"{direction_ac}"
                    incorrect_answer = f"{wrong_direction_ac}"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))


                # above below based on 3d height and not 2d position in the image

                #between A and B
                if objid2info[obj1][3][1] > another_obj_pos[1]:
                    if mark_AB:
                        question = f"Consider the 3D positions of the objects in the scene and not just the 2D positions in the image. Is the centerpoint of  {obj_desc} at a higher height than {another_obj_desc}?"
                    else:
                        question = f"Consider the 3D positions of the objects in the scene and not just the 2D positions in the image. Is the centerpoint of  {obj_desc} at a higher height than {another_obj_desc}?"
                    answer = "yes"
                    incorrect_answer = "no"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                elif objid2info[obj1][3][1] < another_obj_pos[1]:
                    if mark_AB:
                        question = f"Consider the 3D positions of the objects in the scene and not just the 2D positions in the image. Is the centerpoint of  {obj_desc} at a higher height than {another_obj_desc}?"
                    else:
                        question = f"Consider the 3D positions of the objects in the scene and not just the 2D positions in the image. Is the centerpoint of  {obj_desc} at a higher height than {another_obj_desc}?"
                    answer = "no"
                    incorrect_answer = "yes"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))

                # between B and C
                if another_obj_pos[1] > another_obj_2_pos[1]:
                    if mark_AB:
                        question = f"Consider the 3D positions of the objects in the scene and not just the 2D positions in the image. Is the centerpoint of  {another_obj_desc} at a higher height than {another_obj_2_desc}?"
                    else:
                        question = f"Consider the 3D positions of the objects in the scene and not just the 2D positions in the image. Is the centerpoint of  {another_obj_desc} at a higher height than {another_obj_2_desc}?"
                    answer = "yes"
                    incorrect_answer = "no"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                elif another_obj_pos[1] < another_obj_2_pos[1]:
                    if mark_AB:
                        question = f"Consider the 3D positions of the objects in the scene and not just the 2D positions in the image. Is the centerpoint of  {another_obj_desc} at a higher height than {another_obj_2_desc}?"
                    else:
                        question = f"Consider the 3D positions of the objects in the scene and not just the 2D positions in the image. Is the centerpoint of  {another_obj_desc} at a higher height than {another_obj_2_desc}?"
                    answer = "no"
                    incorrect_answer = "yes"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices)) 


                ###### which another obj is closer to obj 1 ######
                if np.linalg.norm(another_obj_pos - objid2info[obj1][3]) < np.linalg.norm(another_obj_2_pos - objid2info[obj1][3]):
                    if mark_AB:
                        question = f"Estimate the real world distances between objects in the image. Which object is closer to {obj_desc}, {another_obj_desc} or {another_obj_2_desc}?"
                    else:
                        question = f"Which object is closer to {obj_desc}, {another_obj_desc} or {another_obj_2_desc}?"
                    answer = f"{another_obj_desc}"
                    incorrect_answer = f"{another_obj_2_desc}"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))
                elif np.linalg.norm(another_obj_pos - objid2info[obj1][3]) > np.linalg.norm(another_obj_2_pos - objid2info[obj1][3]):
                    if mark_AB:
                        question = f"Estimate the real world distances between objects in the image. Which object is closer to {obj_desc}, {another_obj_desc} or {another_obj_2_desc}?"
                    else:
                        question = f"Which object is closer to {obj_desc}, {another_obj_desc} or {another_obj_2_desc}?" 
                    answer = f"{another_obj_2_desc}"
                    incorrect_answer = f"{another_obj_desc}"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init_marked,)
                    qa_pair_choices.append((question, image_order, answer_choices))

                

                if len(qa_pair_choices) > 0:
                    marked_img.save(new_path_init_marked)
                    all_im_qas.append((house_ind, cam_pos, cam_rot, qa_pair_choices))
                
                # pdb.set_trace()
                
                sample_count += 1
            
                controller.stop()
            if house_ind % 100 == 0:
                json.dump(all_im_qas, open(qa_json_path, "w"))

    if vis:
        all_im_qas = json.load(open(qa_json_path, "r"))
        # view in html
        if html_vis_file is not None:
            html_str = f"<html><head></head><body>"
            public_im_folder = "/net/cs-nfs/home/grad2/array/public_html/research/r2d3/multi_qa_ims/spatial/"
            for house_ind, _, _, qa_pairs in all_im_qas[:50]:
                if not os.path.exists(public_im_folder + f"{house_ind}"):
                    os.makedirs(public_im_folder + f"{house_ind}")
                    
                for sample_count, qa_pair in enumerate(qa_pairs):
                    question, image_order, answer_choices = qa_pair
                    html_str += f"<p>{question}</p>"
                    for im in image_order:
                        if im == "":
                            continue
                        public_path_im = public_im_folder + "/".join(im.split("/")[-2:])
                        shutil.copyfile(im, public_path_im)
                        html_im_url = "https://cs-people.bu.edu/array/"+public_path_im.split("/net/cs-nfs/home/grad2/array/public_html/")[-1]
                        html_str += f"<img src='{html_im_url}' style='width: 500px; height: 500px;'>"
                    html_str += "<p>"
                    for ans in answer_choices:
                        html_str += f"<p>{ans}</p>"
                    html_str += "<p>"
                    html_str += "<hr>"
            html_str += "</body></html>"
            with open(html_vis_file, "w") as f:
                f.write(html_str)

        


        