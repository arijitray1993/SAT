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
    for objid in nav_all_visible_objects:
        if objid in objid2info:
            if objid2info[objid][8]>2500:
                visible_objs.append(objid)

    # pdb.set_trace()
    return nav_visible_objects, objid2info, objdesc2cnt, visible_objs


if __name__ == "__main__":
    
    split="train"
    
    qa_im_path = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3DCap_{split}/'
    qa_json_path = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_captions_{split}.json'

    html_vis_file = '/net/cs-nfs/home/grad2/array/public_html/research/r2d3/multi_qa_ims/3d_cap.html'
    generate=True
    vis=False

    if not os.path.exists(qa_im_path):
        os.makedirs(qa_im_path)
    

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
        all_im_caps = []
        #all_im_caps_marked = []
        # all_im_caps = json.load(open(qa_json_path, "r"))

        for house_ind, house in enumerate(tqdm.tqdm(dataset[split])):
            im_qa_pairs = []
            if house_ind >5001:
                break
            house_json = house
            try:
                controller = Controller(scene=house, width=200, height=200, quality="Low", platform=CloudRendering) # quality="Ultra", renderInstanceSegmentation=True, visibilityDistance=30)
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

            if len(reachable_positions)<20:
                print("No reachable positions, continuing")
                controller.stop()
                continue

            random_positions = []
            for cam_pos in random.sample(reachable_positions, 20):
                
                cam_rot = random.choice(range(360))
                try:
                    controller.step(action="Teleport", position=cam_pos, rotation=cam_rot)
                except:
                    print("Cannot teleport, continuing")
                    continue

                nav_visible_objects = controller.step(
                    "GetVisibleObjects",
                    maxDistance=20,
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
            for cam_pos, cam_rot, _ in random_positions[:3]:
                
                random_field_of_view = random.choice([30, 45, 50, 54, 56, 57, 60, 65, 70, 75, 80, 90, 95, 100, 110, 120])
                try:
                    controller = Controller(scene=house_json, width=512, height=512, quality="Ultra", renderInstanceSegmentation=True, platform=CloudRendering, fieldOfView=random_field_of_view)
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
                    
                cam_pos = [cam_pos['x'], cam_pos['y'], cam_pos['z']]
                objid2assetid = {}
                for obj in controller.last_event.metadata['objects']:
                    objid2assetid[obj['objectId']] = obj['assetId']

                img_view = Image.fromarray(controller.last_event.frame)

                new_path_init = qa_im_path + f"{house_ind}/{sample_count}_0.jpg"
                img_view.save(new_path_init)
                new_path_init_marked = qa_im_path + f"{house_ind}/{sample_count}_0_marked.jpg"

                # get random float point coordinates 0-1
                random_xy_point = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
                query = controller.step(
                    action="GetCoordinateFromRaycast",
                    x=random_xy_point[0],
                    y=random_xy_point[1],
                )
                location = query.metadata["actionReturn"]
                marked_image = add_red_dot_with_text(img_view, (int(random_xy_point[0]*512), int(random_xy_point[1]*512)), f"A")

                location = [location['x'], location['y'], location['z']]
                norm_location = (location[0] - cam_pos[0], location[1] - cam_pos[1], location[2] - cam_pos[2])
                norm_location_rot = (norm_location[0]*np.cos(np.deg2rad(cam_rot)) - norm_location[2]*np.sin(np.deg2rad(cam_rot)), norm_location[0]*np.sin(np.deg2rad(cam_rot)) + norm_location[2]*np.cos(np.deg2rad(cam_rot)))
                coordinate_text = f"The red point marked A has a 3D location of ({int(norm_location_rot[0]*100)}, {int(norm_location[1]*100)}, {int(norm_location_rot[1]*100)})"
                marked_image.save(new_path_init_marked)

                # get visible objects
                nav_visible_obj_assets, objid2info, objdesc2cnt, visible_obj_assets = get_current_state(controller)

                # make a text list of all object descriptions and their 3D coordinates that are visible in the scene
                visible_obj_descs = ""
                obj_count=0
                obj_ind_to_color = {
                    0: "red",
                    1: "green",
                    2: "blue",
                    3: "yellow",
                    4: "purple",
                    5: "orange",
                    6: "pink",
                    7: "cyan",
                }
                for obj_ind, obj in enumerate(visible_obj_assets):
                    
                    obj_desc = objid2info[obj][5]
                    obj_pos = objid2info[obj][3]
                    obj_count = objdesc2cnt[objid2info[obj][1]]

                    if obj_desc in ["Floor", "Wall", "Ceiling"]:
                        continue

                    if obj_count>1:
                        # mark the object
                        obj_pos_box = objid2info[obj][11]
                        marked_image = add_box_dot_with_color(marked_image, obj_pos_box, obj_ind_to_color[obj_ind%7])
                        marked_image.save(new_path_init_marked)
                        obj_desc = f"{obj_desc} (marked by {obj_ind_to_color[obj_ind%7]} box)"

                    # normalize the obj position to camera coordinates. 
                    # camera is at (0, 0, 0) and it is looking at the z-axis
                    obj_pos_norm = (obj_pos[0] - cam_pos[0], obj_pos[1] - cam_pos[1], obj_pos[2] - cam_pos[2])
                    obj_pos_norm_rot = (obj_pos_norm[0]*np.cos(np.deg2rad(cam_rot)) - obj_pos_norm[2]*np.sin(np.deg2rad(cam_rot)), obj_pos_norm[0]*np.sin(np.deg2rad(cam_rot)) + obj_pos_norm[2]*np.cos(np.deg2rad(cam_rot)))

                    desc_text = f"{obj_desc} at {int(obj_pos_norm_rot[0]*100)}, {int(obj_pos_norm[1]*100)}, {int(obj_pos_norm_rot[1]*100)}"
                    
                    visible_obj_descs += desc_text + "\n"

                    question = f"What is the rough 3D location of {obj_desc}?"
                    answer = f"({int(obj_pos_norm_rot[0]*100)}, {int(obj_pos_norm[1]*100)}, {int(obj_pos_norm_rot[1]*100)})"
                    im_qa_pairs.append([question, answer])

                    obj_count+=1

                if obj_count<=1:
                    controller.stop()
                    continue
                
                dense_cap_question = "Describe the objects and their rough 3D location in the image."
                dense_cap_answer = visible_obj_descs
                im_qa_pairs.append([dense_cap_question, dense_cap_answer])

                all_im_caps.append((new_path_init, im_qa_pairs, coordinate_text, random_field_of_view))

                # pdb.set_trace()
                sample_count += 1

                controller.stop()
            
            if house_ind % 100 == 0:
                json.dump(all_im_caps, open(qa_json_path, "w"))
    
                