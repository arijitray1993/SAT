from collections import defaultdict
import json
import prior
from ai2thor.controller import Controller
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
import sys
import shutil
sys.path.append("../../")
from utils.ai2thor_utils import generate_program_from_roomjson, generate_room_programs_from_house_json, make_house_from_cfg

from ai2thor.util.metrics import (
    get_shortest_path_to_object_type
)


def add_red_dot_with_text(image, position, text):
    # Load the image
    draw = ImageDraw.Draw(image)

    # Coordinates and radius of the dot
    x, y = position
    radius = 10  # You can adjust the size of the dot

    # Draw the red dot
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='red')

    # Load a font (optional, comment out if not needed)
    try:
        font = ImageFont.truetype("arial.ttf", 8)  # Adjust font and size as needed
    except IOError:
        font = ImageFont.load_default()

    # Calculate text width and height to center it
    text_width = draw.textlength(text, font=font)
    text_x = x - text_width / 2
    text_y = y

    # Draw the text
    draw.text((text_x, text_y), text, fill='white', font=font)

    return image


def get_current_state(controller):
    nav_visible_objects = controller.step("GetVisibleObjects", maxDistance=5).metadata["actionReturn"]
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
            asset_pos_xy = None

        parent = obj_entry.get('parentReceptacles')
        if parent is not None:
            if len(parent) > 0:
                parent = parent[-1]
                if parent== "Floor":
                    parent = "Floor"
                else:
                    parent = objid2assetid[parent]
        
        is_receptacle = obj_entry['receptacle']
        objid2info[obj_id] = (obj_name, obj_type, distance, pos, rotation, desc, moveable, parent, asset_size_xy, is_receptacle, asset_pos_xy)
        objdesc2cnt[obj_type] += 1

    
    moveable_visible_objs = []
    for objid in nav_visible_objects:
        if objid2info[objid][6] and objid2info[objid][8]>1600:
            moveable_visible_objs.append(objid)

    # pdb.set_trace()
    return nav_visible_objects, objid2info, objdesc2cnt, moveable_visible_objs


def get_action_questions(assetid2info):
    qa_list = []
    
    


ACTION_CHOICES = [
    ("MoveAhead", "move ahead by {param} meters", [0.2, 0.25, 0.3, 0.4, 0.5], []),
    ("RotateRight", "rotate right by {param} degrees", range(20, 60, 10), []),
    ("RotateLeft", "rotate left by {param} degrees", range(20, 60, 10), []),
    ("PickupObject", "pick up the {object}", None, ["Pickupable", "Visible", "NoObjectInHand"]),
    ("DropHandObject", "dropped the {object}", None, ["ObjectInHand"]),
    ("ThrowObject", "threw the {object} forward", None, ["ObjectInHand"]),
    ("DropHandObject", "dropped the {object}", None, ["ObjectInHand"]),
    ("OpenObject", "opened the {object}", None, ["Visible", "Openable"]),
    ("CloseObject", "closed the {object}", None, ["Visible", "Openable"]),
    ("ToggleObjectOn", "turned on the {object}", None, ["Visible", "Toggleable", "Off"]),
    ("ToggleObjectOff", "turned off the {object}", None, ["Visible", "Toggleable", "On"]),
    ("SliceObject", "sliced the {object}", None, ["Visible", "Sliceable"]),
    ("BreakObject", "broke the {object}", None, ["Visible", "Breakable"]),
    ("FillObjectWithLiquid", "filled the {object} with liquid", None, ["Visible", "Fillable"]),
]



if __name__ == "__main__":
    
    asset_id_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json", "r"))
    qa_im_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/multi_qa_images/actions/'
    qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_actions_qas.json'

    if not os.path.exists(qa_im_path):
        os.makedirs(qa_im_path)

    assetid2desc = {}
    for asset in asset_id_desc:
        entries = asset_id_desc[asset]
        captions = []
        for im, obj, desc in entries:
            captions.append(desc)
        assetid2desc[asset] = random.choice(captions)

    dataset = prior.load_dataset("procthor-10k")

    all_im_qas = []
    for house_ind, house in enumerate(dataset["train"]):
        house_json = house
        try:
            controller = Controller(scene=house, width=300, height=300, quality="Low") # quality="Ultra", renderInstanceSegmentation=True, visibilityDistance=30)
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
            print("too few reachable positions, continuing")
            controller.stop()
        
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
                maxDistance=1.5,
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
        for cam_pos, cam_rot, _ in random.sample(random_positions, 5):
            controller = Controller(scene=house_json, width=512, height=512, quality="Ultra")

            try:
                controller.step(action="Teleport", position=cam_pos, rotation=cam_rot)
            except:
                print("Cannot teleport, continuing")
                controller.stop()
                continue

            objid2assetid = {}
            for obj in controller.last_event.metadata['objects']:
                objid2assetid[obj['objectId']] = obj['assetId']

            # get visible objects
            nav_visible_obj_assets, assetid2info, objdesc2cnt, moveable_visible_objs, pickupable_objs = get_current_state(controller)
            img_view = Image.fromarray(controller.last_event.frame)
            new_path_init = qa_im_path + f"{house_ind}/{sample_count}_0.jpg"
            img_view.save(new_path_init)

            image_seq = [new_path_init,]
            action_seq = []
            
            action_questions = get_action_questions(assetid2info, image_seq, action_seq)
            all_im_qas.extend(action_questions)

            step_i = 1
            obj_in_hand = False
            curr_obj = None
            while (step_i < 6):
                if obj_in_hand:
                    if random.random() < 0.5:
                        action, action_desc, action_params = random.choice(OBJECT_ACTIONS)
                    else:
                        action, action_desc, action_params = random.choice(ACTION_CHOICES)
                else:
                    action, action_desc, action_params = random.choice(ACTION_CHOICES)

                if action_params is not None:
                    if action == "MoveAhead":
                        action_param = random.choice(action_params)
                        event = controller.step(action=action, moveMagnitude=action_param)
                        if event.metadata['lastActionSuccess']:
                            action_desc = action_desc.format(param=action_param)
                            action_seq.append(action_desc)
                            img_view = Image.fromarray(controller.last_event.frame)
                            new_path = qa_im_path + f"{house_ind}/{sample_count}_{step_i}.jpg"
                            img_view.save(new_path)
                            image_seq.append(new_path)

                    elif action == "RotateRight" or action == "RotateLeft":
                        action_param = random.choice(action_params)
                        event = controller.step(action=action, degrees=action_param)
                        if event.metadata['lastActionSuccess']:
                            action_desc = action_desc.format(param=action_param)
                            action_seq.append(action_desc)
                            img_view = Image.fromarray(controller.last_event.frame)
                            new_path = qa_im_path + f"{house_ind}/{sample_count}_{step_i}.jpg"
                            img_view.save(new_path)
                            image_seq.append(new_path)

                    else:
                        obj_to_pick = random.choice(pickupable_objs)
                        event = controller.step(action=action, objectId=obj_to_pick)
                        
                        if event.metadata['lastActionSuccess']:
                            curr_obj = obj_to_pick
                            obj_in_hand = True
                            action_desc = action_desc.format(object=curr_obj)
                            action_seq.append(action_desc)
                            img_view = Image.fromarray(controller.last_event.frame)
                            new_path = qa_im_path + f"{house_ind}/{sample_count}_{step_i}.jpg"
                            img_view.save(new_path)
                            image_seq.append(new_path)

                else:
                    event = controller.step(action=action)

                    if event.metadata['lastActionSuccess']:
                        action_desc = action_desc.format(object=curr_obj)
                        obj_in_hand = False
                        curr_obj = None
                        action_seq.append(action_desc)
                        img_view = Image.fromarray(controller.last_event.frame)
                        new_path = qa_im_path + f"{house_ind}/{sample_count}_{step_i}.jpg"
                        img_view.save(new_path)
                        image_seq.append(new_path)

                nav_visible_obj_assets, assetid2info, objdesc2cnt, moveable_visible_objs, pickupable_objs = get_current_state(controller)
                
                action_questions = get_action_questions(assetid2info, image_seq, action_seq)
                all_im_qas.extend(action_questions)

                step_i += 1

            sample_count += 1