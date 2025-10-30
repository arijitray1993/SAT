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
import sys
import shutil
# Assuming 'utils' is in a parent directory
# sys.path.append("../../")
# from utils.ai2thor_utils import generate_program_from_roomjson, generate_room_programs_from_house_json, make_house_from_cfg

from ai2thor.util.metrics import (
    get_shortest_path_to_object_type
)


def add_red_dot_with_text(image, position, text):
    # Load the image
    draw = ImageDraw.Draw(image)

    # Coordinates and radius of the dot
    x, y = position
    radius = 15  # You can adjust the size of the dot

    # Draw the red dot
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='red')

    # Load a font (optional, comment out if not needed)
    try:
        font = ImageFont.truetype("LiberationSans-Bold.ttf", 15)  # Adjust font and size as needed
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
    # Make sure to define objid2assetid in the scope where this function is called
    # This is a dependency from the original script that needs to be available
    global objid2assetid, assetid2desc
    
    nav_visible_objects = controller.step("GetVisibleObjects", maxDistance=5).metadata["actionReturn"]
    
    # Filter objects
    filtered_nav_visible = []
    if nav_visible_objects:
        for obj_id in nav_visible_objects:
            if obj_id in objid2assetid and objid2assetid[obj_id] != "":
                filtered_nav_visible.append(obj_id)
    nav_visible_objects = filtered_nav_visible

    bboxes = controller.last_event.instance_detections2D
    vis_obj_to_size = {}
    if bboxes:
        for obj_id in bboxes:
            vis_obj_to_size[obj_id] = (bboxes[obj_id][2] - bboxes[obj_id][0]) * (bboxes[obj_id][3] - bboxes[obj_id][1])

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
            asset_pos_xy = [(asset_pos_box[0] + asset_pos_box[2]) / 2, (asset_pos_box[1] + asset_pos_box[3]) / 2]
        else:
            asset_pos_xy = None

        parent = obj_entry.get('parentReceptacles')
        if parent is not None:
            if len(parent) > 0:
                parent = parent[-1]
                if parent == "Floor":
                    parent = "Floor"
                elif parent in objid2assetid:
                    parent = objid2assetid[parent]
                else:
                    parent = None # Handle case where parent ID might not be in the map
        
        is_receptacle = obj_entry['receptacle']
        objid2info[obj_id] = (obj_name, obj_type, distance, pos, rotation, desc, moveable, parent, asset_size_xy, is_receptacle, asset_pos_xy)
        objdesc2cnt[obj_type] += 1

    moveable_visible_objs = []
    for objid in nav_visible_objects:
        if objid in objid2info and objid2info[objid][6] and objid2info[objid][8] > 1600:
            moveable_visible_objs.append(objid)

    return nav_visible_objects, objid2info, objdesc2cnt, moveable_visible_objs

# Global dictionaries needed by get_current_state
assetid2desc = {}
objid2assetid = {}

if __name__ == "__main__":

    split = "train"
    # Define paths - Update these to your local paths
    asset_info_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json"
    qa_im_path = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/multi_qa_images/cameramove_4frame_{split}/'
    qa_json_path = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_cameramove_4frame_qas_{split}.json'
    
    # Script control flags
    vis = True
    stats = False
    generate = True # Set to True to run generation
    load_progress = False # Set to True to load existing JSON

    # Define all possible motion actions, their commands, and corresponding QA text
    motion_actions = [
        {
            "name": "Still",
            "thor_commands": [{"action": "Pass"}, {"action": "Pass"}, {"action": "Pass"}],
            "desc": "The camera is completely still.",
            "wrong_desc": ["The camera moves forward.", "The camera pans left.", "The camera tilts up.", "The camera moves rightward."],
            "yes_no_positive": ["Is the camera completely still without any motion or shaking?", "Is the scene in the video completely static?"],
            "yes_no_negative_base": [
                "Does the camera move forward (not zooming in) with respect to the initial frame?",
                "Does the camera pan to the right?",
                "Does the camera pan to the left?",
                "Does the camera move rightward in the scene?",
                "Does the camera tilt upward?",
                "Does the camera move backward (not zooming out) with respect to the initial frame?",
                "Does the camera zoom out?",
                "Does the camera tilt downward?",
                "Does the camera move leftward in the scene?",
                "Does the camera zoom in?",
            ]
        },
        {
            "name": "Move Forward",
            "thor_commands": [{"action": "MoveAhead", "moveMagnitude": 0.25}, {"action": "MoveAhead", "moveMagnitude": 0.25}, {"action": "MoveAhead", "moveMagnitude": 0.25}],
            "desc": "The camera moves forward.",
            "wrong_desc": ["The camera moves backward.", "The camera pans left.", "The camera is still."],
            "yes_no_positive": ["Does the camera move forward (not zooming in) with respect to the initial frame?", "Is the camera moving forward in the scene?", "Does the camera zoom in?"],
            "yes_no_negative_base": ["Is the camera completely still?", "Does the camera move backward?", "Does the camera pan to the left?", "Is the camera free from any forward motion?"]
        },
        {
            "name": "Move Backward",
            "thor_commands": [{"action": "MoveBack", "moveMagnitude": 0.25}, {"action": "MoveBack", "moveMagnitude": 0.25}, {"action": "MoveBack", "moveMagnitude": 0.25}],
            "desc": "The camera moves backward.",
            "wrong_desc": ["The camera moves forward.", "The camera pans right.", "The camera is still."],
            "yes_no_positive": ["Does the camera move backward (not zooming out) with respect to the initial frame?", "Is the camera moving backward in the scene?", "Does the camera zoom out?"],
            "yes_no_negative_base": ["Is the camera completely still?", "Does the camera move forward?", "Is the camera free from any backward motion?"]
        },
        {
            "name": "Move Left",
            "thor_commands": [{"action": "MoveLeft", "moveMagnitude": 0.25}, {"action": "MoveLeft", "moveMagnitude": 0.25}, {"action": "MoveLeft", "moveMagnitude": 0.25}],
            "desc": "The camera moves leftward.",
            "wrong_desc": ["The camera moves rightward.", "The camera moves forward.", "The camera is still."],
            "yes_no_positive": ["Does the camera move leftward in the scene?", "Does the camera move laterally to the left?"],
            "yes_no_negative_base": ["Is the camera completely still?", "Does the camera move rightward?", "Is the camera free from any leftward lateral movement?"]
        },
        {
            "name": "Move Right",
            "thor_commands": [{"action": "MoveRight", "moveMagnitude": 0.25}, {"action": "MoveRight", "moveMagnitude": 0.25}, {"action": "MoveRight", "moveMagnitude": 0.25}],
            "desc": "The camera moves rightward.",
            "wrong_desc": ["The camera moves leftward.", "The camera moves backward.", "The camera is still."],
            "yes_no_positive": ["Does the camera move rightward in the scene?", "Does the camera move laterally to the right?"],
            "yes_no_negative_base": ["Is the camera completely still?", "Does the camera move leftward?", "Is the camera free from any rightward lateral movement?"]
        },
        {
            "name": "Pan Left",
            "thor_commands": [{"action": "RotateLeft", "degrees": 20}, {"action": "RotateLeft", "degrees": 20}, {"action": "RotateLeft", "degrees": 20}],
            "desc": "The camera pans to the left.",
            "wrong_desc": ["The camera pans to the right.", "The camera moves forward.", "The camera is still."],
            "yes_no_positive": ["Does the camera pan to the left?"],
            "yes_no_negative_base": ["Is the camera completely still?", "Does the camera pan to the right?", "Is the camera free from any leftward panning motion?"]
        },
        {
            "name": "Pan Right",
            "thor_commands": [{"action": "RotateRight", "degrees": 20}, {"action": "RotateRight", "degrees": 20}, {"action": "RotateRight", "degrees": 20}],
            "desc": "The camera pans to the right.",
            "wrong_desc": ["The camera pans to the left.", "The camera moves forward.", "The camera is still."],
            "yes_no_positive": ["Does the camera pan to the right?"],
            "yes_no_negative_base": ["Is the camera completely still?", "Does the camera pan to the left?"]
        },
        {
            "name": "Tilt Up",
            "thor_commands": [{"action": "LookUp", "degrees": 15}, {"action": "LookUp", "degrees": 15}, {"action": "LookUp", "degrees": 15}],
            "desc": "The camera tilts upward.",
            "wrong_desc": ["The camera tilts downward.", "The camera moves forward.", "The camera is still."],
            "yes_no_positive": ["Does the camera tilt upward?"],
            "yes_no_negative_base": ["Is the camera completely still?", "Does the camera tilt downward?"]
        },
        {
            "name": "Tilt Down",
            "thor_commands": [{"action": "LookDown", "degrees": 15}, {"action": "LookDown", "degrees": 15}, {"action": "LookDown", "degrees": 15}],
            "desc": "The camera tilts downward.",
            "wrong_desc": ["The camera tilts upward.", "The camera moves backward.", "The camera is still."],
            "yes_no_positive": ["Does the camera tilt downward?"],
            "yes_no_negative_base": ["Is the camera completely still?", "Does the camera tilt upward?"]
        }
    ]


    if generate:
        if not os.path.exists(qa_im_path):
            os.makedirs(qa_im_path)

        # Load asset descriptions
        try:
            asset_id_desc_json = json.load(open(asset_info_path, "r"))
            for asset in asset_id_desc_json:
                entries = asset_id_desc_json[asset]
                captions = []
                for im, obj, desc in entries:
                    desc = desc.strip().lower().replace(".", "")
                    captions.append(desc)
                if captions:
                    assetid2desc[asset] = random.choice(captions)
        except FileNotFoundError:
            print(f"Warning: Asset info file not found at {asset_info_path}. Descriptions will be limited.")
            assetid2desc = {}
        except json.JSONDecodeError:
            print(f"Warning: Could not decode asset info file at {asset_info_path}.")
            assetid2desc = {}


        dataset = prior.load_dataset("procthor-10k")
        all_im_qas = []
        
        if load_progress:
            try:
                all_im_qas = json.load(open(qa_json_path, "r"))
                print(f"Loaded {len(all_im_qas)} existing QA pairs.")
            except FileNotFoundError:
                print("No existing QA file found, starting from scratch.")
            except json.JSONDecodeError:
                 print(f"Error reading {qa_json_path}, starting from scratch.")

        for house_ind, house in enumerate(tqdm.tqdm(dataset[split])):
            
            house_json = house

            try:
                controller = Controller(scene=house, width=200, height=200, quality="Low", platform=CloudRendering)
            except Exception as e:
                print(f"Cannot render environment {house_ind}, continuing. Error: {e}")
                continue
            
            try:
                event = controller.step(action="GetReachablePositions")
                reachable_positions = event.metadata["actionReturn"]
            except Exception as e:
                print(f"Cannot get reachable positions for {house_ind}, continuing. Error: {e}")
                controller.stop()
                continue

            if not reachable_positions or len(reachable_positions) < 10:
                print(f"Not enough reachable positions in {house_ind}, continuing")
                controller.stop()
                continue
            
            random_positions = []
            # Sample more positions to find a good one
            num_samples = min(len(reachable_positions), 10)
            for cam_pos in random.sample(reachable_positions, num_samples):
                
                cam_rot = random.choice(range(360))
                try:
                    controller.step(action="Teleport", position=cam_pos, rotation=cam_rot, horizon=0) # Reset horizon
                except Exception as e:
                    print(f"Cannot teleport in {house_ind}, continuing. Error: {e}")
                    continue

                nav_visible_objects = controller.step(
                    "GetVisibleObjects",
                    maxDistance=5,
                ).metadata["actionReturn"]
                
                num_visible = len(nav_visible_objects) if nav_visible_objects else 0
                random_positions.append((cam_pos, cam_rot, num_visible))
            
            if len(random_positions) == 0:
                print(f"No objects visible in any sampled position for {house_ind}, continuing")
                controller.stop()
                continue

            # Sort by number of visible objects and take the best ones
            random_positions = sorted(random_positions, key=lambda x: x[2], reverse=True)
            
            controller.stop() # Stop the low-res controller

            house_img_dir = qa_im_path + f"{house_ind}"
            if not os.path.exists(house_img_dir):
                os.makedirs(house_img_dir)

            sample_count = 0
            # Try to generate from the top 3 best starting positions
            for cam_pos, cam_rot, _ in random_positions[:3]:
                if sample_count >= 1: # Only generate one sample per house for now
                    break

                qa_pair_choices = []
                try:
                    controller = Controller(scene=house_json, width=512, height=512, quality="Ultra", platform=CloudRendering, renderInstanceSegmentation=True)
                except Exception as e:
                    print(f"Cannot render high-res environment {house_ind}, continuing. Error: {e}")
                    continue
                
                try:
                    # Teleport to the starting position, reset horizon
                    controller.step(action="Teleport", position=cam_pos, rotation=cam_rot, horizon=0)
                except Exception as e:
                    print(f"Cannot teleport high-res controller in {house_ind}, continuing. Error: {e}")
                    controller.stop()
                    continue
                
                # Update objid2assetid map for this scene
                objid2assetid = {}
                for obj in controller.last_event.metadata['objects']:
                    objid2assetid[obj['objectId']] = obj['assetId']

                # --- NEW ACTION AND QA GENERATION LOGIC ---

                # 1. Randomly pick a motion action
                action_data = random.choice(motion_actions)
                action_commands = action_data["thor_commands"]
                
                image_seq = []
                all_actions_succeeded = True

                # 2. Save initial frame (Frame 0)
                img_view = Image.fromarray(controller.last_event.frame)
                frame_path = os.path.join(house_img_dir, f"{sample_count}_0.jpg")
                img_view.save(frame_path)
                image_seq.append(frame_path)

                # 3. Execute 3 steps to get 3 more frames (Frames 1, 2, 3)
                for i, command in enumerate(action_commands):
                    try:
                        event = controller.step(**command)
                        if not event.metadata["lastActionSuccess"]:
                            print(f"Action {command['action']} failed in {house_ind}, stopping motion sequence.")
                            all_actions_succeeded = False
                            break
                        
                        # Save frame
                        img_view = Image.fromarray(controller.last_event.frame)
                        frame_path = os.path.join(house_img_dir, f"{sample_count}_{i+1}.jpg")
                        img_view.save(frame_path)
                        image_seq.append(frame_path)

                    except Exception as e:
                        print(f"Error during action {command['action']} in {house_ind}: {e}")
                        all_actions_succeeded = False
                        break
                
                # If motion failed or didn't produce 4 frames, skip this sample
                if not all_actions_succeeded or len(image_seq) != 4:
                    print(f"Skipping sample for {house_ind} due to failed/incomplete action.")
                    controller.stop()
                    # Clean up partial images
                    for img_path in image_seq:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    continue

                # 4. Generate QA pairs
                
                # QA Type 1: "What happened?" (Descriptive Answer)
                question1 = "The first image is from the beginning of the video and the last image is from the end. How did the camera move?"
                correct_answer1 = action_data["desc"]
                wrong_answer1 = random.choice([d for d in action_data["wrong_desc"] if d != correct_answer1])
                
                answer_choices1 = [correct_answer1, wrong_answer1]
                random.shuffle(answer_choices1)
                correct_index1 = answer_choices1.index(correct_answer1)
                qa_pair_choices.append((question1, image_seq, answer_choices1, correct_index1))

                # QA Type 2: "Did [Correct Action] happen?" (Yes/No - Positive)
                if action_data["yes_no_positive"]:
                    question2 = random.choice(action_data["yes_no_positive"])
                    answer_choices2 = ["Yes", "No"]
                    correct_index2 = 0 # "Yes"
                    qa_pair_choices.append((question2, image_seq, answer_choices2, correct_index2))

                # QA Type 3: "Did [Wrong Action] happen?" (Yes/No - Negative)
                question3 = ""
                # 50% chance to pick from the base negative list for this action
                if random.random() < 0.5 and action_data["yes_no_negative_base"]:
                    question3 = random.choice(action_data["yes_no_negative_base"])
                # 50% chance to pick a positive question from a *different* action
                else:
                    wrong_action_data = random.choice([a for a in motion_actions if a["name"] != action_data["name"]])
                    if wrong_action_data["yes_no_positive"]:
                        question3 = random.choice(wrong_action_data["yes_no_positive"])
                
                if question3: # Ensure we got a valid question
                    answer_choices3 = ["Yes", "No"]
                    correct_index3 = 1 # "No"
                    # Shuffle choices for this one to be tricky
                    random.shuffle(answer_choices3)
                    correct_index3 = answer_choices3.index("No")
                    qa_pair_choices.append((question3, image_seq, answer_choices3, correct_index3))

                # --- END NEW LOGIC ---

                if len(qa_pair_choices) > 0:
                    # Save the (house_id, start_pos, start_rot, list_of_qa_tuples)
                    # Each qa_tuple is (question, image_seq_paths, [choice1, choice2], correct_index)
                    all_im_qas.append((house_ind, cam_pos, cam_rot, qa_pair_choices))
                
                sample_count += 1
                controller.stop()
            
            # Save progress periodically
            if house_ind % 100 == 0 and len(all_im_qas) > 0:
                print(f"\nSaving progress: {len(all_im_qas)} total QA sets generated.")
                try:
                    json.dump(all_im_qas, open(qa_json_path, "w"), indent=2)
                except Exception as e:
                    print(f"Error saving progress to JSON: {e}")


    # Final save
    if generate and len(all_im_qas) > 0:
        print(f"\nFinished generation. Total QA sets: {len(all_im_qas)}")
        try:
            json.dump(all_im_qas, open(qa_json_path, "w"), indent=2)
            print(f"Successfully saved to {qa_json_path}")
        except Exception as e:
            print(f"Error on final save to JSON: {e}")
    elif generate:
        print("Generation finished, but no QA pairs were created.")

