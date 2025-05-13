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
    radius = 15  # You can adjust the size of the dot

    # Draw the red dot
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='red')

    # Load a font (optional, comment out if not needed)
    #try:
    font = ImageFont.truetype("LiberationSans-Bold.ttf", 15)  # Adjust font and size as needed
    #except IOError:
    #    font = ImageFont.load_default()

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


if __name__ == "__main__":
    
    asset_id_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json", "r"))
    qa_im_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/multi_qa_images/navigation/'
    qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_split2.json'
    vis = False
    stats = False
    generate = True

    if generate:
        if not os.path.exists(qa_im_path):
            os.makedirs(qa_im_path)

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

        for house_ind, house in enumerate(tqdm.tqdm(dataset["train"])):
            if house_ind < 5000:
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

            if len(reachable_positions) < 10:
                print("Not enough reachable positions, continuing")
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
                    controller = Controller(scene=house_json, width=512, height=512, quality="Ultra", platform=CloudRendering,  renderInstanceSegmentation=True)
                except:
                    print("Cannot render environment, continuing")
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

                # get visible objects
                nav_visible_objects, objid2info, objdesc2cnt, moveable_visible_objs = get_current_state(controller)

                img_view = Image.fromarray(controller.last_event.frame)

                new_path_init = qa_im_path + f"{house_ind}/{sample_count}_0.jpg"
                img_view.save(new_path_init)

                
                actions = []
                simple_actions = []
                simple_wrong_actions = []
                wrong_action_seq = []
                image_seq = [new_path_init,]
                
                step_i = 1
                fail_i = 0
                while step_i < random.choice(range(2, 4)):
                    if fail_i > 5:
                        break
                    
                    # choose a random angle and direction to rotate.
                    angle = random.choice(range(20, 60, 10))
                    distance = random.choice([x/10.0 for x in range(2, 4, 1)])
                    direction = random.choice(["RotateLeft", "RotateRight"])
                    text_direction = "left" if direction=="RotateLeft" else "right"
                    wrong_text_direction = "right" if direction=="RotateLeft" else "left"

                    # pdb.set_trace()
                    try:
                        state = controller.step(action=direction, degrees=angle)
                        state = controller.step(action="MoveAhead", moveMagnitude=distance)
                    except:
                        fail_i += 1
                        continue

                    if state.metadata["lastActionSuccess"]:
                        actions.append(f"rotated {text_direction} by {angle} degrees, moved forward by {distance} meters")
                        simple_actions.append(f"{text_direction}")
                        wrong_action_seq.append(
                            random.choice([f"rotated {wrong_text_direction} by {angle} degrees, moved forward by {distance} meters",
                                           f"rotated {text_direction} by {angle+40} degrees, moved forward by {distance-0.2} meters", 
                                           f"rotated {text_direction} by {angle-20} degrees, moved forward by {distance+0.4} meters"])
                        )
                        simple_wrong_actions.append(f"{wrong_text_direction}")
                        img_view = Image.fromarray(controller.last_event.frame)
                        step_img_path = qa_im_path + f"{house_ind}/{sample_count}_{step_i}.jpg"
                        img_view.save(step_img_path)
                        image_seq.append(step_img_path)
                    else:
                        fail_i += 1
                        continue

                    step_i += 1

                if len(actions) < 1:
                    print("No actions taken, continuing")
                    controller.stop()
                    continue
                
                action = " and then ".join(actions)
                prompt_action = " and then ".join(actions[:2])
                correct_action = actions[-1]
                wrong_action = wrong_action_seq[-1]

                simple_action = " and then ".join(simple_actions)
                simple_wrong_action = " and then ".join(simple_wrong_actions)

                # swap one of the correct actions for a wrong action
                swap_i = random.choice(range(len(actions)))
                hard_wrong_action_seq = copy.deepcopy(actions)
                hard_wrong_action_seq[swap_i] = wrong_action_seq[swap_i]
                hard_wrong_action = " and then ".join(hard_wrong_action_seq)

                # questions about object movement
                #question = "Did any of the objects in the initial frame that you can still see in the subsequent frames move from their original positions?"
                #answer_choices = ["No", "Yes"] # first choice is always correct, randomize later in dataloader
                #qa_pair_choices.append((question, image_seq, answer_choices))
                
                # question about the simple actions taken in the sequence of images
                question = "The first image is from the beginning of the video and the second image is from the end. How did the camera likely move when shooting the video?"
                answer_choices = [simple_action, simple_wrong_action]
                # pdb.set_trace()
                qa_pair_choices.append((question, image_seq, answer_choices))

                # question about the actions taken in the sequence of images
                question = f"The three images are from a camera shooting a video. In the first two images, the camera {prompt_action}. How did the camera likely move from the second image to the third image? Estimate precisely."
                answer_choices = [correct_action, wrong_action]
                qa_pair_choices.append((question, image_seq, answer_choices))

                # pdb.set_trace()
                # check objects we moved closer or further to. 
                obj_distance_changes = []
                move_closer_assets = []
                move_farther_assets = []
                nav_visible_objects_upd, objid2info_upd, objdesc2cnt_upd, moveable_visible_objs_upd = get_current_state(controller)

                image_marked = Image.open(image_seq[0]).convert("RGB")
                for asst_cnt, asset_id in enumerate(nav_visible_objects_upd):
                    asset_size = objid2info[asset_id][8]
                    asset_pos = objid2info[asset_id][10]
                    asset_count = objdesc2cnt[objid2info[asset_id][1]]

                    #check if object too close to the edge:
                    if asset_pos is not None:
                        close_to_edge = asset_pos[0] < 10 or asset_pos[0] > 1000 or asset_pos[1] < 10 or asset_pos[1] > 1000
                    else:
                        close_to_edge = True

                    if asset_id in nav_visible_objects and asset_size>1600 and not close_to_edge:
                        distance = objid2info_upd[asset_id][2]
                        og_distance = objid2info[asset_id][2]  

                        obj_distance_changes.append((asset_id, distance, og_distance, asst_cnt))
                        # if asset_count > 1:
                        image_marked = add_red_dot_with_text(image_marked, (asset_pos[0], asset_pos[1]), str(asst_cnt))

                        if distance < og_distance:
                            move_closer_assets.append((asset_id, asst_cnt))
                        else:
                            move_farther_assets.append((asset_id, asst_cnt))
                
                new_path_marked = qa_im_path + f"{house_ind}/{sample_count}_marked.jpg"
                image_marked.save(new_path_marked)

                
                for asset_id, distance, og_distance, asset_cnt in random.sample(obj_distance_changes, min(3, len(obj_distance_changes))):
                    if distance < og_distance:
                        if random.random() < 0.5:
                            question = f"If I {action}, would {objid2info[asset_id][5]} (near the mark {asset_cnt} in the image) move closer to the camera?"
                            answer_choices = ["yes", "no"]
                            image_order = [new_path_marked,]
                            qa_pair_choices.append((question, image_order, answer_choices))
                        else:
                            question = f"If I {action}, would {objid2info[asset_id][5]} (marked {asset_cnt} in the image) move further from the camera?"    
                            answer_choices = ["no", "yes"]
                            image_order = [new_path_marked,]
                            qa_pair_choices.append((question, image_order, answer_choices)) 

                        if len(move_farther_assets) > 0:
                            question = f"Which of the following objects will move closer to the camera if I {action}?"
                            answer_choices = [objid2info[asset_id][5] + f"(near the mark {asset_cnt} in the image)"] + [objid2info[asst_id][5] + f"(near the mark {asst_cnt} in the image)" for asst_id, asst_cnt in move_farther_assets]
                            image_order = [new_path_marked,]
                            qa_pair_choices.append((question, image_order, answer_choices))
                    elif distance > og_distance:
                        if random.random() < 0.5:
                            question = f"if I {action}, would {objid2info[asset_id][5]} (marked {asset_cnt} in the image) move closer to the camera?"
                            answer_choices = ["no", "yes"]
                            image_order = [new_path_marked,]
                            qa_pair_choices.append((question, image_order, answer_choices))
                        else:
                            question = f"If I {action}, would {objid2info[asset_id][5]} (marked {asset_cnt} in the image) move further from the camera?"
                            answer_choices = ["yes", "no"]
                            image_order = [new_path_marked,]
                            qa_pair_choices.append((question, image_order, answer_choices))

                        if len(move_closer_assets) > 0:
                            question = f"Which of the following objects will move further from the camera if I {action}?"
                            answer_choices = [objid2info[asset_id][5] + f"(marked {asset_cnt} in the image)"] + [objid2info[asst_id][5] + f"(near the mark {asst_cnt} in the image)"  for asst_id, asst_cnt in random.sample(move_closer_assets, 1)]
                            image_order = [new_path_marked,]
                            qa_pair_choices.append((question, image_order, answer_choices))
                    else:
                        question = f"If I {action}, would {objid2info[asset_id][5]} (marked {asset_cnt} in the image) move closer to the camera?"
                        answer_choices = ["no, it would stay the same", "yes"]
                        image_order = [new_path_marked,]
                        qa_pair_choices.append((question, image_order, answer_choices))

                if len(obj_distance_changes) > 3:
                    for random_i in range(3):
                        obj_pair = random.sample(obj_distance_changes, 2)
                        asset_id1, distance1, og_distance1, asset_cnt1 = obj_pair[0]
                        asset_id2, distance2, og_distance2, asset_cnt2 = obj_pair[1]
                        delta = 0.4
                        # check which one moved more closer
                        if og_distance1 - distance1 > (og_distance2 - distance2) + delta:
                            question = f"Would I be moving more in the direction of {objid2info[asset_id1][5]} (near the mark {asset_cnt1} in the image) than {objid2info[asset_id2][5]} (mark {asset_cnt2} in the image) if I {action}?"
                            answer_choices = ["yes", "no"]
                            image_order = [new_path_marked,]
                            qa_pair_choices.append((question, image_order, answer_choices))
                        elif (og_distance1 - distance1) + delta < og_distance2 - distance2:
                            question = f"Would I be moving more in the direction of {objid2info[asset_id1][5]} (near the mark {asset_cnt1} in the image) than {objid2info[asset_id2][5]} (mark {asset_cnt2} in the image) if I {action}?"
                            answer_choices = ["no", "yes"]
                            image_order = [new_path_marked,]
                            qa_pair_choices.append((question, image_order, answer_choices))

                # pdb.set_trace()
                if len(qa_pair_choices) > 0:
                    all_im_qas.append((house_ind, cam_pos, cam_rot, qa_pair_choices))
                sample_count += 1
                controller.stop()
            
            if house_ind % 100 == 0:
                json.dump(all_im_qas, open(qa_json_path, "w"))
            
    if vis:
        all_im_qas = json.load(open(qa_json_path, "r"))
        print("Num samples: ", len(all_im_qas))
        # view in html
        html_str = f"<html><head></head><body>"
        public_im_folder = "/net/cs-nfs/home/grad2/array/public_html/research/r2d3/multi_qa_ims/navigation/"
        for house_ind, _, _, qa_pairs in random.sample(all_im_qas, 50):
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
                    html_str += f"<img src='{html_im_url}' style='width: 300px; height: 300px;'>"
                html_str += "<p>"
                for ans in answer_choices:
                    html_str += f"<p>{ans}</p>"
                html_str += "<p>"
                html_str += "<hr>"
        html_str += "</body></html>"
        with open("/net/cs-nfs/home/grad2/array/public_html/research/r2d3/multi_qa_ims/multi_im_qa_navigation.html", "w") as f:
            f.write(html_str)

        

    if stats:
        all_im_qas = json.load(open(qa_json_path, "r"))
        
        print("Num samples: ", len(all_im_qas))

        avg_qa_pairs = []
        avg_choice_len = []
        avg_answer_len = []
        avg_im_len = []
        avg_action_len = []
        
        for house_ind, _, _, qa_pairs in all_im_qas:
            num_qa = len(qa_pairs)
            avg_qa_pairs.append(num_qa)

            for ques, im_order, answer_choices in qa_pairs:
                avg_choice_len.append(len(answer_choices))
                avg_answer_len.append(len(answer_choices[0].split(" ")))
                avg_im_len.append(len(im_order))
                if len(im_order) > 1:
                    avg_action_len.append(len(im_order))
                
        print("Average number of qa pairs: ", sum(avg_qa_pairs)/len(avg_qa_pairs))
        print("Average number of choices: ", sum(avg_choice_len)/len(avg_choice_len))
        print("Average number of words in answers: ", sum(avg_answer_len)/len(avg_answer_len))
        print("Average number of images per qa pair: ", sum(avg_im_len)/len(avg_im_len))
        print("Average number of action sequences: ", sum(avg_action_len)/len(avg_action_len))
