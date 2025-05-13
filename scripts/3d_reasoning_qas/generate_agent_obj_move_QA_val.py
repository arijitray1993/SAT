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
    
    split = "val"
    asset_id_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json", "r"))
    qa_im_path = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/multi_qa_images/navigation_{split}_v2/'
    qa_json_path = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_{split}_v2.json'
    vis = True
    stats = False
    generate = True
    load_progress = True

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
        
        if load_progress:
            all_im_qas = json.load(open(qa_json_path, "r"))

        for house_ind, house in enumerate(tqdm.tqdm(dataset[split])):
            
            if house_ind<800:
                continue
            
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
            for cam_pos, cam_rot, _ in random.sample(random_positions[:5], 1):
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
                
                new_rot= cam_rot
                xz_rotation_matrix = np.array([[np.cos(math.radians(new_rot)), -np.sin(math.radians(new_rot))], [np.sin(math.radians(new_rot)), np.cos(math.radians(new_rot))]])
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
                
                if random.random() < 0.7: # decide whether to move or not
                    step_i = 1
                    fail_i = 0
                    while step_i < 2:
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
                            state1 = controller.step(action=direction, degrees=angle)
                        except:
                            fail_i += 1
                            continue

                        if state1.metadata["lastActionSuccess"]:
                            simple_actions.append(f"rotated {text_direction}")
                            simple_wrong_actions.append(f"rotated {wrong_text_direction}")
                            step_i += 1
                        else:
                            fail_i += 1
                            continue
                            
                        if random.random() < 0.5:
                            try:
                                state2 = controller.step(action="MoveAhead", moveMagnitude=distance)
                            except:
                                fail_i += 1
                                continue

                            if state2.metadata["lastActionSuccess"]:
                                simple_actions.append(f"moved forward")
                                simple_wrong_actions.append(random.choice(["moved forward", f"rotated {wrong_text_direction}"]))
                                step_i += 1
                            else:
                                fail_i += 1
                                continue

                        
                    if random.random()<0.3:
                        simple_wrong_actions = ["did not move"]
                else:
                    
                    simple_actions.append("did not move")
                    simple_wrong_actions.append(random.choice(["rotated left", "rotated right", "rotated right and moved forward", "rotated left and moved forward"]))

                if len(simple_actions) < 1:
                    print("Some actions failed, continuing")
                    controller.stop()
                    continue
                
                # decide to move some objects randomly
                if len(moveable_visible_objs) > 0:
                    obj_to_move = random.choice(moveable_visible_objs)
                    obj_name = objid2info[obj_to_move][5]
                else:
                    obj_name = random.choice(["table", "cup", "sofa"])
                obj_action = "no objects moved"
                wrong_obj_action = random.choice([
                    f"{obj_name} was moved left and towards the camera in the first frame",
                    f"{obj_name} was moved right and towards the camera in the first frame",
                    f"{obj_name} was moved left and away from the camera in the first frame",
                    f"{obj_name} was moved right and away from the camera in the first frame",
                ])

                if random.random() < 0.7 and len(moveable_visible_objs) > 0:
                    obj_to_move = random.choice(moveable_visible_objs)

                    #get current coordinates
                    obj_pos = objid2info[obj_to_move][3]

                    # try some random movements and pick the first one that is successful
                    for _ in range(5):
                        new_pos = [obj_pos[0] + random.choice([-0.5, -0.25, 0.25, 0.5]), obj_pos[1], obj_pos[2] + random.choice([-0.5, -0.25, 0.25, 0.5])]
                        try:
                            state3 = controller.step(
                                action="PlaceObjectAtPoint",
                                objectId=obj_to_move,
                                position={"x": new_pos[0], "y": new_pos[1], "z": new_pos[2]},
                            )
                        except:
                            continue
                        # check if object is still visble
                        moved_visible_objects = controller.step("GetVisibleObjects", maxDistance=10).metadata["actionReturn"]
                        if obj_to_move not in moved_visible_objects:
                            continue
                        bboxes = controller.last_event.instance_detections2D
                        if obj_to_move in bboxes:
                            size = (bboxes[obj_to_move][2] - bboxes[obj_to_move][0])*(bboxes[obj_to_move][3] - bboxes[obj_to_move][1])
                        else:
                            continue

                        if size < 900:
                            continue

                        # check if object actually moved
                        for new_obj in controller.last_event.metadata['objects']:
                            if new_obj['objectId'] == obj_to_move:
                                moved_obj_pos = new_obj['position']
                                break
                        if np.linalg.norm(np.array([moved_obj_pos['x'], moved_obj_pos['z']]) - np.array([new_pos[0], new_pos[2]])) > 0.1:
                            continue

                        if state3.metadata["lastActionSuccess"]:
                            # figure out if object moved left or right
                            # apply camera rotation transform to og position and new position
                            og_rotated_pos = np.dot(xz_rotation_matrix, [obj_pos[0], obj_pos[2]])
                            new_rotated_pos = np.dot(xz_rotation_matrix, [new_pos[0], new_pos[2]])
                            if new_rotated_pos[0] > og_rotated_pos[0]:
                                obj_action = "moved right"
                                wrong_obj_action = "moved left"
                            elif new_rotated_pos[0] < og_rotated_pos[0]:
                                obj_action = "moved left"
                                wrong_obj_action = "moved right"
                            
                            #check distance to camera
                            og_distance = objid2info[obj_to_move][2]
                            new_distance = np.linalg.norm(np.array([new_pos[0], new_pos[2]]) - np.array([cam_pos['x'], cam_pos['z']]))
                            if new_distance < og_distance:
                                obj_action += " and towards the camera in the first frame"
                                wrong_obj_action += " and away from the camera in the first frame"
                            elif new_distance > og_distance:
                                obj_action += " and away from the camera in the first frame"
                                wrong_obj_action += " and towards the camera in the first frame"
                            
                            obj_action = f"{objid2info[obj_to_move][5]} was {obj_action}"
                            wrong_obj_action = f"{objid2info[obj_to_move][5]} was {wrong_obj_action}"
                            break
                 
                img_view = Image.fromarray(controller.last_event.frame)
                step_img_path = qa_im_path + f"{house_ind}/{sample_count}_{1}.jpg"
                img_view.save(step_img_path)
                image_seq.append(step_img_path)

                simple_action = " and ".join(simple_actions)
                simple_wrong_action = " and ".join(simple_wrong_actions)
                
                if simple_action == "did not move" and obj_action == "no objects moved":
                    controller.stop()
                    continue
                
                # questions about object movement
                question = "Were any of the objects in the initial frame that you can still see in the second frame moved from their original positions?"
                answer_choices = [obj_action, wrong_obj_action]
                qa_pair_choices.append((question, image_seq, answer_choices))
                
                # question about the simple actions taken in the sequence of images
                question = "The first image is from the beginning of the video and the second image is from the end. How did the camera likely rotate when shooting the video?"
                answer_choices = [simple_action, simple_wrong_action]
                # pdb.set_trace()
                qa_pair_choices.append((question, image_seq, answer_choices))

                
                # pdb.set_trace()
                # check objects we moved closer or further to. 
                obj_distance_changes = []
                move_closer_assets = []
                move_farther_assets = []
                nav_visible_objects_upd, objid2info_upd, objdesc2cnt_upd, moveable_visible_objs_upd = get_current_state(controller)

                image_marked_fin = Image.open(image_seq[0]).convert("RGB")
                for asst_cnt, asset_id in enumerate(nav_visible_objects):
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
                        
                        if abs(distance - og_distance) > 0.2:
                            obj_distance_changes.append((asset_id, distance, og_distance, asst_cnt))
                            # if asset_count > 1:
                            image_marked_fin = add_red_dot_with_text(image_marked_fin, (asset_pos[0], asset_pos[1]), str(asst_cnt))

                            if distance < og_distance:
                                move_closer_assets.append((asset_id, asst_cnt))
                            else:
                                move_farther_assets.append((asset_id, asst_cnt))
                
                
                # ask questions about object movement due to camera movement
                new_path_marked = qa_im_path + f"{house_ind}/{sample_count}_marked.jpg"
                image_marked_fin.save(new_path_marked)

                if obj_action == "no objects moved" and simple_action != "did not move":

                    for asset_id, distance, og_distance, asset_cnt in random.sample(obj_distance_changes, min(3, len(obj_distance_changes))):
                        if distance < og_distance:
                            if random.random()<0.3:
                                if random.random() < 0.5:
                                    question = f"If I {simple_action}, did {objid2info[asset_id][5]} (near the mark {asset_cnt} in the image) move closer to the camera?"
                                    answer_choices = ["yes", "no"]
                                    image_order = [new_path_marked,]
                                    qa_pair_choices.append((question, image_order, answer_choices))
                                else:
                                    question = f"If I {simple_action}, would {objid2info[asset_id][5]} (marked {asset_cnt} in the image) move further from the camera?"    
                                    answer_choices = ["no", "yes"]
                                    image_order = [new_path_marked,]
                                    qa_pair_choices.append((question, image_order, answer_choices)) 

                                if len(move_farther_assets) > 0:
                                    question = f"Which of the following objects will move closer to the camera if I {simple_action}?"
                                    answer_choices = [objid2info[asset_id][5] + f"(near the mark {asset_cnt} in the image)"] + [objid2info[asst_id][5] + f"(near the mark {asst_cnt} in the image)" for asst_id, asst_cnt in move_farther_assets]
                                    image_order = [new_path_marked,]
                                    qa_pair_choices.append((question, image_order, answer_choices))
                        elif distance > og_distance:
                            if random.random() < 0.5:
                                question = f"if I {simple_action}, would {objid2info[asset_id][5]} (marked {asset_cnt} in the image) move closer to the camera?"
                                answer_choices = ["no", "yes"]
                                image_order = [new_path_marked,]
                                qa_pair_choices.append((question, image_order, answer_choices))
                            else:
                                question = f"If I {simple_action}, would {objid2info[asset_id][5]} (marked {asset_cnt} in the image) move further from the camera?"
                                answer_choices = ["yes", "no"]
                                image_order = [new_path_marked,]
                                qa_pair_choices.append((question, image_order, answer_choices))

                            if len(move_closer_assets) > 0:
                                question = f"Which of the following objects will move further from the camera if I {simple_action}?"
                                answer_choices = [objid2info[asset_id][5] + f"(marked {asset_cnt} in the image)"] + [objid2info[asst_id][5] + f"(near the mark {asst_cnt} in the image)"  for asst_id, asst_cnt in random.sample(move_closer_assets, 1)]
                                image_order = [new_path_marked,]
                                qa_pair_choices.append((question, image_order, answer_choices))
                        else:
                            question = f"If I {simple_action}, would {objid2info[asset_id][5]} (marked {asset_cnt} in the image) move closer to the camera?"
                            answer_choices = ["no, it would stay the same", "yes"]
                            image_order = [new_path_marked,]
                            qa_pair_choices.append((question, image_order, answer_choices))

                xz_rotation_matrix = np.array([[np.cos(math.radians(cam_rot)), -np.sin(math.radians(cam_rot))], [np.sin(math.radians(cam_rot)), np.cos(math.radians(cam_rot))]])
                
                
                for asset_id, distance, og_distance, asset_cnt in random.sample(obj_distance_changes, min(3, len(obj_distance_changes))):
                    
                    asset_id_pos = objid2info_upd[asset_id][3]

                    normalized_pos = [asset_id_pos[0] - cam_pos['x'], asset_id_pos[2] - cam_pos['z']]
                    # normalize rot such that camera direction is 0
                    normalized_pos = np.dot(xz_rotation_matrix, normalized_pos)
                    
                    #angle formed by camera and object
                    angle = int(math.degrees(math.atan2(normalized_pos[0], normalized_pos[1])))

                    if angle < -10:
                        angle = abs(int(angle))
                        direction = "left by " + str(angle) + " degrees"
                        wrong_direction = random.choice(["right by " + str(angle) + " degrees", "look straight"])
                    elif angle > 10:
                        direction = "right by " + str(angle) + " degrees"
                        wrong_direction = random.choice(["left by " + str(angle) + " degrees", "look straight"])
                    else:
                        direction = "look straight"
                        wrong_direction = random.choice(["left by 30 degrees", "right by 30 degrees",  "right by 50 degrees",  "left by 40 degrees",  "right by 40 degrees",  "left by 50 degrees"])
                    
                    question = f"I need to go to {objid2info[asset_id][5]} (near the mark {asset_cnt} in the image). Which direction should I turn to face the object?"
                    answer_choices = [direction, wrong_direction]
                    image_order = [new_path_marked]
                    qa_pair_choices.append((question, image_order, answer_choices))

                    question = f"If I turn {wrong_direction}, will I be facing away from {objid2info[asset_id][5]} (near the mark {asset_cnt} in the image)?"
                    answer_choices = ["yes", "no"]
                    image_order = [new_path_marked]
                    qa_pair_choices.append((question, image_order, answer_choices))

                    question = f"If I turn {direction}, will I be facing away from {objid2info[asset_id][5]} (near the mark {asset_cnt} in the image)?"
                    answer_choices = ["no", "yes"]
                    image_order = [new_path_marked]
                    qa_pair_choices.append((question, image_order, answer_choices))


                # pdb.set_trace()
                if len(qa_pair_choices) > 0:
                    all_im_qas.append((house_ind, cam_pos, cam_rot, qa_pair_choices))
                sample_count += 1
                controller.stop()
            
            if len(all_im_qas) % 100 == 0:
                json.dump(all_im_qas, open(qa_json_path, "w"))

        json.dump(all_im_qas, open(qa_json_path, "w"))
            
    if vis:
        all_im_qas = json.load(open(qa_json_path, "r"))
        print("Num samples: ", len(all_im_qas))
        # view in html
        html_str = f"<html><head></head><body>"
        public_im_folder = "/net/cs-nfs/home/grad2/array/public_html/research/r2d3/multi_qa_ims/navigation/"
        for house_ind, _, _, qa_pairs in random.sample(all_im_qas, 15):
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
