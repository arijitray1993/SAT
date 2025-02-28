from collections import defaultdict
import json
import os
from utils.ai2thor_utils import make_house_from_cfg
import pdb 
from PIL import Image
import numpy as np
import yaml
from shapely.geometry import Polygon
import shapely
from scipy.optimize import linear_sum_assignment
import re
import ast
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import average_precision_score

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import wandb

class GenHouseIms:

    def __init__(self,):
        self.im_sim_scores = []

    def update(self, output):

        # generate house from output
        house = make_house_from_cfg(output)

        # get image from house

def compute_location_error(pred_locations, gt_locations, max_dimension):
    
    # first we create a cost matrix that calculates distance between pred dist and gt dist
    k_gt = len(gt_locations)
    k_pred = len(pred_locations)
    cost_matrix = np.zeros((k_gt, k_pred))
    for i in range(k_gt):
        for j in range(k_pred):
            cost_matrix[i, j] = np.linalg.norm(pred_locations[j] - gt_locations[i])
    #print(cost_matrix)
    # then we use linear_sum_assignment to find the best matching between gt and pred
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    #compute the mean of the least mapped distances
    mean_dist = 0
    accuracy = []
    for i in range(len(row_ind)):
        if cost_matrix[row_ind[i], col_ind[i]]/max_dimension < 0.1:
            accuracy.append(1)
        else:
            accuracy.append(0)

        mean_dist += cost_matrix[row_ind[i], col_ind[i]]

    mean_dist = mean_dist / len(row_ind)

    #print(cluster_to_label_map)

    return mean_dist, np.mean(accuracy)

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def compute_locationposeattr_error(pred_locations, gt_locations, max_dimension, attrmodel, attrtokenizer):

    # first we create a cost matrix that calculates distance between pred dist and gt dist
    k_gt = len(gt_locations)
    k_pred = len(pred_locations)
    cost_matrix = np.zeros((k_gt, k_pred))
    pose_matrix = np.zeros((k_gt, k_pred))
    for i in range(k_gt):
        for j in range(k_pred):
            try:
                cost_matrix[i, j] = np.linalg.norm(np.array(pred_locations[j][0]) - np.array(gt_locations[i][0]))
            except:
                pdb.set_trace()
            try:
                pose_matrix[i, j] = (abs(pred_locations[j][1][1] - gt_locations[i][1][1])) # just rotation around y axis
            except:
                pose_matrix[i, j] = (abs(pred_locations[j][1][1] - gt_locations[i][1])) # just rotation around y axis
    #print(cost_matrix)
    # then we use linear_sum_assignment to find the best matching between gt and pred
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    #compute the mean of the least mapped distances
    mean_dist = 0
    accuracy = []
    pose_accuracy = []
    attr_sim = []
    mean_pose_dist = 0
    for i in range(len(row_ind)):
        if cost_matrix[row_ind[i], col_ind[i]]/max_dimension < 0.1:
            accuracy.append(1)
            if pose_matrix[row_ind[i], col_ind[i]] < 10:
                pose_accuracy.append(1)
            else:
                pose_accuracy.append(0)
        else:
            pose_accuracy.append(0)
            accuracy.append(0)

        mean_dist += cost_matrix[row_ind[i], col_ind[i]]
        mean_pose_dist += pose_matrix[row_ind[i], col_ind[i]]

        # compute attribute distance
        gt_attr = gt_locations[row_ind[i]][-1]
        pred_attr = pred_locations[col_ind[i]][-1]

        # compute attribute similarity
        batch_dict = attrtokenizer([gt_attr, pred_attr], max_length=512, padding=True, truncation=True, return_tensors='pt')

        batch_dict = {k: v.cuda() for k, v in batch_dict.items()}

        outputs = attrmodel(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        attr_sim.append(scores.item())
        # pdb.set_trace()


    mean_dist = mean_dist / len(row_ind)

    #print(cluster_to_label_map)

    return mean_dist, np.mean(accuracy), np.mean(pose_accuracy), mean_pose_dist, np.mean(attr_sim)


def compute_locationpose_error(pred_locations, gt_locations, max_dimension):
    
    # first we create a cost matrix that calculates distance between pred dist and gt dist
    k_gt = len(gt_locations)
    k_pred = len(pred_locations)
    cost_matrix = np.zeros((k_gt, k_pred))
    pose_matrix = np.zeros((k_gt, k_pred))
    for i in range(k_gt):
        for j in range(k_pred):
            cost_matrix[i, j] = np.linalg.norm(pred_locations[j][0] - gt_locations[i][0])
            pose_matrix[i, j] = (abs(pred_locations[j][1][1] - gt_locations[i][1][1])) # just rotation around y axis
    #print(cost_matrix)
    # then we use linear_sum_assignment to find the best matching between gt and pred
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    #compute the mean of the least mapped distances
    mean_dist = 0
    accuracy = []
    pose_accuracy = []
    mean_pose_dist = 0
    for i in range(len(row_ind)):
        if cost_matrix[row_ind[i], col_ind[i]]/max_dimension < 0.1:
            accuracy.append(1)
            if pose_matrix[row_ind[i], col_ind[i]] < 10:
                pose_accuracy.append(1)
            else:
                pose_accuracy.append(0)
        else:
            pose_accuracy.append(0)
            accuracy.append(0)

        mean_dist += cost_matrix[row_ind[i], col_ind[i]]
        mean_pose_dist += pose_matrix[row_ind[i], col_ind[i]]

    mean_dist = mean_dist / len(row_ind)

    #print(cluster_to_label_map)

    return mean_dist, np.mean(accuracy), np.mean(pose_accuracy), mean_pose_dist


def get_obj_class_from_desc_pred(obj_desc, known_desc_to_class, attrmodel, attrtokenizer):
    obj_desc = obj_desc.lower()
    known_descs = list(known_desc_to_class.keys())
    all_descs = [obj_desc,] + known_descs

    # compute the highest similarity to known descriptions
    batch_dict = attrtokenizer(all_descs, max_length=512, padding=True, truncation=True, return_tensors='pt')

    batch_dict = {k: v.cuda() for k, v in batch_dict.items()}

    outputs = attrmodel(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:1] @ embeddings[1:].T)
    
    max_score_ind = scores.argmax().item()

    best_known_desc= known_descs[max_score_ind]

    return known_desc_to_class[best_known_desc]

def get_obj_prec_from_desc_pred(obj_desc, known_desc_to_class, attrmodel, attrtokenizer, gt_objs):
    obj_desc = obj_desc.lower()
    known_descs = list(known_desc_to_class.keys())
    all_descs = [obj_desc,] + known_descs

    # compute the highest similarity to known descriptions
    batch_dict = attrtokenizer(all_descs, max_length=512, padding=True, truncation=True, return_tensors='pt')

    batch_dict = {k: v.cuda() for k, v in batch_dict.items()}

    outputs = attrmodel(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:1] @ embeddings[1:].T)
    
    pdb.set_trace()
    # compute precision
    known_objs = [known_desc_to_class[desc][1] for entries in desc for desc in known_descs]
    labels = [1 if obj in gt_objs else 0 for obj in known_objs]

    precision = average_precision_score(labels, scores.cpu().numpy())

    return precision



def get_object_class_from_asset(obj):
    pattern = r'[0-9]'
    obj_name = obj.replace("_", " ")
    obj_name = re.sub(pattern, '', obj_name)

    obj_name = obj_name.replace("RoboTHOR", "")

    obj_name = obj_name.replace("jokkmokk", " ")

    if "Countertop" in obj_name:
        shape = obj_name.split(" ")[-1]
        if shape =="I":
            obj_name = "I-shaped countertop"
        elif shape =="L":
            obj_name = "L-shaped countertop"
        elif shape =="C":
            obj_name = "C-shaped countertop"

    obj_name = obj_name.strip()

    return obj_name

class HouseObjectDistancesAccuracy:
    def __init__(self, args):
        self.house_jsons = []
        self.gt_house_jsons = []
        self.selected_objs = []
        self.object_location_error = []
        self.object_location_accuracy = []
        self.logger = args['logger']
    
    def update(self, output, gt):
        if "#room" in output:
            room_response = output.split(": #room \n")[-1]
        else:
            room_response = output.split(": \n")[-1]

        room_json_text = "\n".join(room_response.split("\n")[:-1])
        room_json_text = room_json_text.split("###")[0]
        room_json_text = room_json_text.replace("(", "[")
        room_json_text = room_json_text.replace(")", "]")
        
        try:
            pred_house_dict = yaml.load(room_json_text, Loader=yaml.FullLoader)
        except:
            pred_house_dict = {}
        if pred_house_dict is None:
            print("none house dict")
            print(room_json_text)
        
        self.house_jsons.append(pred_house_dict)
        
        # convert gt program text to gen config
        gt_house_text = gt['text_labels'][0]
        if "#room" in gt_house_text:
            gt_house_text = gt_house_text.split(": #room \n")[-1]
        else:
            gt_house_text = gt_house_text.split(": \n")[-1]
        gt_house_text = gt_house_text.replace("(", "[")
        gt_house_text = gt_house_text.replace(")", "]")
        # pdb.set_trace()

        try:
            gt_house_dict = yaml.load(gt_house_text, Loader=yaml.FullLoader)
        except:
            gt_house_dict = {}

        if gt_house_dict is None:
            print("none house dict GT!")
            print(gt_house_text)
        self.gt_house_jsons.append(gt_house_dict)
        # pdb.set_trace()
        # self.selected_objs.append(gt['objs_present'][0])

    def compute(self):
        for pred_house, gt_house in zip(self.house_jsons, self.gt_house_jsons):   

            house_obj_loc_error = []
            house_obj_loc_acc = []
            if pred_house is None or gt_house is None:
                continue
            if len(pred_house) == 0 or len(gt_house) == 0:
                continue

            pred_polygon = pred_house['polygon']
            gt_polygon = gt_house['polygon']

            max_coordinate = np.max(gt_polygon)
            min_coordinate = np.min(gt_polygon)
            max_dimension = max_coordinate - min_coordinate

            gt_objs = defaultdict(list)
            i=0
            while(True):
                if f'obj_{i}' not in gt_house:
                    break
                gt_object, location, rotation = gt_house[f'obj_{i}']
                gt_objs[gt_object].append(location)
                i += 1
            
            pred_objs = defaultdict(list)
            i=0
            while(True):
                if f'obj_{i}' not in pred_house:
                    break
                try:
                    pred_object, location, rotation = pred_house[f'obj_{i}']
                except:
                    print("error object")
                    i+=1
                    continue
                pred_objs[pred_object].append(location)
                i+=1
            
            pred_obj_classes = defaultdict(list)
            
            for obj in pred_objs:
                obj_name = get_object_class_from_asset(obj)
                pred_obj_classes[obj_name].extend(pred_objs[obj])
            
            pred_locations = {}
            gt_locations = defaultdict(list)
            for obj in gt_objs:
                obj_name = get_object_class_from_asset(obj)
                if obj_name in pred_obj_classes:
                    pred_locations[obj_name] = pred_obj_classes[obj_name]
                    gt_locations[obj_name].extend(gt_objs[obj])
                
            
            # compute distance of each object to other objects
            obj_pair_dist = defaultdict(dict)
            for obj_1 in gt_locations:
                for obj_2 in gt_locations:
                    if obj_1 == obj_2:
                        continue
                    loc_1 = gt_locations[obj_1]
                    loc_2 = gt_locations[obj_2]
                    
                    all_dist = []
                    for l1 in loc_1:
                        for l2 in loc_2:
                            dist = np.linalg.norm(np.array(l1) - np.array(l2))
                            all_dist.append(dist)

                    obj_pair_dist[obj_1][obj_2] = all_dist
            
            pred_obj_pair_dist = defaultdict(dict)
            for obj_1 in pred_locations:
                for obj_2 in pred_locations:
                    if obj_1 == obj_2:
                        continue
                    loc_1 = pred_locations[obj_1]
                    loc_2 = pred_locations[obj_2]
                    
                    all_dist = []
                    for l1 in loc_1:
                        for l2 in loc_2:
                            dist = np.linalg.norm(np.array(l1) - np.array(l2))
                            all_dist.append(dist)
                        
                    pred_obj_pair_dist[obj_1][obj_2] = all_dist
            
            # compute distance error
            for obj_1 in obj_pair_dist:
                for obj_2 in obj_pair_dist[obj_1]:
                    gt_dists = obj_pair_dist[obj_1][obj_2]

                    if obj_1 not in pred_obj_pair_dist or obj_2 not in pred_obj_pair_dist[obj_1]:
                        continue
                    pred_dists = pred_obj_pair_dist[obj_1][obj_2]
                    
                    # hungarian match min distances
                    dist_error, dist_acc = compute_location_error(pred_dists, gt_dists, max_dimension)
                    house_obj_loc_error.append(dist_error/max_dimension)
                    house_obj_loc_acc.append(dist_acc)
                    
            # pdb.set_trace()\
            if len(house_obj_loc_error) == 0:
                continue
            self.object_location_error.append(np.mean(house_obj_loc_error))
            self.object_location_accuracy.append(np.mean(house_obj_loc_acc))
        # pdb.set_trace()
        if len(self.object_location_accuracy) == 0:
            mean_obj_loc_acc = 0
            mean_obj_loc_err = 1
        else:
            mean_obj_loc_acc = np.mean(self.object_location_accuracy)
            mean_obj_loc_err = np.mean(self.object_location_error)
        
        self.logger.log({"ObjectDistancesError": mean_obj_loc_err})
        self.logger.log({"ObjectDistancesAcc": mean_obj_loc_acc})



class HouseSelectedObjectDistancesAccuracy:
    def __init__(self, args):
        self.house_jsons = []
        self.gt_house_jsons = []
        self.selected_objs = []
        self.object_location_error = []
        self.object_location_accuracy = []
        self.logger = args['logger']
        self.selected_objs = []
    
    def update(self, output, gt):
        if "#room" in output:
            room_response = output.split(": #room \n")[-1]
        else:
            room_response = output.split(": \n")[-1]

        room_json_text = "\n".join(room_response.split("\n")[:-1])
        room_json_text = room_json_text.split("###")[0]
        room_json_text = room_json_text.replace("(", "[")
        room_json_text = room_json_text.replace(")", "]")
        
        try:
            pred_house_dict = yaml.load(room_json_text, Loader=yaml.FullLoader)
        except:
            pred_house_dict = {}
        if pred_house_dict is None:
            print("none house dict")
            print(room_json_text)
        
        self.house_jsons.append(pred_house_dict)
        
        # convert gt program text to gen config
        gt_house_text = gt['text_labels'][0]
        if "#room" in gt_house_text:
            gt_house_text = gt_house_text.split(": #room \n")[-1]
        else:
            gt_house_text = gt_house_text.split(": \n")[-1]
        gt_house_text = gt_house_text.replace("(", "[")
        gt_house_text = gt_house_text.replace(")", "]")
        # pdb.set_trace()

        try:
            gt_house_dict = yaml.load(gt_house_text, Loader=yaml.FullLoader)
        except:
            gt_house_dict = {}

        if gt_house_dict is None:
            print("none house dict GT!")
            print(gt_house_text)
        self.gt_house_jsons.append(gt_house_dict)
        
        self.selected_objs.append(gt['objs_present'][0])

    def compute(self):
        for pred_house, gt_house, selected_obj in zip(self.house_jsons, self.gt_house_jsons, self.selected_objs):   

            house_obj_loc_error = []
            house_obj_loc_acc = []
            if pred_house is None or gt_house is None:
                continue
            if len(pred_house) == 0 or len(gt_house) == 0:
                continue

            pred_polygon = pred_house['polygon']
            gt_polygon = gt_house['polygon']

            max_coordinate = np.max(gt_polygon)
            min_coordinate = np.min(gt_polygon)
            max_dimension = max_coordinate - min_coordinate

            gt_objs = defaultdict(list)
            i=0
            while(True):
                if f'obj_{i}' not in gt_house:
                    break
                gt_object, location, rotation = gt_house[f'obj_{i}']
                gt_objs[gt_object].append(location)
                i += 1
            
            pred_objs = defaultdict(list)
            i=0
            while(True):
                if f'obj_{i}' not in pred_house:
                    break
                try:
                    pred_object, location, rotation = pred_house[f'obj_{i}']
                except:
                    print("error object")
                    i+=1
                    continue
                pred_objs[pred_object].append(location)
                i+=1
            
            pred_obj_classes = defaultdict(list)
            
            for obj in pred_objs:
                obj_name = get_object_class_from_asset(obj)
                pred_obj_classes[obj_name].extend(pred_objs[obj])
            
            pred_locations = {}
            gt_locations = defaultdict(list)
            for obj in gt_objs:
                obj_name = get_object_class_from_asset(obj)
                if obj_name in pred_obj_classes:
                    pred_locations[obj_name] = pred_obj_classes[obj_name]
                    gt_locations[obj_name].extend(gt_objs[obj])
                
            
            # compute distance of each object to other objects
            obj_pair_dist = defaultdict(dict)
            for obj_1 in gt_locations:
                for obj_2 in gt_locations:
                    if obj_1 == obj_2:
                        continue
                    loc_1 = gt_locations[obj_1]
                    loc_2 = gt_locations[obj_2]
                    
                    all_dist = []
                    for l1 in loc_1:
                        for l2 in loc_2:
                            dist = np.linalg.norm(np.array(l1) - np.array(l2))
                            all_dist.append(dist)

                    obj_pair_dist[obj_1][obj_2] = all_dist
            
            pred_obj_pair_dist = defaultdict(dict)
            for obj_1 in pred_locations:
                for obj_2 in pred_locations:
                    if obj_1 == obj_2:
                        continue
                    loc_1 = pred_locations[obj_1]
                    loc_2 = pred_locations[obj_2]
                    
                    all_dist = []
                    for l1 in loc_1:
                        for l2 in loc_2:
                            dist = np.linalg.norm(np.array(l1) - np.array(l2))
                            all_dist.append(dist)
                        
                    pred_obj_pair_dist[obj_1][obj_2] = all_dist
            
            # compute distance error
            for obj_1 in obj_pair_dist:
                for obj_2 in obj_pair_dist[obj_1]:
                    gt_dists = obj_pair_dist[obj_1][obj_2]

                    if obj_1 not in pred_obj_pair_dist or obj_2 not in pred_obj_pair_dist[obj_1]:
                        continue
                        
                    if obj_1 in selected_obj and obj_2 in selected_obj:
                        pred_dists = pred_obj_pair_dist[obj_1][obj_2]
                        
                        # hungarian match min distances
                        dist_error, dist_acc = compute_location_error(pred_dists, gt_dists, max_dimension)
                        house_obj_loc_error.append(dist_error/max_dimension)
                        house_obj_loc_acc.append(dist_acc)
                    
            # pdb.set_trace()\
            if len(house_obj_loc_error) == 0:
                continue
            self.object_location_error.append(np.mean(house_obj_loc_error))
            self.object_location_accuracy.append(np.mean(house_obj_loc_acc))
        # pdb.set_trace()
        if len(self.object_location_accuracy) == 0:
            mean_obj_loc_acc = 0
            mean_obj_loc_err = 1
        else:
            mean_obj_loc_acc = np.mean(self.object_location_accuracy)
            mean_obj_loc_err = np.mean(self.object_location_error)
        
        self.logger.log({"SelectedObjectDistancesError": mean_obj_loc_err})
        self.logger.log({"SelectedObjectDistancesAcc": mean_obj_loc_acc})



class HouseSelectedObjAccuracy:
    def __init__(self, args):
        self.house_jsons = []
        self.gt_house_jsons = []
        self.selected_objs = []
        self.object_location_error = []
        self.object_location_accuracy = []
        self.object_pose_accuracy = []
        self.object_pose_error = []
        self.object_class_accuracy = []
        self.object_finegrain_accuracy = []
        self.object_count_error = []
        self.object_count_acc = []
        self.logger = args['logger']
        pass

    def update(self, output, gt):
        if "#room" in output:
            room_response = output.split(": #room \n")[-1]
        else:
            room_response = output.split(": \n")[-1]

        room_json_text = "\n".join(room_response.split("\n")[:-1])
        room_json_text = room_json_text.split("###")[0]
        room_json_text = room_json_text.replace("(", "[")
        room_json_text = room_json_text.replace(")", "]")

        try:
            pred_house_dict = yaml.load(room_json_text, Loader=yaml.FullLoader)
        except:
            pred_house_dict = {}
        if pred_house_dict is None:
            print("none house dict")
            print(room_json_text)
        
        self.house_jsons.append(pred_house_dict)
        
        # convert gt program text to gen config
        gt_house_text = gt['text_labels'][0]
        if "#room" in gt_house_text:
            gt_house_text = gt_house_text.split(": #room \n")[-1]
        else:
            gt_house_text = gt_house_text.split(": \n")[-1]
        gt_house_text = gt_house_text.replace("(", "[")
        gt_house_text = gt_house_text.replace(")", "]")
        # pdb.set_trace()

        try:
            gt_house_dict = yaml.load(gt_house_text, Loader=yaml.FullLoader)
        except:
            gt_house_dict = {}

        if gt_house_dict is None:
            print("none house dict GT!")
            print(gt_house_text)
        self.gt_house_jsons.append(gt_house_dict)
        
        self.selected_objs.append(gt['objs_present'][0])
        
    
    def compute(self):
        
        for pred_house, gt_house, selected_objs in zip(self.house_jsons, self.gt_house_jsons, self.selected_objs):
            if pred_house is None or gt_house is None:
                    continue
            if len(pred_house) == 0 or len(gt_house) == 0:
                continue
            # check accuracy of room polygon i.e number of sides match
            pred_polygon = pred_house['polygon']
            gt_polygon = gt_house['polygon']

            max_coordinate = np.max(gt_polygon)
            min_coordinate = np.min(gt_polygon)
            max_dimension = max_coordinate - min_coordinate

            # check accuracy of object classes
            # make counts of objects and locations
            gt_objs = defaultdict(list)
            i=0
            while(True):
                if f'obj_{i}' not in gt_house:
                    break
                gt_object, location, rotation = gt_house[f'obj_{i}']
                gt_objs[gt_object].append((location, rotation))
                i += 1
            
            pred_objs = defaultdict(list)
            i=0
            while(True):
                if f'obj_{i}' not in pred_house:
                    break
                try:
                    pred_object, location, rotation = pred_house[f'obj_{i}']
                except:
                    print("error object")
                    i+=1
                    continue
                pred_objs[pred_object].append((location, rotation))
                i += 1

            pred_obj_classes = defaultdict(list)
            for obj in pred_objs:
                obj_class = get_object_class_from_asset(obj)
                pred_obj_classes[obj_class].extend(pred_objs[obj])
            
            if len(selected_objs)>0:
                selected_obj_classes = [get_object_class_from_asset(obj) for obj in selected_objs]
            else:
                selected_obj_classes = []
            
            pred_locations = {}
            gt_locations = defaultdict(list)
            for obj in gt_objs:
                obj_class = get_object_class_from_asset(obj)
                if obj_class in pred_obj_classes:
                    if obj_class in selected_obj_classes:
                        self.object_class_accuracy.append(1)
                    pred_locations[obj_class] = pred_obj_classes[obj_class]
                    gt_locations[obj_class].extend(gt_objs[obj])
                else:
                    if obj_class in selected_obj_classes:
                        self.object_class_accuracy.append(0)

            # compute hungarian matching of predicted object locations to gt object locations
            for obj in gt_locations:
                pred_loc = pred_locations[obj]
                gt_loc = gt_locations[obj]
               
                pred_loc = np.array(pred_loc).astype(np.float32)
                gt_loc = np.array(gt_loc).astype(np.float32)
                
                # mean_dist, accuracy = compute_location_error(pred_loc, gt_loc, max_dimension)
                mean_dist, accuracy, pose_accuracy, mean_pose_dist = compute_locationpose_error(pred_loc, gt_loc, max_dimension)
                if obj in selected_obj_classes:
                    self.object_location_accuracy.append(accuracy)
                    self.object_location_error.append(mean_dist/max_dimension)
                    self.object_pose_accuracy.append(pose_accuracy)
                    self.object_pose_error.append(mean_pose_dist)
            
            for obj in gt_objs:
                obj_class = get_object_class_from_asset(obj)
                if obj_class in selected_obj_classes:
                    if obj in pred_objs:
                        self.object_finegrain_accuracy.append(1)
                    else:
                        self.object_finegrain_accuracy.append(0)
            
            for obj in gt_objs:
                obj_class = get_object_class_from_asset(obj)
                if obj_class in selected_obj_classes:
                    if obj in pred_objs:
                        count_diff = abs(len(gt_objs[obj]) - len(pred_objs[obj]))
                        self.object_count_error.append(count_diff)
                        if count_diff == 0:
                            self.object_count_acc.append(1)
                        else:
                            self.object_count_acc.append(0)
                    else:
                        self.object_count_error.append(len(gt_objs[obj]))
                        self.object_count_acc.append(0)

            
        if len(self.object_location_accuracy) == 0:
            return
        # compute mean of all accuracies
        object_class_acc = np.mean(self.object_class_accuracy)
        object_location_err = np.mean(self.object_location_error)
        object_location_acc = np.mean(self.object_location_accuracy)
        object_pose_acc = np.mean(self.object_pose_accuracy)
        object_pose_err = np.mean(self.object_pose_error)
        object_finegrain_acc = np.mean(self.object_finegrain_accuracy)
        object_count_err = np.mean(self.object_count_error)
        object_count_acc = np.mean(self.object_count_acc)

        self.logger.log({"SelObjectClassAcc": object_class_acc})
        self.logger.log({"SelObjectLocationError": object_location_err})
        self.logger.log({"SelObjectLocationAcc": object_location_acc})
        self.logger.log({"SelObjectPoseAcc": object_pose_acc})
        self.logger.log({"SelObjectPoseError": object_pose_err})
        self.logger.log({"SelObjectFinegrainAcc": object_finegrain_acc})
        self.logger.log({"SelObjectCountError": object_count_err})
        self.logger.log({"SelObjectCountAcc": object_count_acc})



class HouseSemanticSimilarity:
    def __init__(self, args):
            
        self.sim_scores = []

        self.house_jsons = []
        self.gt_house_jsons = []
        self.selected_objs = []

        self.polygon_accuracy = []
        self.floor_material_accuracy = []
        self.wall_material_accuracy = []
        self.object_class_accuracy = []
        self.object_location_error = []
        self.object_location_accuracy = []
        self.object_pose_accuracy = []
        self.object_pose_error = []
        self.objectsel_location_accuracy = []
        self.object_count_error = []
        self.num_corner_error = []
        self.object_finegrain_accuracy = []
        self.window_accuracy= []
        self.ignore_loc = []

        self.logger = args['logger']

    def update(self, output, gt):
        # compute text sim
        # convert output program text to gen config
        # pdb.set_trace()
        if "#room" in output:
            room_response = output.split(": #room \n")[-1]
        else:
            room_response = output.split(": \n")[-1]
        # pdb.set_trace()
        room_json_text = "\n".join(room_response.split("\n")[:-1])
        room_json_text = room_json_text.split("###")[0]
        room_json_text = room_json_text.replace("(", "[")
        room_json_text = room_json_text.replace(")", "]")
        
        try:
            pred_house_dict = yaml.load(room_json_text, Loader=yaml.FullLoader)
        except:
            pred_house_dict = {}
        
        if pred_house_dict is None:
            print("none house dict")
            print(room_json_text)
        self.house_jsons.append(pred_house_dict)
        
        # convert gt program text to gen config
        og_gt_house_text = gt['text_labels'][0]
        if "#room" in og_gt_house_text:
            gt_house_text = og_gt_house_text.split(": #room \n")[-1]
        else:
            gt_house_text = og_gt_house_text.split(": \n")[-1]
        gt_house_text = gt_house_text.replace("(", "[")
        gt_house_text = gt_house_text.replace(")", "]")
        # pdb.set_trace()

        try:
            gt_house_dict = yaml.load(gt_house_text, Loader=yaml.FullLoader)
        except:
            gt_house_dict = {}

        if gt_house_dict is None:
            print("none house dict GT!")
            print(gt_house_text)
        self.gt_house_jsons.append(gt_house_dict)

        if "The red circular 1 mark in the image is at 3D coordinate (x, y, z)" in og_gt_house_text:
            ignore_loc = og_gt_house_text.split("The red circular 1 mark in the image is at 3D coordinate (x, y, z)")[-1].split(".")[0]
            ignore_loc = ignore_loc.strip()
            ignore_loc = ast.literal_eval(ignore_loc)
            self.ignore_loc.append(ignore_loc)
        # pdb.set_trace()
        # self.selected_objs.append(gt['objs_present'][0][0])

    def compute(self):
        
        for ignore_ind, (pred_house, gt_house) in enumerate(zip(self.house_jsons, self.gt_house_jsons)):
            # pdb.set_trace()
            try:
                if pred_house is None or gt_house is None:
                    continue
                if len(pred_house) == 0 or len(gt_house) == 0:
                    continue
                # check accuracy of room polygon i.e number of sides match
                pred_polygon = pred_house['polygon']
                gt_polygon = gt_house['polygon']

                max_coordinate = np.max(gt_polygon)
                min_coordinate = np.min(gt_polygon)
                max_dimension = max_coordinate - min_coordinate

                # convert to shapely polygon and simplify
                pred_polygon_simple = Polygon(pred_polygon).simplify(0.01)
                gt_polygon_simple = Polygon(gt_polygon).simplify(0.01)


                self.polygon_accuracy.append(len(pred_polygon_simple.exterior.coords) == len(gt_polygon_simple.exterior.coords))
                self.num_corner_error.append(abs(len(pred_polygon_simple.exterior.coords) - len(gt_polygon_simple.exterior.coords)))

                # check accuracy of wall materials. wall materials are lists
                pred_materials = np.array(pred_house['wall_material'])
                gt_materials = np.array(gt_house['wall_material'])

                polygon_len = min(len(pred_materials), len(gt_materials))
                self.wall_material_accuracy.append(np.mean(pred_materials[:polygon_len] == gt_materials[:polygon_len]))

                # check accuracy of floor materials
                self.floor_material_accuracy.append(pred_house['floor_material'] == gt_house['floor_material'])

                # check accuracy of object classes
                # make counts of objects and locations
                gt_objs = defaultdict(list)
                i=0
                while(True):
                    if f'obj_{i}' not in gt_house:
                        break
                    gt_object, location, rotation = gt_house[f'obj_{i}']
                    gt_objs[gt_object].append((location, rotation))
                    i += 1
                
                pred_objs = defaultdict(list)
                i=0
                while(True):
                    if f'obj_{i}' not in pred_house:
                        break
                    try:
                        pred_object, location, rotation = pred_house[f'obj_{i}']
                    except:
                        print("error object")
                        i+=1
                        continue
                    pred_objs[pred_object].append((location, rotation))
                    i += 1

                pred_obj_classes = defaultdict(list)
                for obj in pred_objs:
                    obj_class = get_object_class_from_asset(obj)
                    pred_obj_classes[obj_class].extend(pred_objs[obj])
                
                pred_locations = {}
                gt_locations = defaultdict(list)
                for obj in gt_objs:
                    obj_class = get_object_class_from_asset(obj)
                    if obj_class in pred_obj_classes:
                        self.object_class_accuracy.append(1)
                        pred_locations[obj_class] = pred_obj_classes[obj_class]
                        gt_locations[obj_class].extend(gt_objs[obj])
                    else:
                        self.object_class_accuracy.append(0)

                # compute hungarian matching of predicted object locations to gt object locations
                # if len(selected_objs)>0:
                #    selected_obj_classes = [obj.split(" ")[0] for obj in selected_objs]
                if len(self.ignore_loc)>0:
                    ignore_locs = self.ignore_loc[ignore_ind]
                else:
                    ignore_locs = -1
                for obj in gt_locations:
                    pred_loc = pred_locations[obj]
                    gt_loc = gt_locations[obj]
                    if gt_loc[0] == ignore_locs:
                        continue
                    try:
                        pred_loc = np.array(pred_loc).astype(np.float32)
                        gt_loc = np.array(gt_loc).astype(np.float32)
                    except:
                         self.object_location_accuracy.append(0)
                         self.object_location_error.append(1)
                         continue
                    # mean_dist, accuracy = compute_location_error(pred_loc, gt_loc, max_dimension)
                    mean_dist, accuracy, pose_accuracy, mean_pose_dist = compute_locationpose_error(pred_loc, gt_loc, max_dimension)
                    self.object_location_accuracy.append(accuracy)
                    self.object_location_error.append(mean_dist/max_dimension)
                    self.object_pose_accuracy.append(pose_accuracy)
                    self.object_pose_error.append(mean_pose_dist)

                    # if len(selected_objs)>0:
                    #     if obj in selected_obj_classes:
                    #         self.objectsel_location_accuracy.append(accuracy)


                for obj in gt_objs:
                    if obj in pred_objs:
                        self.object_finegrain_accuracy.append(1)
                    else:
                        self.object_finegrain_accuracy.append(0)
                
                for obj in gt_objs:
                    if obj in pred_objs:
                        count_diff = abs(len(gt_objs[obj]) - len(pred_objs[obj]))
                        self.object_count_error.append(count_diff)
                    else:
                        self.object_count_error.append(len(gt_objs[obj]))
            except:
                print("some error in JSON")
                continue

        if len(self.polygon_accuracy) == 0:
            return
        # compute mean of all accuracies
        polygon_acc = np.mean(self.polygon_accuracy)
        wall_material_acc = np.mean(self.wall_material_accuracy)
        floor_material_acc = np.mean(self.floor_material_accuracy)
        object_class_acc = np.mean(self.object_class_accuracy)
        num_corner_err = np.mean(self.num_corner_error)
        object_count_err = np.mean(self.object_count_error)
        object_finegrain_acc = np.mean(self.object_finegrain_accuracy)
        object_location_err = np.mean(self.object_location_error)
        object_location_acc = np.mean(self.object_location_accuracy)
        object_pose_acc = np.mean(self.object_pose_accuracy)
        object_pose_err = np.mean(self.object_pose_error)
        #if len(self.objectsel_location_accuracy)>0:
        #    objectsel_location_acc = np.mean(self.objectsel_location_accuracy)
        #else:
        #    objectsel_location_acc = -1

        # log
        self.logger.log({"PolygonAcc": polygon_acc})
        self.logger.log({"WallMaterialAcc": wall_material_acc})
        self.logger.log({"FloorMaterialAcc": floor_material_acc})
        self.logger.log({"ObjectClassAcc": object_class_acc})
        self.logger.log({"NumCornerError": num_corner_err})
        self.logger.log({"ObjectCountError": object_count_err})
        self.logger.log({"ObjectFinegrainAcc": object_finegrain_acc})
        self.logger.log({"ObjectLocationError": object_location_err})
        self.logger.log({"ObjectLocationAcc": object_location_acc})
        self.logger.log({"ObjectPoseAcc": object_pose_acc})
        self.logger.log({"ObjectPoseError": object_pose_err})
        # self.logger.log({"ObjectSelLocationAcc": objectsel_location_acc})



class HouseNatLanguageSemSimSelected:
    def __init__(self, args):
            
        self.house_jsons = []
        self.gt_house_jsons = []
        self.selected_objs = []
        self.object_location_error = []
        self.object_location_accuracy = []
        self.object_pose_accuracy = []
        self.object_pose_error = []
        self.object_class_accuracy = []
        self.object_finegrain_accuracy = []
        self.object_count_error = []
        self.object_count_acc = []
        self.object_attr_similarity = []
        self.json_accuracy = []
        self.logger = args['logger']


        self.assetdesc_to_obj = defaultdict(list)
        asset_descs = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json"))
        
        for asset in asset_descs:
            entries = asset_descs[asset]
            for im, obj, desc in entries:
                self.assetdesc_to_obj[desc].append((obj, asset))

        
        self.attrtokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        self.attrmodel = AutoModel.from_pretrained('intfloat/e5-base-v2').cuda()

    
    def update(self, output, gt):
        if "#room" in output:
            room_response = output.split(": #room \n")[-1]
        else:
            room_response = output.split(": \n")[-1]

        room_json_text = "\n".join(room_response.split("\n")[:-1])
        room_json_text = room_json_text.split("###")[0]
        room_json_text = room_json_text.replace("(", "[")
        room_json_text = room_json_text.replace(")", "]")

        try:
            pred_house_dict = yaml.load(room_json_text, Loader=yaml.FullLoader)
        except:
            pred_house_dict = {}
        if pred_house_dict is None:
            print("none house dict")
            print(room_json_text)
        
        self.house_jsons.append(pred_house_dict)
        
        # convert gt program text to gen config
        gt_house_text = gt['text_labels'][0]
        if "#room" in gt_house_text:
            gt_house_text = gt_house_text.split(": #room \n")[-1]
        else:
            gt_house_text = gt_house_text.split(": \n")[-1]
        gt_house_text = gt_house_text.replace("(", "[")
        gt_house_text = gt_house_text.replace(")", "]")
        # pdb.set_trace()

        try:
            gt_house_dict = yaml.load(gt_house_text, Loader=yaml.FullLoader)
        except:
            gt_house_dict = {}

        if gt_house_dict is None:
            print("none house dict GT!")
            print(gt_house_text)
        self.gt_house_jsons.append(gt_house_dict)
        
        # self.selected_objs.append(gt['objs_present'][0])
        
    
    def compute(self):
        for pred_house, gt_house in tqdm.tqdm(zip(self.house_jsons, self.gt_house_jsons)):
            if pred_house is None:
                self.json_accuracy.append(0)
                continue
            if len(pred_house) == 0:
                self.json_accuracy.append(0)
                continue
            
            if pred_house is None or gt_house is None:
                continue
            if len(pred_house) == 0 or len(gt_house) == 0:
                continue

            # pdb.set_trace()
            # check accuracy of room polygon i.e number of sides match
            #pred_polygon = pred_house['polygon']
            gt_polygon = gt_house['polygon']

            max_coordinate = np.max(gt_polygon)
            min_coordinate = np.min(gt_polygon)
            max_dimension = max_coordinate - min_coordinate

            # check accuracy of object classes
            # make counts of objects and locations
            gt_objs = defaultdict(list)
            i=0
            while(True):
                if f'obj_{i}' not in gt_house and f'Object {i}' not in gt_house:
                    break

                obj_entry = gt_house[f'obj_{i}'] if f'obj_{i}' in gt_house else gt_house[f'Object {i}']
                obj_desc = obj_entry.split("at location")[0].strip()
                obj_loc = ast.literal_eval(obj_entry.split("at location")[-1].split("with rotation")[0].strip())
                obj_rot = ast.literal_eval(obj_entry.split("with rotation")[-1].strip())
                
                
                gt_objs[obj_desc].append((obj_loc, obj_rot, obj_desc))
                i += 1
            
            pred_objs = defaultdict(list)
            i=1
            while(True):
                if f'obj_{i}' not in pred_house and f'Object {i}' not in pred_house:
                    break
                
                obj_entry = pred_house[f'obj_{i}'] if f'obj_{i}' in pred_house else pred_house[f'Object {i}']
                obj_desc = obj_entry.split("at location")[0].strip()
                try:
                    obj_loc = ast.literal_eval(obj_entry.split("at location")[-1].split("with rotation")[0].strip())
                    obj_rot = ast.literal_eval(obj_entry.split("with rotation")[-1].split(" degrees")[0].strip())
                except:
                    i += 1
                    continue
                
                # check if obj_rot is list
                if type(obj_rot) == list:
                    pass
                else:
                    obj_rot = [0, obj_rot, 0]
                pred_objs[obj_desc].append((obj_loc, obj_rot, obj_desc))
                i += 1

            if len(pred_objs) == 0:
                self.json_accuracy.append(0)
                continue
            else:
                self.json_accuracy.append(1)

            pred_obj_classes = defaultdict(list)
            for obj in pred_objs:
                obj_class = get_obj_class_from_desc_pred(obj, self.assetdesc_to_obj, self.attrmodel, self.attrtokenizer)[0][0]
                pred_obj_classes[obj_class].extend(pred_objs[obj])
            
            #if len(selected_objs)>0:
            #    selected_obj_classes = [self.assetdesc_to_obj[obj][0][0] for obj in selected_objs]
            #else:
            #    selected_obj_classes = []
            
            pred_locations = {}
            gt_locations = defaultdict(list)
            for obj in gt_objs:
                obj_class = self.assetdesc_to_obj[obj][0][0]
                if obj_class in pred_obj_classes:
                    # if obj_class in selected_obj_classes:
                    self.object_class_accuracy.append(1)
                    pred_locations[obj_class] = pred_obj_classes[obj_class]
                    gt_locations[obj_class].extend(gt_objs[obj])
                else:
                    # if obj_class in selected_obj_classes:
                    self.object_class_accuracy.append(0)
            
            # pdb.set_trace()
            # compute hungarian matching of predicted object locations to gt object locations
            for obj in gt_locations:
                pred_loc = pred_locations[obj]
                gt_loc = gt_locations[obj]

                # mean_dist, accuracy = compute_location_error(pred_loc, gt_loc, max_dimension)
                # pdb.set_trace()
                mean_dist, accuracy, pose_accuracy, mean_pose_dist, attr_sim = compute_locationposeattr_error(pred_loc, gt_loc, max_dimension, self.attrmodel, self.attrtokenizer)
                # if obj in selected_obj_classes:
                self.object_location_accuracy.append(accuracy)
                self.object_location_error.append(mean_dist/max_dimension)
                self.object_pose_accuracy.append(pose_accuracy)
                self.object_pose_error.append(mean_pose_dist)
                self.object_attr_similarity.append(attr_sim)
            
            
            all_gt_assets = []
            for obj in gt_objs:
                gt_obj = self.assetdesc_to_obj[obj]
                gt_assets = [obj[1] for obj in gt_obj]
                all_gt_assets.extend(gt_assets)

            all_pred_assets = []
            for obj in pred_objs:
                pred_obj = get_obj_class_from_desc_pred(obj, self.assetdesc_to_obj, self.attrmodel, self.attrtokenizer)
                pred_assets = [obj[1] for obj in pred_obj]
                all_pred_assets.extend(pred_assets)

            
            pred_prec = []
            pred_recall = []
            for asset in all_pred_assets:
                if asset in all_gt_assets:
                    pred_prec.append(1)
                else:
                    pred_prec.append(0)
            for asset in all_gt_assets:
                if asset in all_pred_assets:
                    pred_recall.append(1)
                else:
                    pred_recall.append(0)
            
            if np.mean(pred_prec) == 0 and np.mean(pred_recall) == 0:
                pred_f1 = 0
            else:
                pred_f1 = 2*np.mean(pred_prec)*np.mean(pred_recall)/(np.mean(pred_prec)+np.mean(pred_recall))
            self.object_finegrain_accuracy.append(pred_f1)
                
            
            for obj in gt_objs:
                obj_class = self.assetdesc_to_obj[obj][0][0]
                # if obj_class in selected_obj_classes:
                if obj in pred_objs:
                    count_diff = abs(len(gt_objs[obj]) - len(pred_objs[obj]))
                    self.object_count_error.append(count_diff)
                    if count_diff == 0:
                        self.object_count_acc.append(1)
                    else:
                        self.object_count_acc.append(0)
                else:
                    self.object_count_error.append(len(gt_objs[obj]))
                    self.object_count_acc.append(0)

        # pdb.set_trace()
        if len(self.object_location_accuracy) == 0:
            return
        # compute mean of all accuracies
        object_class_acc = np.mean(self.object_class_accuracy)
        object_location_err = np.mean(self.object_location_error)
        object_location_acc = np.mean(self.object_location_accuracy)
        object_pose_acc = np.mean(self.object_pose_accuracy)
        object_pose_err = np.mean(self.object_pose_error)
        object_finegrain_acc = np.mean(self.object_finegrain_accuracy)
        object_count_err = np.mean(self.object_count_error)
        object_count_acc = np.mean(self.object_count_acc)
        object_attr_sim = np.mean(self.object_attr_similarity)
        json_acc = np.mean(self.json_accuracy)

        
        self.logger.log({"SelObjectClassAcc": object_class_acc})
        self.logger.log({"SelObjectLocationError": object_location_err})
        self.logger.log({"SelObjectLocationAcc": object_location_acc})
        self.logger.log({"SelObjectPoseAcc": object_pose_acc})
        self.logger.log({"SelObjectPoseError": object_pose_err})
        self.logger.log({"SelObjectFinegrainAcc": object_finegrain_acc})
        self.logger.log({"SelObjectCountError": object_count_err})
        self.logger.log({"SelObjectCountAcc": object_count_acc})
        self.logger.log({"SelObjectAttrSim": object_attr_sim})
        self.logger.log({"JSONAccuracy": json_acc})


class HouseNatLanguageSemSimSelectedGPT4:
    def __init__(self, args):
            
        self.house_jsons = []
        self.gt_house_jsons = []
        self.selected_objs = []
        self.object_location_error = []
        self.object_location_accuracy = []
        self.object_pose_accuracy = []
        self.object_pose_error = []
        self.object_class_accuracy = []
        self.object_finegrain_accuracy = []
        self.object_count_error = []
        self.object_count_acc = []
        self.object_attr_similarity = []
        self.json_accuracy = []
        self.logger = args['logger']


        self.assetdesc_to_obj = defaultdict(list)
        asset_descs = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json"))

        for asset_id in asset_descs:
            for asst_entry in asset_descs[asset_id]:
                im_f, obj_class, desc = asst_entry    
                self.assetdesc_to_obj[desc].append((obj_class, asset_id))

        self.attrtokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        self.attrmodel = AutoModel.from_pretrained('intfloat/e5-base-v2').cuda()

    
    def update(self, output, gt):
        
        room_json_text = output.split("Answer:")[-1].strip()
        room_json_text = room_json_text.replace("(", "[")
        room_json_text = room_json_text.replace(")", "]")

        # pdb.set_trace()
        try:
            room_list = room_json_text.split("\n")
            if len(room_list) == 1:
                room_list = room_json_text.split(".")
            pred_house_dict = {}
            for entry in room_list:
                obj = entry.split(" at location ")[0]
                loc = entry.split(" at location ")[-1].split(".")[0].strip()
                pred_house_dict[obj] = loc

        except:
            pred_house_dict = {}
        if pred_house_dict is None:
            print("none house dict")
            print(room_json_text)
        
        self.house_jsons.append(pred_house_dict)
        
        # convert gt program text to gen config
        gt_house_text = gt['text_labels'][0]
        if "#room" in gt_house_text:
            gt_house_text = gt_house_text.split(": #room \n")[-1]
        else:
            gt_house_text = gt_house_text.split(": \n")[-1]
        gt_house_text = gt_house_text.replace("(", "[")
        gt_house_text = gt_house_text.replace(")", "]")
        # pdb.set_trace()

        try:
            gt_house_dict = yaml.load(gt_house_text, Loader=yaml.FullLoader)
        except:
            gt_house_dict = {}

        if gt_house_dict is None:
            print("none house dict GT!")
            print(gt_house_text)
        self.gt_house_jsons.append(gt_house_dict)
        
        # self.selected_objs.append(gt['objs_present'])
        
    
    def compute(self):
        for pred_house, gt_house in tqdm.tqdm(zip(self.house_jsons, self.gt_house_jsons,)):
            if pred_house is None:
                self.json_accuracy.append(0)
                continue
            if len(pred_house) == 0:
                self.json_accuracy.append(0)
                continue
            
            if pred_house is None or gt_house is None:
                continue
            if len(pred_house) == 0 or len(gt_house) == 0:
                continue

            # pdb.set_trace()
            # check accuracy of room polygon i.e number of sides match
            #pred_polygon = pred_house['polygon']
            gt_polygon = gt_house['polygon']

            max_coordinate = np.max(gt_polygon)
            min_coordinate = np.min(gt_polygon)
            max_dimension = max_coordinate - min_coordinate

            # check accuracy of object classes
            # make counts of objects and locations
            gt_objs = defaultdict(list)
            i=0
            objid_to_class = {}

            while(True):
                if f'obj_{i}' not in gt_house and f'Object {i}' not in gt_house:
                    break

                obj_entry = gt_house[f'obj_{i}'] if f'obj_{i}' in gt_house else gt_house[f'Object {i}']
                obj_desc = obj_entry.split("at location")[0].strip()
                obj_loc = ast.literal_eval(obj_entry.split("at location")[-1].split("with rotation")[0].strip())
                obj_rot = ast.literal_eval(obj_entry.split("with rotation")[-1].strip())
                objid_to_class[f'obj_{i}'] = get_obj_class_from_desc_pred(obj_desc, self.assetdesc_to_obj, self.attrmodel, self.attrtokenizer)[0][0]
                
                gt_objs[obj_desc].append((obj_loc, [0, 0, 0], obj_desc))
                i += 1
            
            #selected_objs = [objid_to_class[obj] for obj in selected_objs]
            pred_objs = defaultdict(list)
            
            for obj_desc in pred_house:
                try:
                    obj_loc = pred_house[obj_desc]
                except:
                    pdb.set_trace()
                # pdb.set_trace()
                if isinstance(obj_loc, str):
                    obj_loc = obj_loc.strip()
                    if "]" not in obj_loc:
                        obj_loc+="]"
                    if "[" not in obj_loc:
                        obj_loc = "["+obj_loc
                    try:
                        obj_loc = ast.literal_eval(obj_loc)
                    except:
                        self.json_accuracy.append(0)
                        continue
                    if not isinstance(obj_loc, list):
                        self.json_accuracy.append(0)
                        continue
                    if len(obj_loc) != 3:
                        self.json_accuracy.append(0)
                        continue
                
                pred_objs[obj_desc].append((obj_loc, [0, 0, 0], obj_desc))
            # pdb.set_trace()
            if len(pred_objs) == 0:
                self.json_accuracy.append(0)
                continue
            else:
                self.json_accuracy.append(1)

            pred_obj_classes = defaultdict(list)
            for obj in pred_objs:
                obj_class = get_obj_class_from_desc_pred(obj, self.assetdesc_to_obj, self.attrmodel, self.attrtokenizer)[0][0]
                pred_obj_classes[obj_class].extend(pred_objs[obj])
            
            #selected_obj_classes = selected_objs
            
            pred_locations = {}
            gt_locations = defaultdict(list)
            for obj in gt_objs:
                try:
                    obj_class = get_obj_class_from_desc_pred(obj, self.assetdesc_to_obj, self.attrmodel, self.attrtokenizer)[0][0]
                except:
                    pdb.set_trace()
                if obj_class in pred_obj_classes:
                    # if obj_class in selected_obj_classes:
                    self.object_class_accuracy.append(1)
                    pred_locations[obj_class] = pred_obj_classes[obj_class]
                    gt_locations[obj_class].extend(gt_objs[obj])
                else:
                    # if obj_class in selected_obj_classes:
                    self.object_class_accuracy.append(0)
            
            # pdb.set_trace()
            # compute hungarian matching of predicted object locations to gt object locations
            for obj in gt_locations:
                pred_loc = pred_locations[obj]
                gt_loc = gt_locations[obj]

                # mean_dist, accuracy = compute_location_error(pred_loc, gt_loc, max_dimension)
                mean_dist, accuracy, pose_accuracy, mean_pose_dist, attr_sim = compute_locationposeattr_error(pred_loc, gt_loc, max_dimension, self.attrmodel, self.attrtokenizer)
                # if obj in selected_obj_classes:
                self.object_location_accuracy.append(accuracy)
                self.object_location_error.append(mean_dist/max_dimension)
                self.object_pose_accuracy.append(pose_accuracy)
                self.object_pose_error.append(mean_pose_dist)
                self.object_attr_similarity.append(attr_sim)
            all_gt_assets = []
            for obj in gt_objs:
                gt_obj = self.assetdesc_to_obj[obj]
                gt_assets = [obj[1] for obj in gt_obj]
                all_gt_assets.extend(gt_assets)

            all_pred_assets = []
            for obj in pred_objs:
                pred_obj = get_obj_class_from_desc_pred(obj, self.assetdesc_to_obj, self.attrmodel, self.attrtokenizer)
                pred_assets = [obj[1] for obj in pred_obj]
                all_pred_assets.extend(pred_assets)

            pred_prec = []
            pred_recall = []
            for asset in all_pred_assets:
                if asset in all_gt_assets:
                    pred_prec.append(1)
                else:
                    pred_prec.append(0)
            for asset in all_gt_assets:
                if asset in all_pred_assets:
                    pred_recall.append(1)
                else:
                    pred_recall.append(0)
            
            if np.mean(pred_prec) == 0 and np.mean(pred_recall) == 0:
                pred_f1 = 0
            else:
                pred_f1 = 2*np.mean(pred_prec)*np.mean(pred_recall)/(np.mean(pred_prec)+np.mean(pred_recall))
            self.object_finegrain_accuracy.append(pred_f1)
            
            for obj in gt_objs:
                obj_class = get_obj_class_from_desc_pred(obj, self.assetdesc_to_obj, self.attrmodel, self.attrtokenizer)[0][0]
                # if obj_class in selected_obj_classes:
                if obj in pred_objs:
                    count_diff = abs(len(gt_objs[obj]) - len(pred_objs[obj]))
                    self.object_count_error.append(count_diff)
                    if count_diff == 0:
                        self.object_count_acc.append(1)
                    else:
                        self.object_count_acc.append(0)
                else:
                    self.object_count_error.append(len(gt_objs[obj]))
                    self.object_count_acc.append(0)

        # pdb.set_trace()
        if len(self.object_location_accuracy) == 0:
            return
        # compute mean of all accuracies
        object_class_acc = np.mean(self.object_class_accuracy)
        object_location_err = np.mean(self.object_location_error)
        object_location_acc = np.mean(self.object_location_accuracy)
        object_pose_acc = np.mean(self.object_pose_accuracy)
        object_pose_err = np.mean(self.object_pose_error)
        object_finegrain_acc = np.mean(self.object_finegrain_accuracy)
        object_count_err = np.mean(self.object_count_error)
        object_count_acc = np.mean(self.object_count_acc)
        object_attr_sim = np.mean(self.object_attr_similarity)
        json_acc = np.mean(self.json_accuracy)

        
        self.logger.log({"SelObjectClassAcc": object_class_acc})
        self.logger.log({"SelObjectLocationError": object_location_err})
        self.logger.log({"SelObjectLocationAcc": object_location_acc})
        self.logger.log({"SelObjectPoseAcc": object_pose_acc})
        self.logger.log({"SelObjectPoseError": object_pose_err})
        self.logger.log({"SelObjectFinegrainAcc": object_finegrain_acc})
        self.logger.log({"SelObjectCountError": object_count_err})
        self.logger.log({"SelObjectCountAcc": object_count_acc})
        self.logger.log({"SelObjectAttrSim": object_attr_sim})
        self.logger.log({"JSONAccuracy": json_acc})


class AttributeObjectMetrics:
    def __init__(self, args):
        self.attrtokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        self.attrmodel = AutoModel.from_pretrained('intfloat/e5-base-v2').cuda()

        asset_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/asset_descriptions_all.json"))
        self.assetid2desc = {}
        for image_file, asset_name, object_class, caption in asset_desc:
            self.assetid2desc[asset_name] = caption
        
        self.house_jsons = []
        self.gt_house_jsons = []
        self.selected_objs = []

        self.object_class_accuracy = []
        self.object_location_error = []
        self.object_location_accuracy = []
        self.object_pose_accuracy = []
        self.object_pose_error = []
        self.object_finegrain_accuracy = []
        self.ignore_loc = []
        self.object_attr_similarity = []

        self.logger = args['logger']

    def update(self, output, gt):
        # compute text sim
        # convert output program text to gen config
        # pdb.set_trace()
        if "#room" in output:
            room_response = output.split(": #room \n")[-1]
        else:
            room_response = output.split(": \n")[-1]
        # pdb.set_trace()
        room_json_text = "\n".join(room_response.split("\n")[:-1])
        room_json_text = room_json_text.split("###")[0]
        room_json_text = room_json_text.replace("(", "[")
        room_json_text = room_json_text.replace(")", "]")
        
        try:
            pred_house_dict = yaml.load(room_json_text, Loader=yaml.FullLoader)
        except:
            pred_house_dict = {}
        
        if pred_house_dict is None:
            print("none house dict")
            print(room_json_text)
        self.house_jsons.append(pred_house_dict)
        
        # convert gt program text to gen config
        og_gt_house_text = gt['text_labels'][0]
        if "#room" in og_gt_house_text:
            gt_house_text = og_gt_house_text.split(": #room \n")[-1]
        else:
            gt_house_text = og_gt_house_text.split(": \n")[-1]
        gt_house_text = gt_house_text.replace("(", "[")
        gt_house_text = gt_house_text.replace(")", "]")
        # pdb.set_trace()

        try:
            gt_house_dict = yaml.load(gt_house_text, Loader=yaml.FullLoader)
        except:
            gt_house_dict = {}

        if gt_house_dict is None:
            print("none house dict GT!")
            print(gt_house_text)
        self.gt_house_jsons.append(gt_house_dict)

        if "The red circular 1 mark in the image is at 3D coordinate (x, y, z)" in og_gt_house_text:
            ignore_loc = og_gt_house_text.split("The red circular 1 mark in the image is at 3D coordinate (x, y, z)")[-1].split(".")[0]
            ignore_loc = ignore_loc.strip()
            ignore_loc = ast.literal_eval(ignore_loc)
            self.ignore_loc.append(ignore_loc)

    def compute(self):
        for ignore_ind, (pred_house, gt_house) in enumerate(zip(self.house_jsons, self.gt_house_jsons)):
            # pdb.set_trace()
            
            if pred_house is None or gt_house is None:
                continue
            if len(pred_house) == 0 or len(gt_house) == 0:
                continue
            # check accuracy of room polygon i.e number of sides match
            pred_polygon = pred_house['polygon']
            gt_polygon = gt_house['polygon']

            max_coordinate = np.max(gt_polygon)
            min_coordinate = np.min(gt_polygon)
            max_dimension = max_coordinate - min_coordinate

            # convert to shapely polygon and simplify
            pred_polygon_simple = Polygon(pred_polygon).simplify(0.01)
            gt_polygon_simple = Polygon(gt_polygon).simplify(0.01)

            # check accuracy of object classes
            # make counts of objects and locations
            gt_objs = defaultdict(list)
            i=0
            while(True):
                if f'obj_{i}' not in gt_house:
                    break
                gt_object, location, rotation = gt_house[f'obj_{i}']
                gt_objs[gt_object].append((location, rotation, self.assetid2desc[gt_object]))
                i += 1
            
            pred_objs = defaultdict(list)
            i=0
            while(True):
                if f'obj_{i}' not in pred_house:
                    break
                try:
                    pred_object, location, rotation = pred_house[f'obj_{i}']
                except:
                    print("error object")
                    i+=1
                    continue
                pred_objs[pred_object].append((location, rotation, self.assetid2desc.get(pred_object, get_object_class_from_asset(pred_object))))
                i += 1

            pred_obj_classes = defaultdict(list)
            for obj in pred_objs:
                obj_class = get_object_class_from_asset(obj)
                pred_obj_classes[obj_class].extend(pred_objs[obj])
            
            pred_locations = {}
            gt_locations = defaultdict(list)
            for obj in gt_objs:
                obj_class = get_object_class_from_asset(obj)
                if obj_class in pred_obj_classes:
                    self.object_class_accuracy.append(1)
                    pred_locations[obj_class] = pred_obj_classes[obj_class]
                    gt_locations[obj_class].extend(gt_objs[obj])
                else:
                    self.object_class_accuracy.append(0)

            # compute hungarian matching of predicted object locations to gt object locations
            # if len(selected_objs)>0:
            #    selected_obj_classes = [obj.split(" ")[0] for obj in selected_objs]
            if len(self.ignore_loc)>0:
                ignore_locs = self.ignore_loc[ignore_ind]
            else:
                ignore_locs = -1
            for obj in gt_locations:
                pred_loc = pred_locations[obj]
                gt_loc = gt_locations[obj]
                
                # mean_dist, accuracy = compute_location_error(pred_loc, gt_loc, max_dimension)
                mean_dist, accuracy, pose_accuracy, mean_pose_dist, attr_sim = compute_locationposeattr_error(pred_loc, gt_loc, max_dimension, self.attrmodel, self.attrtokenizer)
                self.object_location_accuracy.append(accuracy)
                self.object_location_error.append(mean_dist/max_dimension)
                self.object_pose_accuracy.append(pose_accuracy)
                self.object_pose_error.append(mean_pose_dist)
                self.object_attr_similarity.append(attr_sim)

                # if len(selected_objs)>0:
                #     if obj in selected_obj_classes:
                #         self.objectsel_location_accuracy.append(accuracy)
                
            #except:
            #    print("some error in JSON")
            #    continue

        if len(self.object_location_accuracy) == 0:
            return
        # compute mean of all accuracies
        object_class_acc = np.mean(self.object_class_accuracy)
        object_location_err = np.mean(self.object_location_error)
        object_location_acc = np.mean(self.object_location_accuracy)
        object_pose_acc = np.mean(self.object_pose_accuracy)
        object_pose_err = np.mean(self.object_pose_error)
        object_attr_sim = np.mean(self.object_attr_similarity)
        #if len(self.objectsel_location_accuracy)>0:
        #    objectsel_location_acc = np.mean(self.objectsel_location_accuracy)
        #else:
        #    objectsel_location_acc = -1

        # log
        self.logger.log({"ObjectClassAcc": object_class_acc})
        self.logger.log({"ObjectLocationError": object_location_err})
        self.logger.log({"ObjectLocationAcc": object_location_acc})
        self.logger.log({"ObjectPoseAcc": object_pose_acc})
        self.logger.log({"ObjectPoseError": object_pose_err})
        self.logger.log({"ObjectAttrSim": object_attr_sim})



class HouseJsonSimilarity:
    def __init__(self, args):
        self.sim_scores = []
        self.house_jsons = []
        self.gt_images = []

        self.exp_folder = os.path.join("checkpoints", args['exp_name'])

    def update(self, output, gt):
        # compute text sim
        # convert gt program text to gen config
        gt_house_text = gt['text_labels'][0]
        gt_house_text = gt_house_text.split(": \n")[-1]
        gt_house_text = gt_house_text.replace("(", "[")
        gt_house_text = gt_house_text.replace(")", "]")
        # pdb.set_trace()

        try:
            gt_house_dict = yaml.load(gt_house_text, Loader=yaml.FullLoader)
        except:
            gt_house_dict = {}

        if 'objs_present' in gt:
            gt_house_dict['objs_present'] = gt['objs_present']
        
        if 'camera_pos' in gt:
            gt_house_dict['camera_pos'] = gt['camera_pos'].tolist()

        if 'polygon' in gt:
            gt_house_dict['polygon'] = gt['polygon'].tolist()
            

        if 'house_json' in gt:
            self.house_jsons.append((output, gt['house_json'][0], gt_house_dict, gt['text_labels'][0]))    
        else:
            self.house_jsons.append((output, 0, {}, gt['text_labels'][0]))
        
        if 'image_lists' in gt:
            self.gt_images.append(gt['image_lists'][0])

    def compute(self):
        # pdb.set_trace()
        # save the json for now in a file
        with open(os.path.join(self.exp_folder, 'output.json'), 'w') as f:
            json.dump(self.house_jsons, f)
        
        # save the gt images one by one in vis
        os.makedirs(os.path.join(self.exp_folder, "vis"), exist_ok=True)
        
        for i, image_list in enumerate(self.gt_images):
            images = [Image.open(image_path) for image_path in image_list]

            for corner_ind, image in enumerate(images):
                image.save(os.path.join(self.exp_folder, f"vis/gt_images_{corner_ind}_{i}.png"))



class QAAccuracy:
    def __init__(self, args):
        self.outputs = []
        self.accs = []
        self.logger = args['logger']
        self.exp_folder = os.path.join("checkpoints", args['exp_name'])
        self.table = wandb.Table(columns=["DataName", "Question", "GT Answer", "Pred Answer", "Correct", "Image"])
        self.log_table = args.get('log_table', False)
    
    def update(self, output, gt):
        
        data_name = gt['dataset'][0]
        gt_question = gt['prompts'][0]
        gt_answers = gt['answers']
        pred_answers = output[0].strip()

        format_answer = pred_answers.split("###")[0].strip().lower()
        
        if format_answer == "":
            if "###Assistant" in pred_answers:
                try:
                    format_answer = pred_answers.split("###Assistant:")[1].split("###")[0].strip().lower()
                except:
                    format_answer = pred_answers.split("###")[0].strip().lower()
            elif "###Human" in pred_answers:
                try:
                    format_answer = pred_answers.split("###Human:")[1].split("###")[0].strip().lower()
                except:
                    format_answer = pred_answers.split("###Assistant:")[1].split("###")[0].strip().lower()
            else:
                format_answer = pred_answers.split("###")[0].strip().lower().split(".")[0]

        if gt_answers[0] in ["yes", "no"]:
            if format_answer not in ["yes", "no"]:
                try:
                    format_answer = pred_answers.split("###Assistant:")[1].split("###")[0].strip().lower()
                except:
                    format_answer = pred_answers.split("###")[0].strip().lower()
        
        if format_answer == "ight": # a bug in prediction parsing, fix later
            format_answer = "right"
        
        if "ight stand" in format_answer:
            format_answer = format_answer.replace("ight stand", "night stand")

        # pdb.set_trace()
        format_answer_words = [format_answer,]
        # pdb.set_trace()
        gt_answer = gt_answers[0].lower().strip()

        gt_answer = gt_answer.replace("(", "")
        gt_answer = gt_answer.replace(")", "")

        gt_answer_words = [gt_answer,]
        
        if "BLINK" in data_name:
            if "is closer" in gt_answer:
                gt_answer_word = gt_answer.split(" is closer")[0].split(" ")[-1].strip().lower()
                gt_answer_words = [gt_answer_word]

                format_answer_word = format_answer.split(" is closer")[0].split(" ")[-1].strip().lower()
                format_answer_words = [format_answer_word]
                # pdb.set_trace()

        if "which object is closer" in gt_question.lower():
            if 'is closer' in format_answer:
                format_answer_words = [format_answer.split(" is closer")[0].split(" ")[-1].strip().lower(), ]
            elif 'are closer' in format_answer:
                format_answer_words = [format_answer.split(" are closer")[0].split(" ")[-1].strip().lower(), ]
            elif "is situated closer" in format_answer:
                format_answer_words = [format_answer.split(" is situated closer")[0].split(" ")[-1].strip().lower(), ]
            elif "are situated closer" in format_answer:
                format_answer_words = [format_answer.split(" are situated closer")[0].split(" ")[-1].strip().lower(), ]
            else:
                format_answer_words = [format_answer_words[0],]

        if 'is the camera moving' in gt_question.lower():
            if 'moving' in format_answer:
                format_answer_word = format_answer.split("moving ")[1].split(" ")[0].strip().lower()
                if format_answer_word == "clockwise":
                    format_answer_word = "left"
                elif format_answer_word == "counter-clockwise":
                    format_answer_word = "right"
                else:
                    format_answer_word = format_answer_word
                format_answer_words = [format_answer_word]
            else:
                format_answer_words = [format_answer_words[0],]
        
        if 'considering the relative positions' in gt_question.lower():
            if 'is located to' in format_answer:
                if "right" in format_answer:
                    format_answer_words = ["right"]
                else:
                    format_answer_words = ["left"]
            else:
                format_answer_words = [format_answer_words[0],]

        if '(highlighted by' in format_answer_words[0]:
            format_answer_words = [format_answer_words[0].split("(highlighted by")[0].strip().lower()]
        
        if '(marked by' in format_answer_words[0]:
            format_answer_words = [format_answer_words[0].split("(highlighted by")[0].strip().lower()]

        if "1." in format_answer_words[0]:
            format_answer_words = [format_answer_words[0].split("1.")[1].strip().lower()]

        if "is located" in format_answer_words[0]:
            format_answer_words = [format_answer_words[0].split("is located ")[1].split(" ")[0].strip().lower()]
        
        if "is located on the" in format_answer_words[0]:
            format_answer_words = [format_answer_words[0].split("is located on the ")[1].split(" ")[0].strip().lower()]

        if "answer:" in format_answer_words[0]:
            format_answer_words = [format_answer_words[0].split("answer: ")[1].split(" ")[0].strip().lower()]

        
        

        correct = 0
        for word in gt_answer_words:
            if word in format_answer_words:
                correct = 1

        self.accs.append(correct)

        print("GT: ", gt_answer_words)
        print("Pred: ", format_answer_words)
        print("Correct: ", correct)

        self.outputs.append((gt_question, gt_answers, pred_answers, data_name, correct))

        if self.log_table:
            if 'considering the relative positions' in gt_question.lower():
                if random.random() < 0.2:
                    self.table.add_data(data_name, gt_question, gt_answers, pred_answers, correct, wandb.Image(gt['images'][0][0]))
            elif 'cvbench' in data_name:
                if random.random() < 0.4:
                    self.table.add_data(data_name, gt_question, gt_answers, pred_answers, correct, wandb.Image(gt['images'][0][0]))
            else:
                if random.random() < 0.1:
                    self.table.add_data(data_name, gt_question, gt_answers, pred_answers, correct, wandb.Image(gt['images'][0][0]))    
            
            
            if data_name == "BLINK_Spatial_Relation":
                self.table.add_data(data_name, gt_question, gt_answers, pred_answers, correct, wandb.Image(gt['images'][0][0]))


        
    def compute(self):
        try:
            with open(os.path.join(self.exp_folder, 'output.json'), 'w') as f:
                json.dump(self.outputs, f)
        except:
            print("Error in saving output json")
            pass

        # acc by data name
        data_accs = {}
        for out_entry, acc in zip(self.outputs, self.accs):
            data_name = out_entry[3]
            if data_name not in data_accs:
                data_accs[data_name] = []
            data_accs[data_name].append(acc)

        for data_name in data_accs:
            acc = np.mean(data_accs[data_name])
            print("Num data points: ", len(data_accs[data_name]))
            self.logger.log({f"{data_name}_acc": acc})
        
        if self.log_table:
            self.logger.log({"SPrel_examples": self.table})


class QAAccuracyBatch:
    def __init__(self, args):
        self.outputs = []
        self.accs = []
        self.logger = args['logger']
        self.exp_folder = os.path.join("checkpoints", args['exp_name'])
        self.table = wandb.Table(columns=["DataName", "Question", "GT Answer", "Pred Answer", "Correct", "Image"])
        self.log_table = args.get('log_table', False)
    
    def update(self, output, gt):
        
        data_name = gt['dataset'][0]
        gt_question = gt['prompts'][0]
        gt_answers = gt['answers']
        pred_answers = output

        format_answer = pred_answers.split("###")[0].strip().lower()
        
        if format_answer == "":
            if "###Human" in pred_answers:
                try:
                    format_answer = pred_answers.split("###Human:")[1].split("###")[0].strip().lower()
                except:
                    format_answer = pred_answers.split("###Assistant:")[1].split("###")[0].strip().lower()
            elif "###Assistant" in pred_answers:
                try:
                    format_answer = pred_answers.split("###Assistant:")[1].split("###")[0].strip().lower()
                except:
                    format_answer = pred_answers.split("###")[0].strip().lower()
            else:
                format_answer = pred_answers.split("###")[0].strip().lower().split(".")[0]

        if gt_answers[0] in ["yes", "no"]:
            if format_answer not in ["yes", "no"]:
                format_answer = pred_answers.split("###Assistant:")[1].split("###")[0].strip().lower()
        
        if format_answer == "ight": # a bug in prediction parsing, fix later
            format_answer = "right"
        
        if "ight stand" == format_answer:
            format_answer = format_answer.replace("ight stand", "night stand")


        # pdb.set_trace()
        format_answer_words = [format_answer,]
        # pdb.set_trace()
        gt_answer = gt_answers[0].lower().strip()

        gt_answer = gt_answer.replace("(", "")
        gt_answer = gt_answer.replace(")", "")

        gt_answer_words = [gt_answer,]
        
        if "BLINK" in data_name:
            if "is closer" in gt_answer:
                gt_answer_word = gt_answer.split(" is closer")[0].split(" ")[-1].strip().lower()
                gt_answer_words = [gt_answer_word]

                format_answer_word = format_answer.split(" is closer")[0].split(" ")[-1].strip().lower()
                format_answer_words = [format_answer_word]
                # pdb.set_trace()

        if "which object is closer" in gt_question.lower():
            if 'is closer' in format_answer:
                format_answer_words = [format_answer.split(" is closer")[0].split(" ")[-1].strip().lower(), format_answer.split("is closer ")[1].split(" ")[0].strip().lower()]
            elif 'are closer' in format_answer:
                format_answer_words = [format_answer.split(" are closer")[0].split(" ")[-1].strip().lower(), format_answer.split("are closer ")[1].split(" ")[0].strip().lower()]
            elif "is situated closer" in format_answer:
                format_answer_words = [format_answer.split(" is situated closer")[0].split(" ")[-1].strip().lower(), format_answer.split("is situated closer ")[1].split(" ")[0].strip().lower()]
            elif "are situated closer" in format_answer:
                format_answer_words = [format_answer.split(" are situated closer")[0].split(" ")[-1].strip().lower(), format_answer.split("are situated closer ")[1].split(" ")[0].strip().lower()]
            else:
                format_answer_words = [format_answer_words[0],]

        if 'is the camera moving' in gt_question.lower():
            if 'moving' in format_answer:
                format_answer_word = format_answer.split("moving ")[1].split(" ")[0].strip().lower()
                if format_answer_word == "clockwise":
                    format_answer_word = "left"
                elif format_answer_word == "counter-clockwise":
                    format_answer_word = "right"
                else:
                    format_answer_word = format_answer_word
                format_answer_words = [format_answer_word]
            else:
                format_answer_words = [format_answer_words[0],]
        
        if 'considering the relative positions' in gt_question.lower():
            if 'is located to' in format_answer:
                if "right" in format_answer:
                    format_answer_words = ["right"]
                else:
                    format_answer_words = ["left"]
            else:
                format_answer_words = [format_answer_words[0],]

        if '(highlighted by' in format_answer_words[0]:
            format_answer_words = [format_answer_words[0].split("(highlighted by")[0].strip().lower()]

        if "1." in format_answer_words[0]:
            format_answer_words = [format_answer_words[0].split("1.")[1].strip().lower()]

        if "is located" in format_answer_words[0]:
            format_answer_words = [format_answer_words[0].split("is located ")[1].split(" ")[0].strip().lower()]
        
        if "is located on the" in format_answer_words[0]:
            format_answer_words = [format_answer_words[0].split("is located on the ")[1].split(" ")[0].strip().lower()]

        if "answer:" in format_answer_words[0]:
            format_answer_words = [format_answer_words[0].split("answer: ")[1].split(" ")[0].strip().lower()]


        correct = 0
        for word in gt_answer_words:
            if word in format_answer_words:
                correct = 1

        self.accs.append(correct)

        print("GT: ", gt_answer_words)
        print("Pred: ", format_answer_words)
        print("Correct: ", correct)

        self.outputs.append((gt_question, gt_answers, pred_answers, data_name, correct))

        if self.log_table:
            if 'considering the relative positions' in gt_question.lower():
                if random.random() < 0.2:
                    self.table.add_data(data_name, gt_question, gt_answers, pred_answers, correct, wandb.Image(gt['images'][0][0]))
            elif 'cvbench' in data_name:
                if random.random() < 0.4:
                    self.table.add_data(data_name, gt_question, gt_answers, pred_answers, correct, wandb.Image(gt['images'][0][0]))
            else:
                if random.random() < 0.1:
                    self.table.add_data(data_name, gt_question, gt_answers, pred_answers, correct, wandb.Image(gt['images'][0][0]))    
            
            
            if data_name == "BLINK_Spatial_Relation":
                self.table.add_data(data_name, gt_question, gt_answers, pred_answers, correct, wandb.Image(gt['images'][0][0]))


        
    def compute(self):
        try:
            with open(os.path.join(self.exp_folder, 'output.json'), 'w') as f:
                json.dump(self.outputs, f)
        except:
            print("Error in saving output json")
            pass

        # acc by data name
        data_accs = {}
        for out_entry, acc in zip(self.outputs, self.accs):
            data_name = out_entry[3]
            if data_name not in data_accs:
                data_accs[data_name] = []
            data_accs[data_name].append(acc)

        for data_name in data_accs:
            acc = np.mean(data_accs[data_name])
            print("Num data points: ", len(data_accs[data_name]))
            self.logger.log({f"{data_name}_acc": acc})
        
        if self.log_table:
            self.logger.log({"SPrel_examples": self.table})




class QA_Accuracy_choice:
    def __init__(self, args):
        self.outputs = []
        self.accs = []
        self.logger = args['logger']
        self.exp_folder = os.path.join("checkpoints", args['exp_name'])
    
    def update(self, output, gt):
        
        data_name = gt['dataset'][0]
        gt_question = gt['prompts'][0]
        gt_answers = gt['answers']
        images = gt['images'][0]
        pred_answers = output

        format_answer = pred_answers.split("###")[0].strip().lower()
        
        if format_answer == "":
            if "###Human" in pred_answers:
                format_answer = pred_answers.split("###Human:")[1].split("###")[0].strip().lower()
            elif "###Assistant" in pred_answers:
                format_answer = pred_answers.split("###Assistant:")[1].split("###")[0].strip().lower()
            else:
                format_answer = pred_answers.split("###")[0].strip().lower()

        # pdb.set_trace()
        gt_answer = gt_answers[0].lower().strip()

        gt_answer = gt_answer.replace("(", "")
        gt_answer = gt_answer.replace(")", "")

        format_answer = format_answer.replace("(", "")
        format_answer = format_answer.replace(")", "")

        format_answer = format_answer.strip().split(" ")[0]

        print("GT: ", gt_answer)
        print("Pred: ", format_answer)
        print("Correct: ", gt_answer == format_answer)

        correct = gt_answer == format_answer

        self.accs.append(correct)

        self.outputs.append((gt_question, gt_answers, pred_answers, data_name))

        
    def compute(self):
        
        with open(os.path.join(self.exp_folder, 'output.json'), 'w') as f:
            json.dump(self.outputs, f)

        # acc by data name
        data_accs = {}
        for out_entry, acc in zip(self.outputs, self.accs):
            data_name = out_entry[-1]
            if data_name not in data_accs:
                data_accs[data_name] = []
            data_accs[data_name].append(acc)

        for data_name in data_accs:
            acc = np.mean(data_accs[data_name])
            print("Num data points: ", len(data_accs[data_name]))
            self.logger.log({f"{data_name}_acc": acc})


class ReasoningAccuracy:
    def __init__(self, args):
        self.outputs = []
        self.accs = []
        self.random_accs = []
        self.logger = args['logger']
        self.exp_folder = os.path.join("checkpoints", args['exp_name'])
    
    def update(self, output, gt):
        
        data_name = gt['dataset'][0]
        gt_question = gt['prompts']
        gt_answers = gt['answers']
        # answer_choices = gt['answer_choices'][0]
        pred_answers = output[0].strip()

        format_answer = pred_answers.split("\n")[0].split("###")[-1].strip().lower()
        gt_answer = gt_answers[0].lower().strip()

        correct = gt_answer in format_answer or format_answer in gt_answer

        #incorrect_answers = [ans for ans in answer_choices if gt_answer not in ans.lower().strip()]

        #for inc_ans in incorrect_answers:
        #    if inc_ans in format_answer:
        #        correct = False
        #        break
        
        if "rotated left and rotated right" in gt_answer:
            if "rotated left and rotated right" in format_answer or "rotated right and rotated left" in format_answer or "did not move" in format_answer:
                correct = True

        print("GT: ", gt_answer)
        print("Pred: ", format_answer)
        print("Correct: ", correct)

        # random_correct = random.choice(answer_choices).lower().strip() == gt_answer
            
        self.accs.append(correct)
        # self.random_accs.append(random_correct)

        self.outputs.append((gt_question, gt_answers, pred_answers, data_name)) #, answer_choices))

        
    def compute(self):
        
        try:
            with open(os.path.join(self.exp_folder, 'output.json'), 'w') as f:
                json.dump(self.outputs, f)
        except:
            print("Error in saving output json")

        # overall acc
        acc = np.mean(self.accs)
        self.logger.log({"overall_acc": acc})

        # random_acc = np.mean(self.random_accs)
        # self.logger.log({"random_acc": random_acc})

        # acc by data name
        '''
        data_accs = {}
        data_random_acc = {}
        for out_entry, acc, ran_acc in zip(self.outputs, self.accs, self.random_accs):
            data_name = out_entry[-1]
            if data_name not in data_accs:
                data_accs[data_name] = []
                data_random_acc[data_name] = []
            data_accs[data_name].append(acc)
            data_random_acc[data_name].append(ran_acc)
        '''
        data_accs = {}
        for out_entry, acc in zip(self.outputs, self.accs):
            data_name = out_entry[-1]
            if data_name not in data_accs:
                data_accs[data_name] = []
            data_accs[data_name].append(acc)

        for data_name in data_accs:
            acc = np.mean(data_accs[data_name])
            self.logger.log({f"{data_name}_acc": acc})


class ReasoningAccuracyVSR:
    def __init__(self, args):
        self.outputs = []
        self.accs = []
        self.random_accs = []
        self.logger = args['logger']
        self.exp_folder = os.path.join("checkpoints", args['exp_name'])
    
    def update(self, output, gt):
        
        data_name = gt['dataset'][0]
        gt_question = gt['prompts']
        gt_answers = gt['answers']
        answer_choices = gt['answer_choices'][0]
        pred_answers = output[0].strip()

        format_answer = pred_answers.split("\n")[0].split("###")[-1].strip().lower()
        gt_answer = gt_answers[0].lower().strip()
        if format_answer == "" or format_answer == "assistant:":
            if "###Assistant" in pred_answers:
                try:
                    format_answer = pred_answers.split("###Assistant:")[1].split("###")[0].strip().lower()
                except:
                    format_answer = pred_answers.split("###")[0].strip().lower()
            else:
                format_answer = pred_answers.split("###")[0].strip().lower().split(".")[0]

        format_answer = format_answer.replace("(", "")
        format_answer = format_answer.replace(")", "")
        gt_answer = gt_answer.replace("(", "")
        gt_answer = gt_answer.replace(")", "")


        correct = gt_answer in format_answer or format_answer in gt_answer

        incorrect_answers = [ans for ans in answer_choices if gt_answer not in ans.lower().strip()]

        for inc_ans in incorrect_answers:
            if inc_ans in format_answer:
                correct = False
                break
        

        print("GT: ", gt_answer)
        print("Pred: ", format_answer)
        print("Correct: ", correct)

        random_correct = random.choice(answer_choices).lower().strip() == gt_answer
            
        self.accs.append(correct)
        self.random_accs.append(random_correct)

        self.outputs.append((gt_question, gt_answers, pred_answers, data_name))

        
    def compute(self):
        
        try:
            with open(os.path.join(self.exp_folder, 'output.json'), 'w') as f:
                json.dump(self.outputs, f)
        except:
            print("Error in saving output json")

        # overall acc
        acc = np.mean(self.accs)
        self.logger.log({"overall_acc": acc})

        random_acc = np.mean(self.random_accs)
        self.logger.log({"random_acc": random_acc})

        # acc by data name
        data_accs = {}
        data_random_acc = {}
        for out_entry, acc, ran_acc in zip(self.outputs, self.accs, self.random_accs):
            data_name = out_entry[-1]
            if data_name not in data_accs:
                data_accs[data_name] = []
                data_random_acc[data_name] = []
            data_accs[data_name].append(acc)
            data_random_acc[data_name].append(ran_acc)

        for data_name in data_accs:
            acc = np.mean(data_accs[data_name])
            self.logger.log({f"{data_name}_acc": acc})

            ran_acc = np.mean(data_random_acc[data_name])
            self.logger.log({f"{data_name}_random_acc": ran_acc})


class ReconQAAccGPT:
    def __init__(self, args):
        self.outputs = []
        self.obj_class_accs = []
        self.obj_loc_accs = []
        self.logger = args['logger']
    
    def update(self, output, gt):
        
        gt_question = gt['prompts']
        gt_answers = gt['answers']
        answer_choices = gt['answer_choices']
        pred_answers = output

        format_answer = pred_answers.strip().lower()
        gt_answer = gt_answers.lower().strip()

        not_present_words = ['not present', 'no']
        
        not_present = False
        for no_word in not_present_words:
            if no_word in format_answer:
                not_present = True

        if not_present:
            self.obj_class_accs.append(0)
        else:
            self.obj_class_accs.append(1)
            format_answer = format_answer.replace('"', '')
            format_answer = format_answer.replace("'", "")
            format_answer = format_answer.replace("(", "[")
            format_answer = format_answer.replace(")", "]")
            if "[" not in format_answer:
                format_answer = "["+format_answer
            if "]" not in format_answer:
                format_answer = format_answer + "]"
            
            # if multiple such pattersn, just take the last one since the answer is often at the end. 
            format_answer = "["+format_answer.split("[")[-1].split("]")[0].strip()+"]"

            
            try:
                format_answer = ast.literal_eval(format_answer)
                gt_answer = ast.literal_eval(gt_answer)
                correct = np.allclose(gt_answer, format_answer, atol=0.1)
            except:
                correct = False
            
            print("GT: ", gt_answer)
            print("Pred: ", format_answer)
            print("Correct: ", correct)

            self.obj_loc_accs.append(correct)
        
    def compute(self):
        
        # overall acc
        obj_class_acc = np.mean(self.obj_class_accs)
        obj_loc_acc = np.mean(self.obj_loc_accs)

        self.logger.log({"reconqa_obj_class_acc": obj_class_acc})
        self.logger.log({"reconqa_obj_loc_acc": obj_loc_acc})


class ReconQAAcc:
    def __init__(self, args):
        self.outputs = []
        self.obj_class_accs = []
        self.obj_loc_accs = []
        self.logger = args['logger']
    
    def update(self, output, gt):
        
        gt_question = gt['prompts'][0]
        gt_answers = gt['answers'][0]
        answer_choices = gt['answer_choices'][0]
        pred_answers = output

        format_answer = pred_answers.strip().lower()
        gt_answer = gt_answers.lower().strip()

        not_present_words = ['not present', 'no']
        
        not_present = False
        for no_word in not_present_words:
            if no_word in format_answer:
                not_present = True

        if not_present:
            self.obj_class_accs.append(0)
        else:
            format_answer = format_answer.replace('"', '')
            format_answer = format_answer.replace("'", "")
            format_answer = format_answer.replace("(", "[")
            format_answer = format_answer.replace(")", "]")
            if "[" not in format_answer:
                format_answer = "["+format_answer
            if "]" not in format_answer:
                format_answer = format_answer + "]"
            
            # if multiple such pattersn, just take the first one 
            format_answer = "["+format_answer.split("[")[1].split("]")[0].strip()+"]"

            try:
                format_answer = ast.literal_eval(format_answer)
                gt_answer = ast.literal_eval(gt_answer)
                correct = np.allclose(gt_answer, format_answer, atol=0.1)
                self.obj_class_accs.append(1)
            except:
                self.obj_class_accs.append(0)
                correct = False
            
            print("GT: ", gt_answer)
            print("Pred: ", format_answer)
            print("Correct: ", correct)

            self.obj_loc_accs.append(correct)
        
    def compute(self):
        
        # overall acc
        obj_class_acc = np.mean(self.obj_class_accs)
        obj_loc_acc = np.mean(self.obj_loc_accs)

        self.logger.log({"reconqa_obj_class_acc": obj_class_acc})
        self.logger.log({"reconqa_obj_loc_acc": obj_loc_acc})

class PreciseAcc:
    def __init__(self, args):
        self.outputs = []
        self.obj_loc_accs = []
        self.logger = args['logger']
        self.obj_locs_distance = []
    
    def update(self, output, gt):
        
        gt_question = gt['prompts'][0]
        gt_answers = gt['answers'][0]
        pred_answers = output[0]

        format_answer = pred_answers.strip().lower()
        gt_answer = gt_answers.lower().strip()

        format_answer = format_answer.replace('"', '')
        format_answer = format_answer.replace("'", "")
        format_answer = format_answer.replace("(", "[")
        format_answer = format_answer.replace(")", "]")
        if "[" not in format_answer:
            format_answer = "["+format_answer
        if "]" not in format_answer:
            format_answer = format_answer + "]"
        
        # if multiple such patterns, just take the first one 
        format_answer = "["+format_answer.split("[")[1].split("]")[0].strip()+"]"
        correct = False

        try:
            format_answer = ast.literal_eval(format_answer)
            gt_answer = ast.literal_eval(gt_answer)
            distance = np.linalg.norm(np.array(gt_answer) - np.array(format_answer))
            if distance < 100:
                correct = True
            self.obj_locs_distance.append(distance)
        except:
            correct = False
        
        print("GT: ", gt_answer)
        print("Pred: ", format_answer)
        print("Distance: ", distance)
        print("Correct: ", correct)

        self.obj_loc_accs.append(correct)
        
    def compute(self):
        
        # overall acc
        obj_loc_acc = np.mean(self.obj_loc_accs)
        obj_loc_dist = np.mean(self.obj_locs_distance)

        self.logger.log({"reconqa_obj_loc_acc": obj_loc_acc})
        self.logger.log({"reconqa_obj_loc_dist": obj_loc_dist})
