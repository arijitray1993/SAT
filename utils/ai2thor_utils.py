from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import json

import yaml
import ast
import pdb
import random
import numpy as np
import torch.nn.functional as F
import ast

def tokenize_polygon(polygon, max_dim_x, max_dim_y, max_dim_z):
    # get the tokens for the x,y,z coordinates of the polygon
    # normalized by max length
    tokens = []
    '''
    for point in polygon:
        token_x = int(point['x']/max_dim_x*256)
        token_y = int(point['y']/max_dim_y*256)
        token_z = int(point['z']/max_dim_z*256)
        tokens.append((token_x, token_y, token_z))
    '''
    for point in polygon:
        
        tokens.append((int(round(point['x'], 2)*100), int(round(point['y'], 2)*100), int(round(point['z'], 2)*100)))
    return tokens

def get_token_from_coordinate(x,y,z, max_dim_x, max_dim_y, max_dim_z):
    # 32 tokens represent 0 to 8. Convert 0-8 for x, y, z to 0-31 tokens

    # get the number from the token
    '''
    token_x = int((x/max_dim_x)*256)
    token_y = int((y/max_dim_y)*256)
    token_z = int((z/max_dim_z)*256)
    '''
    token_x = int(round(x, 2)*100)
    token_y = int(round(y, 2)*100)
    token_z = int(round(z, 2)*100)
    return token_x, token_y, token_z

def get_token_from_rotation(x,y,z):
    # 360 tokens represent 0 to 359. Convert 0-359 for x, y, z to 0-359 tokens

    # get the number from the token
    token_x = int(x)
    token_y = int(y)
    token_z = int(z)

    return token_x, token_y, token_z

def get_coordinate_from_token(token, max_dim):
    # 32 tokens represent 0 to 8. Convert 0-31 to 0-8

    # get the number from the token
    #num = int(token)

    # now get corrdinate
    #coordinate = (num/31)*max_dim
    coordinate = token/100.0

    return coordinate

def get_rotation_from_token(token):
    # 360 tokens represent 0 to 359. Convert 0-359 to 0-359

    # get the number from the token
    num = int(token)

    # now get rotation
    rotation = num

    return rotation

def get_rotation_from_tokens(token_x, token_y, token_z):
    # get the x,y,z coordinates from the tokens
    x = get_rotation_from_token(token_x)
    y = get_rotation_from_token(token_y)
    z = get_rotation_from_token(token_z)

    return {"x": x, "y": y, "z": z} 

def get_xyz_from_tokens(token_x, token_y, token_z, max_dim_x, max_dim_y, max_dim_z):

    # get the x,y,z coordinates from the tokens
    #x = get_coordinate_from_token(token_x, max_dim_x)
    #y = get_coordinate_from_token(token_y, max_dim_y)
    #z = get_coordinate_from_token(token_z, max_dim_z)

    x = token_x/100.0
    y = token_y/100.0
    z = token_z/100.0
    return {"x": x, "y": y, "z": z}

def get_polygon_from_polygon_tokens(polygon_tokens, max_dim_x, max_dim_y, max_dim_z):
    # get the polygon from the tokens
    polygon = []
    for token in polygon_tokens:
        x = get_coordinate_from_token(token[0], max_dim_x)
        y = get_coordinate_from_token(token[1], max_dim_y)
        z = get_coordinate_from_token(token[2], max_dim_z)
        polygon.append({"x": x, "y": y, "z": z})

    return polygon

def Random_Points_in_Polygon(polygon, number):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points

def generate_room_programs_from_house_json(house_json):
    # note that this takes an entire house json (og json from ai2thor), and makes a room program and a json just for that room.

    num_rooms = len(house_json['rooms'])

    room_programs = []
    for i in range(num_rooms):
        room_id = house_json['rooms'][i]['id']
        # get polygon
        polygon = house_json['rooms'][i]['floorPolygon']

        # get max dimensions
        max_dim_x = max([point['x'] for point in polygon])

        # max y_dim from wall height
        max_dim_y = 0
        for wall in house_json['walls']:
            for point in wall['polygon']:
                if point['y'] > max_dim_y:
                    max_dim_y = point['y']
        
        max_dim_z = max([point['z'] for point in polygon])

        # pdb.set_trace()
        # get floor material
        floor_material = house_json['rooms'][i]['floorMaterial']['name']

        # get wall materials
        wall_materials = []
        wall_id_to_programid = {}
        for wall in house_json['walls']:
            if wall['roomId'] == room_id:
                wall_materials.append(wall['material']['name'])
                wall_id_to_programid[wall['id']] = len(wall_materials)-1
        
        # tokenize polygon, floor material, and wall materials
        polygon_tokens = tokenize_polygon(polygon, max_dim_x, max_dim_y, max_dim_z)
        # pdb.set_trace()
        floor_material_token = floor_material
        wall_material_tokens = wall_materials

        # get objects
        shapely_polygon = Polygon([(room_point['x'], room_point['z']) for room_point in polygon])
        objects = []
        for obj in house_json['objects']:
            # use shapely polygon to check if the object is inside the room
            point = Point(obj['position']['x'], obj['position']['z'])
            if shapely_polygon.contains(point):
                objects.append(obj)
        
        # get windows
        windows=[]
        for ind, window in enumerate(house_json['windows']):
            if window['room0'] != room_id:
                continue
            window_token = window['assetId']
            window_position = get_token_from_coordinate(window['assetPosition']['x'], window['assetPosition']['y'], window['assetPosition']['z'], max_dim_x, max_dim_y, max_dim_z)
            window_polygon = tokenize_polygon(window['holePolygon'], max_dim_x, max_dim_y, max_dim_z)
            window_wall = wall_id_to_programid[window['wall0']]

            # find the window wall in program text
            # window_wall_program_id = wall_id_to_programid[window_wall]
            program_window_entry = (window_token, window_position, window_polygon, window_wall)
            windows.append(program_window_entry)
        
        # make a program
        # max_dims: [{max_dim_x}, {max_dim_y}, {max_dim_z}]
        program_text = f"""
polygon: {polygon_tokens}
floor_material: '{floor_material_token}'
wall_material: {wall_material_tokens}
"""
        for ind, obj in enumerate(objects):
            obj_token = obj['assetId']
            obj_position = get_token_from_coordinate(obj['position']['x'], obj['position']['y'], obj['position']['z'], max_dim_x, max_dim_y, max_dim_z)
            obj_rotation = get_token_from_rotation(obj['rotation']['x'], obj['rotation']['y'], obj['rotation']['z'])
            obj_id = obj['id']
            # pdb.set_trace()
            obj_entry_dict = (obj_token, obj_position, obj_rotation)
            program_text += f"\nobj_{ind}: {obj_entry_dict}"
            
            if 'children' in obj:
                for child_ind, child in enumerate(obj['children']):
                    child_token = child['assetId']
                    child_position = get_token_from_coordinate(child['position']['x'], child['position']['y'], child['position']['z'], max_dim_x, max_dim_y, max_dim_z)
                    child_rotation = get_token_from_rotation(child['rotation']['x'], child['rotation']['y'], child['rotation']['z'])
                    child_parent_id = f"obj_{ind}"
                    child_entry_dict = (child_token, child_position, child_rotation, child_parent_id)
                    program_text += f"\nchild_{child_ind}: {child_entry_dict}"

        for ind, window in enumerate(windows):
            program_text += f"\nwindow_{ind}: {window}"

        #### actually execute the program text to get the house json
        program_text = program_text.replace("(", "[")
        program_text = program_text.replace(")", "]")
        room_json = make_house_from_cfg(program_text)
        # pdb.set_trace()
        room_programs.append((program_text, room_json.house_json, house_json))

    return room_programs

def generate_attribute_program_from_roomjson(house_json, include_objects=True, include_windows=True, include_children=False, asset_desc=None):
    # this takes the json of just a one room house and make a program for the room.
    # for each room, we need polygon, floor material, wall materials, and objects at location and rotation
   
    room_id = house_json['rooms'][0]['id']
    # get polygon
    polygon = house_json['rooms'][0]['floorPolygon']

    # get max dimensions
    max_dim_x = max([point['x'] for point in polygon])

    # max y_dim from wall height
    max_dim_y = 0
    for wall in house_json['walls']:
        for point in wall['polygon']:
            if point['y'] > max_dim_y:
                max_dim_y = point['y']
    
    max_dim_z = max([point['z'] for point in polygon])

    # pdb.set_trace()
    # get floor material
    floor_material = house_json['rooms'][0]['floorMaterial']['name']
    
    # tokenize polygon, floor material, and wall materials
    polygon_tokens = tokenize_polygon(polygon, max_dim_x, max_dim_y, max_dim_z)
    # pdb.set_trace()
    floor_material_token = floor_material

    wall_materials = []
    for wall in house_json['walls']:
        if wall['roomId'] == room_id:
            wall_materials.append(wall['material']['name'])
    wall_material_tokens = wall_materials

    # get objects
    shapely_polygon = Polygon([(room_point['x'], room_point['z']) for room_point in polygon])
    objects = []
    for obj in house_json['objects']:
        # use shapely polygon to check if the object is inside the room
        point = Point(obj['position']['x'], obj['position']['z'])
        if shapely_polygon.contains(point):
            objects.append(obj)
    
    # get windows
    windows=[]
    for ind, window in enumerate(house_json['windows']):
        window_token = window['assetId']
        window_position = get_token_from_coordinate(window['assetPosition']['x'], window['assetPosition']['y'], window['assetPosition']['z'], max_dim_x, max_dim_y, max_dim_z)
        window_polygon = tokenize_polygon(window['holePolygon'], max_dim_x, max_dim_y, max_dim_z)
        window_wall = window['wall0']

        # find the window wall in program text
        # window_wall_program_id = wall_id_to_programid[window_wall]
        program_window_entry = (window_token, window_position, window_polygon, window_wall)
        windows.append(program_window_entry)
    
    # make a program
    # max_dims: [{max_dim_x}, {max_dim_y}, {max_dim_z}]
    program_text = f"""
polygon: {polygon_tokens}
floor_material: '{floor_material_token}'
wall_material: {wall_material_tokens}
"""
    if include_objects:
        for ind, obj in enumerate(objects):
            obj_token = obj['assetId']
            obj_desc = random.choice(asset_desc[obj_token])[2]
            obj_position = get_token_from_coordinate(obj['position']['x'], obj['position']['y'], obj['position']['z'], max_dim_x, max_dim_y, max_dim_z)
            obj_rotation = get_token_from_rotation(obj['rotation']['x'], obj['rotation']['y'], obj['rotation']['z'])
            obj_id = obj['id']
            # pdb.set_trace()
            obj_entry_dict = f"{obj_desc} at location {obj_position} with rotation {obj_rotation}"
            program_text += f"\nobj_{ind}: {obj_entry_dict}"

            if 'children' in obj and include_children:
                for child_ind, child in enumerate(obj['children']):
                    child_token = child['assetId']
                    child_position = get_token_from_coordinate(child['position']['x'], child['position']['y'], child['position']['z'], max_dim_x, max_dim_y, max_dim_z)
                    child_rotation = get_token_from_rotation(child['rotation']['x'], child['rotation']['y'], child['rotation']['z'])
                    child_parent_id = f"obj_{ind}"
                    child_entry_dict = (child_token, child_position, child_rotation, child_parent_id)
                    program_text += f"\nchild_{child_ind}: {child_entry_dict}"
    
    if include_windows:
        for ind, window in enumerate(windows):
            program_text += f"\nwindow_{ind}: {window}"

    #### actually execute the program text to get the house json
    program_text = program_text.replace("(", "[")
    program_text = program_text.replace(")", "]")
    
    return program_text




def generate_program_from_roomjson(house_json, include_objects=True, include_windows=True, include_children=True):
    # this takes the json of just a one room house and make a program for the room.
    # for each room, we need polygon, floor material, wall materials, and objects at location and rotation
   
    room_id = house_json['rooms'][0]['id']
    # get polygon
    polygon = house_json['rooms'][0]['floorPolygon']

    # get max dimensions
    max_dim_x = max([point['x'] for point in polygon])

    # max y_dim from wall height
    max_dim_y = 0
    for wall in house_json['walls']:
        for point in wall['polygon']:
            if point['y'] > max_dim_y:
                max_dim_y = point['y']
    
    max_dim_z = max([point['z'] for point in polygon])

    # pdb.set_trace()
    # get floor material
    floor_material = house_json['rooms'][0]['floorMaterial']['name']
    
    # tokenize polygon, floor material, and wall materials
    polygon_tokens = tokenize_polygon(polygon, max_dim_x, max_dim_y, max_dim_z)
    # pdb.set_trace()
    floor_material_token = floor_material

    wall_materials = []
    for wall in house_json['walls']:
        if wall['roomId'] == room_id:
            wall_materials.append(wall['material']['name'])
    wall_material_tokens = wall_materials

    # get objects
    shapely_polygon = Polygon([(room_point['x'], room_point['z']) for room_point in polygon])
    objects = []
    for obj in house_json['objects']:
        # use shapely polygon to check if the object is inside the room
        point = Point(obj['position']['x'], obj['position']['z'])
        if shapely_polygon.contains(point):
            objects.append(obj)
    
    # get windows
    windows=[]
    for ind, window in enumerate(house_json['windows']):
        window_token = window['assetId']
        window_position = get_token_from_coordinate(window['assetPosition']['x'], window['assetPosition']['y'], window['assetPosition']['z'], max_dim_x, max_dim_y, max_dim_z)
        window_polygon = tokenize_polygon(window['holePolygon'], max_dim_x, max_dim_y, max_dim_z)
        window_wall = window['wall0']

        # find the window wall in program text
        # window_wall_program_id = wall_id_to_programid[window_wall]
        program_window_entry = (window_token, window_position, window_polygon, window_wall)
        windows.append(program_window_entry)
    
    # make a program
    # max_dims: [{max_dim_x}, {max_dim_y}, {max_dim_z}]
    program_text = f"""
polygon: {polygon_tokens}
floor_material: '{floor_material_token}'
wall_material: {wall_material_tokens}
"""
    if include_objects:
        for ind, obj in enumerate(objects):
            obj_token = obj['assetId']
            obj_position = get_token_from_coordinate(obj['position']['x'], obj['position']['y'], obj['position']['z'], max_dim_x, max_dim_y, max_dim_z)
            obj_rotation = get_token_from_rotation(obj['rotation']['x'], obj['rotation']['y'], obj['rotation']['z'])
            obj_id = obj['id']
            # pdb.set_trace()
            obj_entry_dict = (obj_token, obj_position, obj_rotation)
            program_text += f"\nobj_{ind}: {obj_entry_dict}"

            if 'children' in obj and include_children:
                for child_ind, child in enumerate(obj['children']):
                    child_token = child['assetId']
                    child_position = get_token_from_coordinate(child['position']['x'], child['position']['y'], child['position']['z'], max_dim_x, max_dim_y, max_dim_z)
                    child_rotation = get_token_from_rotation(child['rotation']['x'], child['rotation']['y'], child['rotation']['z'])
                    child_parent_id = f"obj_{ind}"
                    child_entry_dict = (child_token, child_position, child_rotation, child_parent_id)
                    program_text += f"\nchild_{child_ind}: {child_entry_dict}"
    
    if include_windows:
        for ind, window in enumerate(windows):
            program_text += f"\nwindow_{ind}: {window}"

    #### actually execute the program text to get the house json
    program_text = program_text.replace("(", "[")
    program_text = program_text.replace(")", "]")
    
    return program_text


def generate_program_from_roomjson_holodeckeval(house_json, include_objects=True, include_windows=True, include_children=True):
    # this takes the json of just a one room house and make a program for the room.
    # for each room, we need polygon, floor material, wall materials, and objects at location and rotation
   
    room_id = house_json['rooms'][0]['id']
    # get polygon
    polygon = house_json['rooms'][0]['floorPolygon']

    # get max dimensions
    max_dim_x = max([point['x'] for point in polygon])

    # max y_dim from wall height
    max_dim_y = 0
    for wall in house_json['walls']:
        for point in wall['polygon']:
            if point['y'] > max_dim_y:
                max_dim_y = point['y']
    
    max_dim_z = max([point['z'] for point in polygon])

    # pdb.set_trace()
    # get floor material
    floor_material = house_json['rooms'][0]['floorMaterial']['name']
    
    # tokenize polygon, floor material, and wall materials
    polygon_tokens = tokenize_polygon(polygon, max_dim_x, max_dim_y, max_dim_z)
    # pdb.set_trace()
    floor_material_token = floor_material

    wall_materials = []
    for wall in house_json['walls']:
        if wall['roomId'] == room_id:
            wall_materials.append(wall['material']['name'])
    wall_material_tokens = wall_materials

    # get objects
    shapely_polygon = Polygon([(room_point['x'], room_point['z']) for room_point in polygon])
    objects = []
    for obj in house_json['objects']:
        # use shapely polygon to check if the object is inside the room
        point = Point(obj['position']['x'], obj['position']['z'])
        if shapely_polygon.contains(point):
            objects.append(obj)
    
    # get windows
    windows=[]
    for ind, window in enumerate(house_json['windows']):
        window_token = window['assetId']
        window_position = get_token_from_coordinate(window['assetPosition']['x'], window['assetPosition']['y'], window['assetPosition']['z'], max_dim_x, max_dim_y, max_dim_z)
        window_polygon = tokenize_polygon(window['holePolygon'], max_dim_x, max_dim_y, max_dim_z)
        window_wall = window['wall0']

        # find the window wall in program text
        # window_wall_program_id = wall_id_to_programid[window_wall]
        program_window_entry = (window_token, window_position, window_polygon, window_wall)
        windows.append(program_window_entry)
    
    # make a program
    # max_dims: [{max_dim_x}, {max_dim_y}, {max_dim_z}]
    program_text = f"""
polygon: {polygon_tokens}
floor_material: '{floor_material_token}'
wall_material: {wall_material_tokens}
"""
    if include_objects:
        for ind, obj in enumerate(objects):
            obj_token = obj['assetId']
            obj_name = obj.get('object_name')
            if obj_name is None:
                obj_name = obj['id']
            
            obj_name = obj_name.split("|")[0]
            obj_position = get_token_from_coordinate(obj['position']['x'], obj['position']['y'], obj['position']['z'], max_dim_x, max_dim_y, max_dim_z)
            obj_rotation = get_token_from_rotation(obj['rotation']['x'], obj['rotation']['y'], obj['rotation']['z'])
            obj_id = obj['id']
            # pdb.set_trace()
            obj_entry_dict = (obj_name, obj_position, obj_rotation)
            program_text += f"\nobj_{ind}: {obj_entry_dict}"

            if 'children' in obj and include_children:
                for child_ind, child in enumerate(obj['children']):
                    child_token = child['assetId']
                    child_name = child.get('object_name')
                    if child_name is None:
                        child_name = child['id']
                    
                    child_name = child_name.split("|")[0]
                    child_position = get_token_from_coordinate(child['position']['x'], child['position']['y'], child['position']['z'], max_dim_x, max_dim_y, max_dim_z)
                    child_rotation = get_token_from_rotation(child['rotation']['x'], child['rotation']['y'], child['rotation']['z'])
                    child_parent_id = f"obj_{ind}"
                    child_entry_dict = (child_name, child_position, child_rotation, child_parent_id)
                    program_text += f"\nchild_{child_ind}: {child_entry_dict}"
    
    if include_windows:
        for ind, window in enumerate(windows):
            program_text += f"\nwindow_{ind}: {window}"

    #### actually execute the program text to get the house json
    program_text = program_text.replace("(", "[")
    program_text = program_text.replace(")", "]")
    
    return program_text


def generate_program_from_polygon_objects(polygon, floor_material_token, wall_material_token, objects, include_windows=False):
    
    polygon_tokens = tokenize_polygon(polygon, 8, 8, 8)

    #floor_material_token= 'WoodFineDarkFloorsRedNRM1'
    #wall_material_tokens = ['PureWhite']*len(polygon_tokens)

    # make a program
    program_text = f"""
polygon: {polygon_tokens}
"""

    for ind, obj in enumerate(objects):
        obj_token = obj['assetId']
        obj_position = get_token_from_coordinate(obj['position']['x'], obj['position']['y'], obj['position']['z'], 8, 8, 8)
        obj_rotation = get_token_from_rotation(obj['rotation']['x'], obj['rotation']['y'], obj['rotation']['z'])
        obj_id = obj['id']
        # pdb.set_trace()
        obj_entry_dict = (obj_token, obj_position, obj_rotation)
        program_text += f"\nobj_{ind}: {obj_entry_dict}"
    
    if include_windows:
        for ind, window in enumerate(windows):
            program_text += f"\nwindow_{ind}: {window}"

    #### actually execute the program text to get the house json
    program_text = program_text.replace("(", "[")
    program_text = program_text.replace(")", "]")
    
    return program_text





def format_program(program_text):

    # remove the child from program for now since it doesn't work
    program_text = program_text.split("\n")
    new_program_text = []
    for line in program_text:
        if "child" in line:
            continue
        new_program_text.append(line)
    
    program_text = "\n".join(new_program_text)

    return program_text

class House:

    def __init__(self, args={}, house_json="", max_dim_x=8, max_dim_y=8, max_dim_z=8):
        if house_json == "":
            self.house_json = self.generate_house_template()
        else:
            self.house_json = house_json
        
        self.max_dim_x = max_dim_x
        self.max_dim_y = max_dim_y
        self.max_dim_z = max_dim_z
        #self.object_choices = json.load(open("all_objects.json", "r"))
        #self.wall_material_choices = json.load(open("all_wall_materials.json", "r"))
        #self.room_type_choices = json.load(open("all_room_types.json", "r"))
        #self.floor_material_choices = json.load(open("all_floor_materials.json", "r"))

    # utils
    def get_object_from_token(self, token):
        # get the object from the token
        obj = self.object_choices[int(token.split("_")[1])]
        return obj

    def get_wall_material_from_token(self, token):
        # get the wall material from the token
        wall_material = self.wall_material_choices[int(token.split("_")[1])]
        return wall_material

    def get_floor_material_from_token(self, token):
        # get the floor material from the token
        floor_material = self.floor_material_choices[int(token.split("_")[1])]
        return floor_material
    
    def get_token_from_object(self, obj):
        # get the token from the object
        token = f"object_{self.object_choices.index(obj)}"
        return token

    def get_token_from_wall_material(self, wall_material):
        token = f"wallmaterial_{self.wall_material_choices.index(wall_material)}"
        return token

    def get_token_from_floor_material(self, floor_material):
        token = f"floormaterial_{self.floor_material_choices.index(floor_material)}"
        return token

    # house generation functions
    def generate_house_template(self,):
        # generate emtpy house json structure
        house_json = {
            "doors": [],
            "metadata": {
                "agent": {
                    "horizon": 30,
                    "position": {
                        "x": 3.5,
                        "y": 0.95,
                        "z": 2
                    },
                    "rotation": {
                        "x": 0,
                        "y": 90,
                        "z": 0
                    },
                    "standing": True
                },
                "roomSpecId": "kitchen",
                "schema": "1.0.0",
                "warnings": {},
                "agentPoses": {
                    "arm": {
                        "horizon": 30,
                        "position": {
                        "x": 3.5,
                        "y": 0.95,
                        "z": 2
                        },
                        "rotation": {
                        "x": 0,
                        "y": 90,
                        "z": 0
                        },
                        "standing": True
                    },
                    "default": {
                        "horizon": 30,
                        "position": {
                        "x": 3.5,
                        "y": 0.95,
                        "z": 2
                        },
                        "rotation": {
                        "x": 0,
                        "y": 90,
                        "z": 0
                        },
                        "standing": True
                    },
                    "locobot": {
                        "horizon": 30,
                        "position": {
                        "x": 3.5,
                        "y": 0.95,
                        "z": 2
                        },
                        "rotation": {
                        "x": 0,
                        "y": 90,
                        "z": 0
                        }
                    },
                    "stretch": {
                        "horizon": 30,
                        "position": {
                        "x": 3.5,
                        "y": 0.95,
                        "z": 2
                        },
                        "rotation": {
                        "x": 0,
                        "y": 90,
                        "z": 0
                        },
                        "standing": True
                    }
                }
            },
            "objects": [],
            "proceduralParameters": {
                "ceilingColor": {
                "b": 0.3058823529411765,
                "g": 0.3843137254901961,
                "r": 0.42745098039215684
                },
                "ceilingMaterial": {
                "name": "PureWhite",
                "color": {
                    "b": 0.3058823529411765,
                    "g": 0.3843137254901961,
                    "r": 0.42745098039215684
                }
                },
                "floorColliderThickness": 1,
                "lights": [
                {
                    "id": "DirectionalLight",
                    "indirectMultiplier": 1,
                    "intensity": 1,
                    "position": {
                    "x": 0.84,
                    "y": 0.1855,
                    "z": -1.09
                    },
                    "rgb": {
                    "r": 1,
                    "g": 1,
                    "b": 1
                    },
                    "rotation": {
                    "x": 66,
                    "y": 75,
                    "z": 0
                    },
                    "shadow": {
                    "type": "Soft",
                    "strength": 1,
                    "normalBias": 0,
                    "bias": 0,
                    "nearPlane": 0.2,
                    "resolution": "FromQualitySettings"
                    },
                    "type": "directional"
                },
                {
                    "id": "light_2",
                    "intensity": 0.85,
                    "position": {
                    "x": 3.9899999999999998,
                    "y": 4.351928794929904,
                    "z": 2.49375
                    },
                    "range": 15,
                    "rgb": {
                    "r": 1,
                    "g": 0.855,
                    "b": 0.722
                    },
                    "shadow": {
                    "type": "Soft",
                    "strength": 1,
                    "normalBias": 0,
                    "bias": 0.05,
                    "nearPlane": 0.2,
                    "resolution": "FromQualitySettings"
                    },
                    "type": "point"
                }
                ],
                "receptacleHeight": 0.7,
                "reflections": [],
                "skyboxId": "SkyAlbany"
            },
            "rooms": [
                {
                "ceilings": [],
                "children": [],
                "floorMaterial": {
                    "name": ""
                },
                "floorPolygon": [                    
                ],
                "id": "",
                "roomType": ""
                }
            ],
            "walls": [],
            "windows": []
        }

        return house_json

    def make_floorplan_walls(self, args):
        '''
        expects keys:
            - polygon: list of x,y,z coordinates that join to make a polygon
            - floor_material
            - wall_material: list of wall materials for each wall (number of edges in polygon)
        '''
        polygon = args['polygon']
        floor_material = args['floor_material']

        polygon = [get_xyz_from_tokens(token_x, token_y, token_z, self.max_dim_x, self.max_dim_y, self.max_dim_z) for token_x, token_y, token_z in polygon]
        
        self.house_json['rooms'][0]['floorPolygon'] = polygon
        self.house_json['rooms'][0]['floorMaterial']['name'] = floor_material
        self.house_json['rooms'][0]['id'] = "room_0"

        # make walls from polygon
        walls = []
        for i in range(len(polygon)):
            wall_height_point_1 = polygon[i].copy()
            wall_height_point_1['y'] = self.max_dim_y
            wall_height_point_2 = polygon[(i+1)%len(polygon)].copy()
            wall_height_point_2['y'] = self.max_dim_y

            wall = {
                "id": i,
                "color": {
                    "r": 1,
                    "g": 1,
                    "b": 1,
                    "a": 1
                },
                "material": {
                    "name": args['wall_material'][i],
                    "color": {
                        "r": 1,
                        "g": 1,
                        "b": 1,
                        "a": 1
                    },
                },
                "roomId": "room_0",
                "polygon": [
                    polygon[i],
                    polygon[(i+1)%len(polygon)],
                    wall_height_point_1,
                    wall_height_point_2
                ]
            }
            walls.append(wall)
            wall_exterior = wall.copy()
            wall_exterior['roomId'] = "exterior"
            wall_exterior['id'] = f"exterior_{i}"
            walls.append(wall_exterior)

        self.house_json['walls'] = walls

        # fix the agent position based on the floor ploygon to make sure it is inside the room
        room_polygon = Polygon([(point['x'], point['z']) for point in polygon])
        reachable_positions = Random_Points_in_Polygon(room_polygon, 1)

        self.house_json['metadata']['agent']['position']['x'] = reachable_positions[0].x
        self.house_json['metadata']['agent']['position']['z'] = reachable_positions[0].y
        for entry in self.house_json['metadata']['agentPoses']:
            self.house_json['metadata']['agentPoses'][entry]['position']['x'] = reachable_positions[0].x
            self.house_json['metadata']['agentPoses'][entry]['position']['z'] = reachable_positions[0].y

        # fix the light position based on the floor ploygon to make sure it is inside the room
        self.house_json['proceduralParameters']['lights'][1]['position']['x'] = reachable_positions[0].x
        self.house_json['proceduralParameters']['lights'][1]['position']['y'] = self.max_dim_y - 0.3
        self.house_json['proceduralParameters']['lights'][1]['position']['z'] = reachable_positions[0].y

    def add_window(self, args):

        window = {
            'assetId': args['assetId'],
            'assetPosition': get_xyz_from_tokens(*args['position'], self.max_dim_x, self.max_dim_y, self.max_dim_z),
            'holePolygon': get_polygon_from_polygon_tokens(args['windowPolygon'], self.max_dim_x, self.max_dim_y, self.max_dim_z),
            'id': args['id'],
            'room0': "room_0",
            'room1': "room_0",
            'wall0': args['wall'],
            'wall1': args['wall_exterior']
        }
        self.house_json['windows'].append(window)

    def add_object(self, args):
        '''
        expects keys:
        - assetId: id of the object, used to identify the object from a list of possible choices.
        - position: x,y,z coordinates of the object
        - rotation: x,y,z angles 0-360 - what does this exactly mean? not sure
        '''
        obj = {
            "assetId": args['assetId'],
            "position": get_xyz_from_tokens(*args['position'], self.max_dim_x, self.max_dim_y, self.max_dim_z),
            "rotation": get_rotation_from_tokens(*args['rotation']),
            "id": args['id'],
            "kinematic": True,
            "layer": '',
            "material": None,
            "children": []
        }
        self.house_json['objects'].append(obj)

    def add_object_children(self, args):
        '''
        expects keys:
        - children: list of objects that are children of other objects
            each children object should have:
            - parent_id: id of the parent object. we will attach the child object to this parent object
            - assetId: id of the object, used to identify the object from a list of possible choices.
            - position: x,y,z coordinates of the object
            - rotation: x,y,z angles 0-360 - what does this exactly mean? not sure
        '''
        child = args
        child_obj = {
            "assetId": child['assetId'],
            "position": get_xyz_from_tokens(*child['position'], self.max_dim_x, self.max_dim_y, self.max_dim_z),
            "rotation": get_rotation_from_tokens(*child['rotation']),
            "id": child['id'],
            "parent_id": child['parent_id']
        }
        
        # search for the house object with the parent id
        for obj in self.house_json['objects']:
            if obj['id'] == child_obj['parent_id']:
                if 'children' not in obj:
                    obj['children'] = []

                child_obj_to_add = child_obj.copy()
                del child_obj_to_add['parent_id']
                obj['children'].append(child_obj_to_add)
                break
    
def make_house_from_cfg(cfg):
    '''
    cfg is a string like:
    polygon: [[358, 0, 358], [358, 0, 894], [894, 0, 894], [894, 0, 0], [536, 0, 0], [536, 0, 358]]
    floor_material: 'WoodFineDarkFloorsRedNRM1'
    wall_material: ['PureWhite', 'PureWhite', 'PureWhite', 'PureWhite', 'PureWhite', 'PureWhite'] # the wall ids are in order of the polygon points 0 to n.

    obj_0: ['TV_Stand_206_3', [861, 38, 63], [0, 270, 0]] # asset id, position, rotation
    obj_1: ['Cart_1', [636, 64, 666], [0, 0, 0]]
    obj_2: ['Dining_Table_221_1', [852, 39, 802], [0, 270, 0]]
    obj_3: ['RoboTHOR_sofa_vreta', [579, 36, 91], [0, 90, 0]]
    obj_4: ['Wall_Decor_Photo_1V', [892, 161, 863], [0, 270, 0]]
    obj_5: ['Wall_Decor_Photo_1', [892, 191, 137], [0, 270, 0]]
    window_0: ['Window_Hung_48x24', [101, 150, 0], [[40, 117, 0], [162, 183, 0]], 5] # assetid, position, window polygon, wallid

    outputs a room (not a house really) json that can be used with the Controller in AI2thor. 
    '''

    # get dictionary from yaml-like string
    cfg_dict = yaml.load(cfg, Loader=yaml.FullLoader)

    # get max dimensions
    floor_polygon = cfg_dict['polygon']
    floor_material = cfg_dict['floor_material']
    wall_material = cfg_dict['wall_material']

    # make house
    house = House()
    house.make_floorplan_walls({
        'polygon': floor_polygon,
        'floor_material': floor_material,
        'wall_material': wall_material
    })

    # make windows
    i = 0
    while(True):
        if f'window_{i}' in cfg_dict:
            window = cfg_dict[f'window_{i}']
            house.add_window({
                'assetId': window[0],
                'position': window[1],
                'windowPolygon': window[2],
                'id': f'window_{i}',
                'wall': window[3],
                'wall_exterior': f'exterior_{window[3]}'
            })
            i += 1
        else:
            break
    # print(f"made {i} windows")

    # get obj0, obj1, ... until we run out of objects in dict
    i = 0
    while(True):
        if f'obj_{i}' in cfg_dict:
            obj = cfg_dict[f'obj_{i}']
            # print(obj[1])
            house.add_object({
                'assetId': obj[0],
                'position': list(obj[1]),
                'rotation': list(obj[2]),
                'id': f'obj_{i}'
            })
            i += 1
        else:
            break
    # print(f"made {i} objects")

    # get child0, child1, ... until we run out of children in dict
    i = 0
    while(True):
        if f'child_{i}' in cfg_dict:
            child = cfg_dict[f'child_{i}']
            house.add_object_children({
                'assetId': child[0],
                'position': list(child[1]),
                'rotation': list(child[2]),
                'id': f'child_{i}',
                'parent_id': child[3]
            })
            i += 1
        else:
            break

    return house

def get_obj_from_desc_pred(obj_desc, known_desc_to_class, attrmodel, attrtokenizer):
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

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def make_house_from_cfg_language(cfg, attrmodel, attrtokenizer, known_desc_to_class):
    '''
    cfg is a string like:
    polygon: [[358, 0, 358], [358, 0, 894], [894, 0, 894], [894, 0, 0], [536, 0, 0], [536, 0, 358]]
    floor_material: 'WoodFineDarkFloorsRedNRM1'
    wall_material: ['PureWhite', 'PureWhite', 'PureWhite', 'PureWhite', 'PureWhite', 'PureWhite'] # the wall ids are in order of the polygon points 0 to n.

    obj_0: ['TV_Stand_206_3', [861, 38, 63], [0, 270, 0]] # asset id, position, rotation
    obj_1: ['Cart_1', [636, 64, 666], [0, 0, 0]]
    obj_2: ['Dining_Table_221_1', [852, 39, 802], [0, 270, 0]]
    obj_3: ['RoboTHOR_sofa_vreta', [579, 36, 91], [0, 90, 0]]
    obj_4: ['Wall_Decor_Photo_1V', [892, 161, 863], [0, 270, 0]]
    obj_5: ['Wall_Decor_Photo_1', [892, 191, 137], [0, 270, 0]]
    window_0: ['Window_Hung_48x24', [101, 150, 0], [[40, 117, 0], [162, 183, 0]], 5] # assetid, position, window polygon, wallid

    outputs a room (not a house really) json that can be used with the Controller in AI2thor. 
    '''

    # get dictionary from yaml-like string
    cfg_dict = yaml.load(cfg, Loader=yaml.FullLoader)

    # get max dimensions
    floor_polygon = cfg_dict['polygon']
    floor_material = cfg_dict['floor_material']
    wall_material = cfg_dict['wall_material']

    # make house
    house = House()
    house.make_floorplan_walls({
        'polygon': floor_polygon,
        'floor_material': floor_material,
        'wall_material': wall_material
    })

    # make windows
    i = 0
    while(True):
        if f'window_{i}' in cfg_dict:
            window = cfg_dict[f'window_{i}']
            house.add_window({
                'assetId': window[0],
                'position': window[1],
                'windowPolygon': window[2],
                'id': f'window_{i}',
                'wall': window[3],
                'wall_exterior': f'exterior_{window[3]}'
            })
            i += 1
        else:
            break
    # print(f"made {i} windows")

    # get obj0, obj1, ... until we run out of objects in dict
    i = 0
    while(True):
        if f'obj_{i}' in cfg_dict:
            obj = cfg_dict[f'obj_{i}']
            
            obj_desc = obj.split(" at location ")[0].strip()
            obj_position = obj.split(" at location ")[1].split(" with rotation ")[0].strip()
            obj_rotation = obj.split(" with rotation ")[1].strip()
            
            #retrieve closest assetid
            obj_assetid = get_obj_from_desc_pred(obj_desc, known_desc_to_class, attrmodel, attrtokenizer)[0][1]

            house.add_object({
                'assetId': obj_assetid,
                'position': ast.literal_eval(obj_position),
                'rotation': ast.literal_eval(obj_rotation),
                'id': f'obj_{i}'
            })
            i += 1
        else:
            break
    # print(f"made {i} objects")

    return house