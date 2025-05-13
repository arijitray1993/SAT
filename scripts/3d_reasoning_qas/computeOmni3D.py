from collections import defaultdict
import json
from PIL import Image, ImageDraw
import os
import pdb
import random


def add_box_dot_with_color(image, box, color):
    # box coordinate is in x1, y1, x2, y2 format

    draw = ImageDraw.Draw(image)

    x1, y1, x2, y2 = box

    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    return image


if __name__=="__main__":
    
    data = json.load(open("/projectnb/ivc-ml/array/data/omni3d/datasets/Omni3D/ARKitScenes_val.json"))
    im_path = "/projectnb/ivc-ml/array/data/omni3d/datasets/"

    # data has keys 'info', 'images', 'categories', 'annotations'
    # table = wandb.Table(columns=["Image", "object", "2DBBox", "3DCenter"])

    imid2image_file = {}
    for image in data['images']:
        if image['src_flagged']:
            continue
        
        imid2image_file[image['id']] = (image['file_path'], image['width'], image['height'], image['K'])
    
    img2ann = defaultdict(list)
    for annotation in data['annotations']:
        # has keys:
        # 'behind_camera', 'truncation', 'bbox2D_tight', 'visibility', 'segmentation_pts', 'lidar_pts', 
        # 'valid3D', 'category_id', 'category_name', 'id', 'image_id', 'dataset_id', 'bbox2D_proj', 
        # 'depth_error', 'center_cam', 'dimensions', 'bbox3D_cam', 'R_cam', 'bbox2D_trunc'

        imid = annotation['image_id']
        if imid not in imid2image_file:
            continue

        if not annotation['valid3D']:
            continue

        depth_error = annotation['depth_error']
        truncation = annotation['truncation']

        if depth_error > 0.6 or truncation > 0.6:
            continue
        
        behind_camera = annotation['behind_camera']
        if behind_camera:
            continue

        image_file = os.path.join(im_path, imid2image_file[annotation['image_id']][0])

        cam_k = imid2image_file[annotation['image_id']][3]
        # [f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]
        

        center_3d = annotation['center_cam']

        bbox_2d = annotation['bbox2D_trunc']

        object_name = annotation['category_name']

        # image = Image.open(image_file)
        # im_marked = add_box_dot_with_color(image, bbox_2d, 'red')

        img2ann[image_file].append((object_name, center_3d, bbox_2d, cam_k))

    with open("omni3d_img2ann.json", "w") as f:
        json.dump(img2ann, f)

    im_qa_pairs = []
    for key in img2ann:

        if len(img2ann[key]) <= 1:
            continue

        obj_entries = img2ann[key]

        # choose two random objects

        obj1_entry, obj2_entry = random.sample(obj_entries, 2)

        obj1_name = obj1_entry[0]
        obj1_3dLoc = obj1_entry[1]
        obj1_bbox = obj1_entry[2]

        obj2_name = obj2_entry[0]
        obj2_3dLoc = obj2_entry[1]
        obj2_bbox = obj2_entry[2]

        if obj1_name == obj2_name:
            # mark each object with a box of different color
            image = Image.open(key)
            im_marked = add_box_dot_with_color(image, obj1_bbox, 'red')
            im_marked = add_box_dot_with_color(im_marked, obj2_bbox, 'blue')
        
        # make some spatial QA
        question = f"Is the {obj1_name} to the left or right of the {obj2_name}?"
        if obj1_3dLoc[0] < obj2_3dLoc[0]:
            answer = "left"
        else:
            answer = "right"
        
        im_qa_pairs.append((key, question, answer))

        question = f"Is the {obj1_name} above or below the {obj2_name}?"
        if obj1_3dLoc[1] < obj2_3dLoc[1]:
            answer = "above"
        else:
            answer = "below"

        im_qa_pairs.append((key, question, answer))

        question = f"Is the {obj1_name} in front of or behind the {obj2_name}?"
        if obj1_3dLoc[2] < obj2_3dLoc[2]:
            answer = "in front of"
        else:
            answer = "behind"

        im_qa_pairs.append((key, question, answer))


    with open("omni3d_qa_pairs.json", "w") as f:
        json.dump(im_qa_pairs, f)
    
