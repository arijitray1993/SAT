import json
import os
import sys
import random
import tqdm
from collections import defaultdict
import pdb

def get_qa_type(question):
    question_type = "other"
    
    if "how did the camera" in question.lower() or "is the camera moving" in question.lower():
        question_type = "action_sequence"

    if ("need to go" in question.lower()):
        question_type = "goal_aim"

    if "any of the objects in the initial" in question.lower():
        question_type = "obj_movement"

    if "if i" in question.lower():
        question_type = "action_consequence"

    if 'if i move to the' in question.lower() or "for someone at the" in question.lower():
        question_type = "perspective"

    
    return question_type


if __name__ == "__main__":
    '''
    required format for llava
    [
        {
            "id": "997bb945-628d-4724-b370-b84de974a19f",
            "image": "part-000001/997bb945-628d-4724-b370-b84de974a19f.jpg",
            "conversations": [
            {
                "from": "human",
                "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image."
            },
            {
                "from": "gpt",
                "value": "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render. "
            },
            ]
        },
        ...
    ]
    '''
    split = "train"

    if split=="train":
        spatial_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_spatial_qas_v2_train.json' 
        spatial_data = json.load(open(spatial_qa_json_path))   

        complex_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_train_v2.json' # remove v2 for prev version.
        complex_qa_json_path_split2 = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_train_v2_split2.json'
        complex_data = json.load(open(complex_qa_json_path)) + json.load(open(complex_qa_json_path_split2))
        complex_data = random.sample(complex_data, 6900)

        perspective_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/perspective_qas.json'
        perspective_data = json.load(open(perspective_qa_json_path))
        perspective_data = random.sample(perspective_data, 300) 

        print("num images in spatial data:", len(spatial_data))
        print("num images in complex data:", len(complex_data))
        print("num images in perspective data:", len(perspective_data))
        all_complex_data = []

        for house_ind, cam_pos, cam_rot, qa_entries in spatial_data:
            for question, im_order, answers in qa_entries:
                all_complex_data.append((question, im_order, answers))

        for house_ind, cam_pos, cam_rot, qa_entries in complex_data:
            for question, im_order, answers in qa_entries:
                question = question.replace("turn look straight", "look straight")

                if answers[0] == "rotated left and rotated right" or answers[0] == "rotated right and rotated left": # bug fix
                    new_answers = ["did not move", random.choice(["rotated left", "rotated right"])]
                    answers = new_answers

                all_complex_data.append((question, im_order, answers))
        
        perspective_count = 0
        qa_len_hist = defaultdict(int)
        for _,_,_, qa_entries in perspective_data:
            qa_len_hist[len(qa_entries)] += 1
            for question, im_order, answers in qa_entries:
                question = question.replace("turned towards the", "facing 90 degrees to the")
                question = question.replace("turned right", "turned right by 90 degrees")
                question = question.replace("turned left", "turned left by 90 degrees")
                perspective_count+=1
                all_complex_data.append((question, im_order, answers))
        print("pers count", perspective_count)
        print(qa_len_hist)
        
    else:
        complex_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_val_v2.json' # remove v2 for prev version.
        complex_data = json.load(open(complex_qa_json_path))

        perspective_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/perspective_qas.json'
        perspective_data = json.load(open(perspective_qa_json_path))

        print("num images in complex data:", len(complex_data))
        
        all_complex_data = []
        all_action_consequence_data = []
        for house_ind, cam_pos, cam_rot, qa_entries in complex_data[int(len(complex_data)*0.1):]:
            for question, im_order, answers in qa_entries:
                question = question.replace("turn look straight", "look straight")

                if answers[0] == "rotated left and rotated right" or answers[0] == "rotated right and rotated left": # bug fix
                    new_answers = ["did not move", random.choice(["rotated left", "rotated right"])]
                    answers = new_answers
                
                qa_type = get_qa_type(question)
                #if qa_type == "action_consequence":
                #    all_action_consequence_data.append((question, im_order, answers))
                #else: 
                all_complex_data.append((question, im_order, answers))
        
        #all_complex_data += random.sample(all_action_consequence_data, 1000)

        perspective_count = 0
        pers_im_count = 0
        for _,_,_, qa_entries in perspective_data[int(len(perspective_data)*0.1):]:
            pers_im_count += 1
            for question, im_order, answers in qa_entries:
                question = question.replace("turned towards the", "facing 90 degrees to the")
                question = question.replace("turned right", "turned right by 90 degrees")
                question = question.replace("turned left", "turned left by 90 degrees")

                all_complex_data.append((question, im_order, answers))
                perspective_count += 1
                if perspective_count >= 779:
                    break
            if perspective_count >= 779:
                break

        print("num images in perspective data:", pers_im_count)

    llava_data = []
    qa_type_counts = defaultdict(int)
    answer_hist = defaultdict(lambda: defaultdict(int))
    # pdb.set_trace()
    for question, image_paths, answer_choices in all_complex_data:

            qa_type = get_qa_type(question)
            qa_type_counts[qa_type] += 1

            image_paths = ["/data/input/arijitr/research/r2d3/custom_datasets/custom_datasets/"+os.path.join(*image_path.split('/')[-5:]) for image_path in image_paths]

            answer = answer_choices[0]
            answer_hist[qa_type][answer] += 1

            ans_choice_order = answer_choices
            ans_choice_order = ['"'+ans+'"' for ans in ans_choice_order]
            random.shuffle(ans_choice_order)
            answer_choices_format = " or ".join(ans_choice_order)

            image_prompt = "<image>"*len(image_paths)
            question = image_prompt + "\n" + question + "Answer the question using a single word or phrase. Choose between the following options: " + answer_choices_format+ "."

            # make it in required format
            entry = {
                "id": house_ind,
                "image": image_paths,
                "conversations": [
                    {
                        "from": "human",
                        "value": question
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }

            llava_data.append(entry)

    print(qa_type_counts)
    print("Total:", len(llava_data))
    #pdb.set_trace()
    #print(answer_hist['action_consequence'])

    precise_json_data1 = json.load(open(f"/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_captions_train.json"))
    precise_json_data2 = json.load(open(f"/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_captions_train_split2.json"))
    precise_json_data = precise_json_data1 + precise_json_data2

    print("num images in precise data:", len(precise_json_data))
    num_precise_qa = 0
    for frame_path, visible_obj_descs, coordinate_text, field_of_view in precise_json_data:
        num_precise_qa += len(visible_obj_descs)

    print("num precise qa:", num_precise_qa)
    
    with open(f'llava_complex_spatial_reasoning_{split}.json', 'w') as f:
        json.dump(llava_data, f)