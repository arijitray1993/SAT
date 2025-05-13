import json


qa_json_path1 = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas.json'
qa_json_path2 = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_split2.json'

split1 = json.load(open(qa_json_path1))
split2 = json.load(open(qa_json_path2))

qas = split1 + split2

print("total_num_points: ", len(qas))

with open('/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_train.json', 'w') as f:
    json.dump(qas, f)