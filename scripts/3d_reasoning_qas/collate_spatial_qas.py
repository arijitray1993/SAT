import json


qa_json_path1 = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_spatial_qas_new_split_v2.json'
qa_json_path2 = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_spatial_qas_new_split_v2_split2.json'

spatial_split1 = json.load(open(qa_json_path1))
spatial_split2 = json.load(open(qa_json_path2))

spatial_qas = spatial_split1 + spatial_split2

print("total_num_points: ", len(spatial_qas))

with open('/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_spatial_qas_v2_train.json', 'w') as f:
    json.dump(spatial_qas, f)