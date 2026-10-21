import os
import json
import random
import numpy as np
from PIL import Image


def _resolve_dataset_path(path):
    path = os.path.expandvars(str(path))
    if os.path.exists(path):
        return path
    data_root = os.environ.get("EMBODIED3DBENCH_DATA_ROOT")
    if data_root and path.startswith("/data/"):
        return os.path.join(data_root, path.removeprefix("/data/"))
    return path

def get_intr_extr(img1_path):
    # /data/simulations/3480e3727f8a357b79be64f1bccb088ac52c4c7222bcdb7fc082f2f1732935b6/execution/20251105111855_20065389a01b4f43933a1b68e7e04d0f/output/replay_output/pick&place/camera/173/cam_high_color.jpg
    img1_path = _resolve_dataset_path(img1_path)
    w, h = Image.open(img1_path).size

    dir_split = img1_path.split('/')
    img1_name = dir_split[-1][:-10]
    replay_root = '/'.join(dir_split[:-5]) + '/replay_output/'
    replay_root = replay_root + os.listdir(replay_root)[0]
    generate_root = '/'.join(dir_split[:-5]) + '/generate_output/'
    generate_root = generate_root + os.listdir(generate_root)[0]

    state1 = replay_root + '/state.json'
    with open(state1, 'r') as f:
        state1_data = json.load(f)
    state2 = generate_root + '/state.json'
    with open(state2, 'r') as f:
        state2_data = json.load(f)

    camera1 = state1_data['cameras'] # dict
    camera2 = state2_data['cameras'] # dict

    img1_intrinsic = []
    if img1_name in camera1:
        img1_intrinsic = camera1[img1_name]['intrinsic']
    else:
        img1_intrinsic = camera2[img1_name]['intrinsic']
    
    # img1_intrinsic = {'width': 1280, 'height': 720, 'fx': 911.5816257540606, 'fy': 911.5816257540606, 'ppx': 640.0, 'ppy': 360.0}
    # convert into 3*3 matrix
    img1_intr = [
        [img1_intrinsic['fx'], 0, img1_intrinsic['ppx']],
        [0, img1_intrinsic['fy'], img1_intrinsic['ppy']],
        [0, 0, 1]
    ]
    
    # standard extrinsic 3*4
    img1_extr = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ]
    return img1_intr, img1_extr, w, h

def get_other_image(img1_path):
    # /data/simulations/3480e3727f8a357b79be64f1bccb088ac52c4c7222bcdb7fc082f2f1732935b6/execution/20251105111855_20065389a01b4f43933a1b68e7e04d0f/output/replay_output/pick&place/camera/173/cam_high_color.jpg
    img1_path = _resolve_dataset_path(img1_path)

    dir_split = img1_path.split('/')
    img1_name = dir_split[-1][:-10]
    replay_root = '/'.join(dir_split[:-5]) + '/replay_output/'
    replay_root = replay_root + os.listdir(replay_root)[0]
    generate_root = '/'.join(dir_split[:-5]) + '/generate_output/'
    generate_root = generate_root + os.listdir(generate_root)[0]
    
    state1 = replay_root + '/state.json'
    with open(state1, 'r') as f:
        state1_data = json.load(f)
    state2 = generate_root + '/state.json'
    with open(state2, 'r') as f:
        state2_data = json.load(f)

    camera1 = state1_data['cameras'] # dict
    camera2 = state2_data['cameras'] # dict
    # random select one from camera1 + camera2, its key not equal to dir_split[-1]
    all_keys = list(camera1.keys()) + list(camera2.keys())
    all_keys = [k for k in all_keys if k != dir_split[-1]]
    img2_name = random.choice(all_keys)
    
    if img2_name in camera1:
        img2_path = replay_root + '/camera/' + dir_split[-2] + '/' + img2_name + '_color.jpg'
    else:
        img2_path = generate_root + '/camera/' + dir_split[-2] + '/' + img2_name + '_color.jpg'
        
    if img1_name in camera1:
        try:
            cam1_pose = state1_data['frames'][int(dir_split[-2])]['cameras'][img1_name]['pose']
        except KeyError:
            cam1_pose = state1_data['frames'][int(dir_split[-2])]['objects'][img1_name]['pose']
    else:
        try:
            cam1_pose = state2_data['frames'][int(dir_split[-2])]['cameras'][img1_name]['pose']
        except KeyError:
            cam1_pose = state2_data['frames'][int(dir_split[-2])]['objects'][img1_name]['pose']

    if img2_name in camera1:
        try:
            cam2_pose = state1_data['frames'][int(dir_split[-2])]['cameras'][img2_name]['pose']
        except KeyError:
            cam2_pose = state1_data['frames'][int(dir_split[-2])]['objects'][img2_name]['pose']
        cam2_intrinsic = camera1[img2_name]['intrinsic']
    else:
        try:
            cam2_pose = state2_data['frames'][int(dir_split[-2])]['cameras'][img2_name]['pose']
        except KeyError:
            cam2_pose = state2_data['frames'][int(dir_split[-2])]['objects'][img2_name]['pose']
        cam2_intrinsic = camera2[img2_name]['intrinsic']

    cam1_pose = np.array(cam1_pose)
    cam2_pose = np.array(cam2_pose)
    

    # calculate the relative pose
    relative_pose = np.linalg.inv(cam1_pose) @ cam2_pose
    # calc the extrinsic of camera2 relative to camera1 3*4
    extrinsic = relative_pose[:3, :].tolist()

    return img2_path, cam2_intrinsic, extrinsic

def get_other_images(img1_path):
    # /data/simulations/3480e3727f8a357b79be64f1bccb088ac52c4c7222bcdb7fc082f2f1732935b6/execution/20251105111855_20065389a01b4f43933a1b68e7e04d0f/output/replay_output/pick&place/camera/173/cam_high_color.jpg
    img1_path = _resolve_dataset_path(img1_path)

    dir_split = img1_path.split('/')
    img1_name = dir_split[-1][:-10]
    replay_root = '/'.join(dir_split[:-5]) + '/replay_output/'
    replay_root = replay_root + os.listdir(replay_root)[0]
    generate_root = '/'.join(dir_split[:-5]) + '/generate_output/'
    generate_root = generate_root + os.listdir(generate_root)[0]
    
    state1 = replay_root + '/state.json'
    with open(state1, 'r') as f:
        state1_data = json.load(f)
    state2 = generate_root + '/state.json'
    with open(state2, 'r') as f:
        state2_data = json.load(f)

    camera1 = state1_data['cameras'] # dict
    camera2 = state2_data['cameras'] # dict
    # random select one from camera1 + camera2, its key not equal to dir_split[-1]
    all_keys = list(camera1.keys()) + list(camera2.keys())
    all_keys = [k for k in all_keys if k != dir_split[-1][:10]]
    all_keys = list(set(all_keys))

    mst_keys = []
    rdm_keys = []
    
    for img2_name in all_keys:
        rdm_keys.append(img2_name)
        # if img2_name in camera1:
        #     img2_path = replay_root + '/camera/' + dir_split[-2] + '/' + img2_name + '_color.jpg'
        # else:
        #     img2_path = generate_root + '/camera/' + dir_split[-2] + '/' + img2_name + '_color.jpg'
        # if "Left" in img2_path or "Right" in img2_path:
        #     mst_keys.append(img2_name)
        # elif "Hand" in img2_path:
        #     pass
        # else:
        #     rdm_keys.append(img2_name)

    use_keys = mst_keys
    random.shuffle(rdm_keys)
    while len(use_keys) < 2 and len(rdm_keys) > 0:
        use_keys.append(rdm_keys.pop())
    

    ret = []
    
    for img2_name in use_keys:
    
        if img2_name in camera1:
            img2_path = replay_root + '/camera/' + dir_split[-2] + '/' + img2_name + '_color.jpg'
        else:
            img2_path = generate_root + '/camera/' + dir_split[-2] + '/' + img2_name + '_color.jpg'
            
        if img1_name in camera1:
            try:
                cam1_pose = state1_data['frames'][int(dir_split[-2])]['cameras'][img1_name]['pose']
            except KeyError:
                cam1_pose = state1_data['frames'][int(dir_split[-2])]['objects'][img1_name]['pose']
        else:
            try:
                cam1_pose = state2_data['frames'][int(dir_split[-2])]['cameras'][img1_name]['pose']
            except KeyError:
                cam1_pose = state2_data['frames'][int(dir_split[-2])]['objects'][img1_name]['pose']
    
        if img2_name in camera1:
            try:
                cam2_pose = state1_data['frames'][int(dir_split[-2])]['cameras'][img2_name]['pose']
            except KeyError:
                cam2_pose = state1_data['frames'][int(dir_split[-2])]['objects'][img2_name]['pose']
            cam2_intrinsic = camera1[img2_name]['intrinsic']
        else:
            try:
                cam2_pose = state2_data['frames'][int(dir_split[-2])]['cameras'][img2_name]['pose']
            except KeyError:
                cam2_pose = state2_data['frames'][int(dir_split[-2])]['objects'][img2_name]['pose']
            cam2_intrinsic = camera2[img2_name]['intrinsic']
    
        cam1_pose = np.array(cam1_pose)
        cam2_pose = np.array(cam2_pose)
        
    
        # calculate the relative pose
        relative_pose = np.linalg.inv(cam1_pose) @ cam2_pose
        # calc the extrinsic of camera2 relative to camera1 3*4
        extrinsic = relative_pose[:3, :].tolist()

        ret.append((img2_path, cam2_intrinsic, extrinsic))

    return ret

if __name__ == "__main__":
    img1_path = '/data/simulations/3480e3727f8a357b79be64f1bccb088ac52c4c7222bcdb7fc082f2f1732935b6/execution/20251105111855_20065389a01b4f43933a1b68e7e04d0f/output/replay_output/pick&place/camera/173/cam_high_color.jpg'
    img2_path, cam2_intrinsic, extrinsic = get_other_image(img1_path)
    print("img2_path:", img2_path)
    print(f"intrinsic: {cam2_intrinsic}")
    print("extrinsic:", extrinsic)
