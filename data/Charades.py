import torch.utils.data as data
import torch
import os
import sys
import random
import numpy as np
import cv2
import glob
import csv
from PIL import Image
import matplotlib.pyplot as plt
import math
import json
import pickle


def get_video_list(data_info_scv, data_root, multi_sample_interval=10, num_classes=157):
    dataset = []
    Data_reader = csv.DictReader(open(data_info_scv, 'r'))
    mapping_file = './data/Charades/Charades_v1_mapping.txt'
    with open(mapping_file, 'r') as mapfile:
        map_act = mapfile.readlines()
    if data_info_scv[-9:-4] == 'train':    phase = 'train'
    if data_info_scv[-8:-4] == 'test':    phase = 'val'
    dataset_name = './data/Charades_' + phase + '.pkl'

    sample_num = 1

    if os.path.exists(dataset_name):
        with open(dataset_name, 'rb') as f:
            print("Loading cached result from '%s'" % dataset_name)
            return pickle.load(f)
    else:
        for No_v, vd_info in enumerate(Data_reader):
            print(No_v+1)
            v_id = vd_info['id']
            v_subject = vd_info['subject']
            v_scene = vd_info['scene']
            v_object = vd_info['objects']
            v_actions = vd_info['actions']
            v_length = float(vd_info['length'])
            num_frames = len(os.listdir(os.path.join(data_root, v_id)))

            label = np.zeros((num_classes, num_frames), np.uint8)
            fps = num_frames / v_length  # eg: 100/5.02 = 24

            assert 24 < fps < 25, "The fps is  error "

            # ---------get video label -------
            video_label = np.zeros((num_classes, 1))
            if len(v_actions) > 0:
                for ann in v_actions.split(';'):
                    v_class = int(ann.split(' ')[0][1:])
                    video_label[v_class] = 1
            # ---------get video label -------

            map_v = []
            if len(v_actions) > 0:
                for ann in v_actions.split(';'):
                    label_v = int(ann.split(' ')[0][1:])
                    map_v.append(map_act[label_v][:-1])
                    start = float(ann.split(' ')[1])
                    end = float(ann.split(' ')[2])
                    for fm in range(0, num_frames, 1):
                        if (fm/fps) > start and (fm/fps) < end:
                            label[label_v, fm] = 1
                for copy_time in range(sample_num):
                    # dataset.append((v_id, label, v_length, num_frames, map_v, v_scene))
                    dataset.append((v_id, label, v_length, num_frames, map_v, v_scene,video_label ))
            else:
                for copy_time in range(sample_num):
                    # dataset.append((v_id, label, v_length, num_frames, map_v, v_scene))
                    dataset.append((v_id, label, v_length, num_frames, map_v, v_scene,video_label))

        with open(dataset_name, 'wb') as file:
            pickle.dump(dataset,  file)

        return dataset


def Fm_list(input_frame_num, start_frame, duration, sample_interval):
    offset = [(x*sample_interval+start_frame-1) % duration for x in range(input_frame_num)]
    return (np.array(offset) + 1).tolist()


def Ld_RGB_Fm(image_dir, vid, fm_list):
    frames = []
    for _, frame_No in enumerate(fm_list):
        img = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(frame_No).zfill(6) + '.jpg'))[:, :, [2, 1, 0]]
        frames.append(img)
    return frames
    # return np.asarray(frames, dtype=np.uint8)


def Val_video(data_info_scv, data_root, num_classes=157, stride=32,fm_use=64):
    dataset = []
    Data_reader = csv.DictReader(open(data_info_scv, 'r'))
    mapping_file = './data/Charades/Charades_v1_mapping.txt'
    with open(mapping_file, 'r') as mapfile:
        map_act = mapfile.readlines()
    dataset_name = './data/Charades_' + 'Val_Video' + '.pkl'

    if os.path.exists(dataset_name):
        with open(dataset_name, 'rb') as f:
            print("Loading cached result from '%s'" % dataset_name)
            return pickle.load(f)
    else:
        for No_v, vd_info in enumerate(Data_reader):
            print(No_v+1)
            v_id = vd_info['id']
            v_subject = vd_info['subject']
            v_scene = vd_info['scene']
            v_object = vd_info['objects']
            v_actions = vd_info['actions']
            v_length = float(vd_info['length'])
            num_frames = len(os.listdir(os.path.join(data_root, v_id)))
            assert num_frames > 50, ' Empty video'

            map_v = []
            video_label = np.zeros((num_classes, 1))
            if len(v_actions) > 0:
                for ann in v_actions.split(';'):
                    v_class = int(ann.split(' ')[0][1:])
                    video_label[v_class] = 1
                    map_v.append(map_act[v_class][:-1])

            for start in range(1, num_frames-fm_use, stride):
                dataset.append((v_id, video_label, start, v_scene))
            dataset.append((v_id, video_label, num_frames-fm_use+1, v_scene))

        with open(dataset_name, 'wb') as file:
            pickle.dump(dataset,  file)

        return dataset


def load_bbox(bs_dir, vid, fm_list):
    file_name = bs_dir + vid + '.json'
    vd_bbox = json.load(open(file_name, 'r'))
    out_bbox = []
    for i in fm_list:
        out_bbox.append(vd_bbox['%06d' % i])
    return out_bbox


class Charades(data.Dataset):
    def __init__(self, data_info_scv, data_root, mode, fm_use=64, transforms=None, bbox_dir=None):

        self.data = get_video_list(data_info_scv, data_root)
        self.transforms = transforms
        self.mode = mode
        self.root = data_root
        self.fm_num = fm_use  # 64 or other
        self.bbox_dir = bbox_dir

    def __getitem__(self, index):
        V_id, V_label, V_duration, V_fr_num, V_map, Scene, V_Gt = self.data[index]

        st_f = random.randint(1, abs(V_fr_num-self.fm_num+1))   # some video has 60 frames
        fm_list = Fm_list(self.fm_num, st_f,  V_fr_num, 1)

        # bbox = load_bbox(self.bbox_dir, V_id, fm_list)

        imgs = Ld_RGB_Fm(self.root, V_id, fm_list)
        Org_H, Org_W = imgs[0].shape[0:2]
        label = torch.from_numpy(V_label[:, (np.array(fm_list)-1).tolist()].astype(np.int)).contiguous()   # OK pass  tensor 157 * 64

        imgs = self.transforms(imgs)

        fm_list.insert(0, index)
        fm_list.insert(1, Org_H)
        fm_list.insert(2, Org_W)
        return imgs, label, torch.FloatTensor(np.array(fm_list)).contiguous(), V_id, V_Gt
        # return imgs, label, torch.FloatTensor(np.array([index, st_f, self.fm_num])).unsqueeze(0)

    def __len__(self):
        return len(self.data)



class Charades_VD(data.Dataset):
    def __init__(self, data_info_scv, data_root, mode, fm_use=64, transforms=None):
        self.data = Val_video(data_info_scv, data_root)
        self.transforms = transforms
        self.mode = mode
        self.root = data_root
        self.fm_num = fm_use  # 64 or other

    def __getitem__(self, index):
        V_id, V_label, st_f, Scene = self.data[index]
        fm_list = [st_f+x for x in range(self.fm_num)]
        imgs = Ld_RGB_Fm(self.root, V_id, fm_list)
        Org_H, Org_W = imgs[0].shape[0:2]
        label = torch.from_numpy(np.tile(V_label, (1, self.fm_num)))
        imgs = self.transforms(imgs)

        return imgs, label, V_id, st_f, Scene

    def __len__(self):
        return len(self.data)



