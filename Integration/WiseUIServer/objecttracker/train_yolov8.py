import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))     # append current dir to PATH
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../"))
import copy

import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch


def put_in_eval_mode(trainer, n_layers=22):
  for i, (name, module) in enumerate(trainer.model.named_modules()):
    if name.endswith("bn") and int(name.split('.')[1]) < n_layers:
      module.eval()
      module.track_running_stats = False


def compare_dicts(state_dict1, state_dict2):
    # Compare the keys
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    if keys1 != keys2:
        print("Models have different parameter names.")
        return False

    # Compare the values (weights)
    for key in keys1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            print(f"Weights for parameter '{key}' are different.")
            if "bn" in key and "22" not in key:
              state_dict1[key] = state_dict2[key]

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    old_dict = copy.deepcopy(model.state_dict())

    model.add_callback("on_train_epoch_start", put_in_eval_mode)
    model.add_callback("on_pretrain_routine_start", put_in_eval_mode)

    data = "C:/Woojin/research/wiseui_base/Integration/WiseUIServer/objecttracker/Yolov8_extraobjshape.v2i.yolov8/data.yaml"
    # results = model.train(data=data, pretrained=True, epochs=20, imgsz=640, device=[0])
    results = model.train(data=data, freeze=22, epochs=20, imgsz=640)
    print("end")

    compare_dicts(old_dict, model.state_dict())

    new_state_dict = dict()

    #  Increment the head number by 1 in the state_dict
    for k, v in model.state_dict().items():
        if k.startswith("model.model.22"):
            new_state_dict[k.replace("model.22", "model.23")] = v

    # Save the current state_dict. Only layer 23.
    torch.save(new_state_dict, "yolov8n_lp.pth")

    model_2 = YOLO('ultralytics/cfg/models/v8/yolov8-2xhead.yaml', task="detect").load('yolov8n.pt')
    state_dict = torch.load("yolov8n_lp.pth")

    model_2.load_state_dict(state_dict, strict=False)

"""
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 
15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 
45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 
55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 
65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 
75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
"""


