
import os, gc, sys, yaml, shutil
from pathlib import Path
from tqdm.auto import tqdm

import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection

import PIL
import cv2

import xml
import xml.etree.ElementTree as ET
from xml.dom import minidom

random.seed(42)  

# import wandb
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# secret_value_0 = user_secrets.get_secret("WANDB_API_KEY")
# wandb.login(key=secret_value_0)

# DATA_PATH = Path("/kaggle/input/road-sign-detection")
# os.listdir(DATA_PATH)

# !cat /kaggle/input/road-sign-detection/annotations/road4.xml

# function to get the data from XML annotation
# def extract_info_from_xml(xml_file):
#     root = ET.parse(xml_file).getroot()

#     # initialize the info dict
#     info_dict = {}
#     info_dict['bboxes'] = []

#     # parse the xml tree
#     for elem in root:
#         # get file name
#         if elem.tag == "filename":
#             info_dict['filename'] = elem.text

#         # get the image size
#         elif elem.tag == "size":
#             image_size = []
#             for subelem in elem:
#                 image_size.append(int(subelem.text))

#             info_dict['image_size'] = tuple(image_size)

#         # get details of the bounding box
#         elif elem.tag == "object":
#             bbox = {}
#             for subelem in elem:
#                 if subelem.tag == "name":
#                     bbox["class"] = subelem.text
#                 elif subelem.tag == "bndbox":
#                     for subsubelem in subelem:
#                         bbox[subsubelem.tag] = int(subsubelem.text)
#             info_dict['bboxes'].append(bbox)

#     return info_dict

# class_name_to_id_mapping = {
#     "trafficlight": 0,
#     "stop": 1,
#     "speedlimit": 2,
#     "crosswalk": 3
# }

# ANNOTATION_DIR = Path("/kaggle/input/road-sign-detection/annotations")
# info_dict = extract_info_from_xml(ANNOTATION_DIR/"road350.xml")
# info_dict['filename'] = info_dict['filename'].replace("png", "txt")
# info_dict

# PIL.Image.open("/kaggle/input/road-sign-detection/images/road0.png").size

# labels = []
# for box in info_dict['bboxes']:
#     label = [class_name_to_id_mapping[box['class']], box['xmin'], box['ymin'], box['xmax'], box['ymax']]
#     labels.append(label)

# labels

# def create_dataset(data_path):
#     LABELS = []
#     IMAGE_PATHS = []
#     IMAGE_DIR = data_path/'images'
#     img_paths = os.listdir(IMAGE_DIR)
#     ANNOTATION_DIR = data_path/'annotations'
#     for i, path in enumerate(tqdm(img_paths, total=len(img_paths))):
#         iPath = f"{IMAGE_DIR}/{path}"
#         assert os.path.exists(iPath)
#         image = PIL.Image.open(iPath)
#         image_w, image_h = image.size
#         name = Path(iPath).stem
#         aPath = f"{ANNOTATION_DIR}/{name}.xml"
#         assert os.path.exists(aPath)
#         info_dict = extract_info_from_xml(aPath)
#         annots = []
#         for b in info_dict['bboxes']:
#             b_center_x = (b["xmin"] + b["xmax"]) / 2
#             b_center_y = (b["ymin"] + b["ymax"]) / 2
#             b_width    = (b["xmax"] - b["xmin"])
#             b_height   = (b["ymax"] - b["ymin"])

#             # Normalize the coordinates
#             b_center_x /= image_w
#             b_center_y /= image_h
#             b_width    /= image_w
#             b_height   /= image_h

#             # Convert to YOLO format
#             annot = [class_name_to_id_mapping[b['class']], b_center_x, b_center_y, b_width, b_height]
#             annots.append(annot)

#         # Append the image path and corresponding annotations
#         IMAGE_PATHS.append(iPath)
#         LABELS.append(annots)

#     return IMAGE_PATHS, LABELS

# iPath, lPath = create_dataset(DATA_PATH)

# # lPath

# from sklearn import model_selection

# train_images, val_images, train_annotations, val_annotations = model_selection.train_test_split(
#     iPath, lPath, test_size=0.2, random_state=42
# )

# # val_images, val_annotations, test_images, test_annotations = model_selection.train_test_split(
# #     val_images, val_annotations, test_size=0.5, random_state=42
# # )

# # !mkdir /kaggle/working/roadsigndetection
# # !mkdir /kaggle/working/roadsigndetection/train
# # !mkdir /kaggle/working/roadsigndetection/valid

# # !mkdir /kaggle/working/roadsigndetection/train/images /kaggle/working/roadsigndetection/train/labels
# # !mkdir /kaggle/working/roadsigndetection/valid/images /kaggle/working/roadsigndetection/valid/labels

# def move_file_to_folder(file, destination):
#     try:
#         shutil.move(file, destination)
#     except Exception as e:
#         print(f"Error moving file {file}: {e}")
#         assert False

# def write_annotations(name, annots, annot_dest):
#     with open(f'{annot_dest}/{name}.txt', 'w') as f:
#         for annot in annots:
#             annot_str = ' '.join(map(str, annot))
#             f.write(annot_str + '\n')

# def create_folders(img_paths, annot_paths, img_dest, annot_dest):
#     for i in tqdm(range(len(img_paths))):
#         img_path = img_paths[i]
#         img = PIL.Image.open(img_path).convert("RGB")
#         name = Path(img_path).stem
#         img.save(f"{name}.png")
#         file = f"{name}.png"
#         move_file_to_folder(file, img_dest)

#         annots = annot_paths[i]
#         write_annotations(name, annots, annot_dest)

# create_folders(img_paths=train_images,
#                annot_paths=train_annotations,
#                img_dest="/kaggle/working/roadsigndetection/train/images",
#                annot_dest="/kaggle/working/roadsigndetection/train/labels"
#               )

# create_folders(img_paths=val_images,
#                annot_paths=val_annotations,
#                img_dest="/kaggle/working/roadsigndetection/valid/images",
#                annot_dest="/kaggle/working/roadsigndetection/valid/labels"
#               )

# # os.listdir("/kaggle/working/roadsigndetection/train/labels")

# class CFG:
#     DEBUG = False
#     FRACTION = 0.05 if DEBUG else 1.0
#     SEED = 42

#     # training
# #     EPOCHS = 3 if DEBUG else 50 # 100
#     EPOCHS = 15
#     BATCH_SIZE = 8

#     BASE_MODEL = 'yolov8n' # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, yolov9c, yolov9e
#     BASE_MODEL_WEIGHTS = f'{BASE_MODEL}.pt'
#     EXP_NAME = f'road_sign_{EPOCHS}_epochs'

#     OPTIMIZER = 'auto' # SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto
#     LR = 1e-3
#     LR_FACTOR = 0.01
#     WEIGHT_DECAY = 5e-4
#     DROPOUT = 0.0
#     PATIENCE = 20
#     PROFILE = False

#     CUSTOM_DATASET_DIR = "/kaggle/working/roadsigndetection"
#     OUTPUT_DIR = os.getcwd()
# #     OUTPUT_DIR = "/kaggle/working/roadsigndetection/"

# dict_file = {
#     "train": os.path.join(CFG.CUSTOM_DATASET_DIR, "train"),
#     "val": os.path.join(CFG.CUSTOM_DATASET_DIR, "valid"),
# #     "test": os.path.join(CFG.CUSTOM_DATASET_DIR, "test"),
#     "nc": 4,
#     "names": {0: "trafficlight",
#              1: "stop",
#              2: "speedlimit",
#              3: "crosswalk"},
# }

# with open(os.path.join(CFG.OUTPUT_DIR, "data.yaml"), "w+") as f:
#     yaml.dump(dict_file, f)

# # read yaml file created
# def read_yaml_file(file_path=CFG.CUSTOM_DATASET_DIR):
#     with open(file_path, "r") as f:
#         try:
#             data = yaml.safe_load(f)
#             return data
#         except yaml.YAMLError as e:
#             print(f"Error reading YAML: {e}")
#             return None

# def print_yaml_data(data):
#     formatted_yaml = yaml.dump(data, default_style=False)
#     print(formatted_yaml)

# file_path = os.path.join(CFG.OUTPUT_DIR, "data.yaml")
# yaml_data = read_yaml_file(file_path)

# if yaml_data:
#     print_yaml_data(yaml_data)

# !pip install -qU ultralytics

# import torch

# from ultralytics import YOLO

# # Load pre-trained YOLO model
# # flush()
# model = YOLO(CFG.BASE_MODEL_WEIGHTS)

# model.train(
#     data = os.path.join(CFG.OUTPUT_DIR, 'data.yaml'),

#     task = 'detect',

#     imgsz = (512, 512), # (img_properties['height'], img_properties['width'])

#     epochs = CFG.EPOCHS,
#     batch = CFG.BATCH_SIZE,
#     optimizer = CFG.OPTIMIZER,
#     lr0 = CFG.LR,
#     lrf = CFG.LR_FACTOR,
#     weight_decay = CFG.WEIGHT_DECAY,
#     dropout = CFG.DROPOUT,
#     fraction = CFG.FRACTION,
#     patience = CFG.PATIENCE,
#     profile = CFG.PROFILE,

#     name = f'{CFG.BASE_MODEL}_{CFG.EXP_NAME}',
#     seed = CFG.SEED,

#     val = True,
#     amp = True,
#     exist_ok = True,
#     resume = False,
#     device = 0, # [0,1]
#     verbose = False,
# )

# # flush()

# !pip install -q roboflow

# from roboflow import Roboflow

# os.listdir("/kaggle/working/runs/detect/yolov8n_road_sign_15_epochs/weights")

# # rf = Roboflow(api_key="OCBifQIjZFRiufOmNlKZ")
# # project = rf.workspace().project("")