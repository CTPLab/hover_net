from collections import defaultdict
from random import shuffle
from datetime import datetime
from PIL import Image
from pathlib import Path
from typing import NamedTuple
import numpy as np
import warnings
import torch
import os
import csv
import re
import json
import itertools
import argparse
import random
import sys
import cv2
sys.path.append('.')


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def visual_instances(mask, type_map, color_dict, canvas=None):
    """
    Args:
        mask: array of NXW
    Return:
        Image with the instance overlaid
    """
    canvas = np.full(mask.shape + (3,), 0., dtype=np.float) \
        if canvas is None else np.copy(canvas)

    for tmap in type_map:
        # tmap: [instance_id, type_id]
        if tmap[1] == 0:
            continue
        inst_map = np.array(mask == tmap[0], np.uint8)
        y1, y2, x1, x2 = bbox(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= mask.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= mask.shape[0] - 1 else y2
        inst_map_crop = inst_map[y1:y2, x1:x2]
        inst_canvas_crop = canvas[y1:y2, x1:x2]
        contours, _ = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        color = color_dict[str(tmap[1])][1]
        # print(color)
        cv2.drawContours(inst_canvas_crop, contours, -1, color, 2)
        canvas[y1:y2, x1:x2] = inst_canvas_crop
    return canvas


out_dir = Path('/home/histopath/Data/SCRC_cell_test/')
inst_files = out_dir.glob('**/*ImageActualTif_inst.npy*')
segment_dict = defaultdict(list)
for inst_file in inst_files:
    # TMA_15_10_IA_HE_12786_C05R02_ImageActualTif_inst.npy
    inst_nm = inst_file.stem
    # TMA_15_10_IA_HE_12786_C05R02_
    inst_nm = inst_nm.split('ImageActualTif')[0]
    inst_file = str(inst_file)
    type_file = inst_file.replace('_inst.npy', '_type.npy')
    segment_dict[inst_nm].append(inst_file)
    segment_dict[inst_nm].append(type_file)

tma_files = out_dir.glob('**/*ImageActualTif.tif*')
tma_dict = defaultdict(str)
for tma_file in tma_files:
    # TMA_15_10_IA_HE_12786_C05R02_ImageActualTif.tif
    tma_nm = tma_file.stem
    # TMA_15_10_IA_HE_12786_C05R02_
    tma_nm = tma_nm.split('ImageActualTif')[0]
    tma_dict[tma_nm] = str(tma_file) 


color_json = Path('type_info.json')
with open(str(color_json), 'r') as c_json:
    color_dict = json.load(c_json)
    print(color_dict)

# print(segment_dict)
# print(tma_dict)
for key, val in tma_dict.items():
    print(key)
    inst_file, type_file = segment_dict[key]
    inst_npy = np.load(inst_file)
    type_npy = np.load(type_file)
    tma_img = Image.open(str(val))
    # tma_img = tma_img.resize((2048, 2048))
    tma_npy = np.asarray(tma_img.convert('RGB'))
    tma_npy = visual_instances(inst_npy, type_npy, color_dict, canvas=tma_npy)
    tma_vis = Image.fromarray(
        np.uint8(tma_npy)).convert('RGB')
    # tma_vis = tma_vis.resize((2048, 2048))
    tma_vis_path = out_dir / \
        'overlay' / '{}.jpg'.format(Path(val).stem)
    tma_vis.save(str(tma_vis_path))
    # break
