# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 01:49:02 2021

@author: nasheedyasin

Parses the VOC XML file for each image into the text file that is required by 
this repo.
"""

import os
from tqdm import tqdm
from lxml import etree

CLASS_CNT = 0
CLASS_MAPPING = dict()

def format_dataline(xml_fpath: str,
                    im_dir: str = None):

    global CLASS_CNT

    # Create element tree object 
    tree = etree.parse(xml_fpath)
    root = tree.getroot()
    
    impath = root.findtext('./path')
    if not os.path.dirname(impath):
        if im_dir is None: 
            impath = os.path.join(os.path.dirname(xml_fpath), impath)
        else:
            impath = os.path.join(im_dir, impath)

    assert os.path.isfile(impath), "Image not found at {}".format(impath)

    bounding_boxes = list()
    for item in root.findall('./object'):
        bbox = [child.text.strip()
                for child in item.xpath('./bndbox')[0].getchildren()]

        obj_name = item.findtext('./name')
        # Object class-code is appended to the end of the 'bbox'
        if obj_name not in CLASS_MAPPING:
            CLASS_MAPPING[obj_name] = CLASS_CNT
            bbox.append(str(CLASS_MAPPING[obj_name]))
            CLASS_CNT += 1
        else:
            bbox.append(str(CLASS_MAPPING[obj_name]))

        bounding_boxes.append(",".join(bbox))

    # One line in the labelled-data textfile
    bounding_boxes = " ".join(bounding_boxes)   
    dataline = "{} {}".format(impath, bounding_boxes)

    return dataline

def conv_frm_dir(dir_path: str, op_fpath: str, im_dir: str = None):
    assert os.path.isdir(dir_path), "{} is not a valid path".format(dir_path)
    assert op_fpath.endswith('.txt'), "{} is not a valid text file path"\
        .format(op_fpath)
    assert im_dir is None or os.path.isdir(im_dir), "{} is not a valid path"\
        .format(im_dir)

    _, _, files = next(os.walk(dir_path))
    valid_files = [file for file in files if file.lower().endswith('.xml')]

    datalines = list()
    for file in tqdm(valid_files):
        xml_fpath = os.path.join(dir_path, file)
        datalines.append(format_dataline(xml_fpath, im_dir))

    # Writing to file
    with open(op_fpath, 'w') as f:
        f.write("\n".join(datalines))

def sav_class_map(op_fpath: str):
    assert op_fpath.endswith('.txt'), "{} is not a valid text file path"\
        .format(op_fpath)

    # Sorting by class_cnt
    mapping = [[val, key] for key, val in CLASS_MAPPING.items()]
    mapping.sort()
    mapping = [key for val, key in mapping]

    with open(op_fpath, 'w') as f:
        f.write("\n".join(mapping))

if __name__ == '__main__':
    dir_path = r"D:\Datasets\asl-voc\train"
    op_fpath = r"D:\Datasets\asl-voc\train.txt"
    cl_fpath = r"D:\Datasets\asl-voc\classes.txt"
    
    conv_frm_dir(dir_path, op_fpath)
    sav_class_map(cl_fpath)
