# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 01:49:02 2021

@author: nasheedyasin
"""

import os
from tqdm import tqdm
from lxml import etree


class VocParser(object):
    """Parses the VOC XML file for each image into the text file format 
    (Yolo v3 Text) that is required by this repo.
    """
    
    def __init__(self):
        self.class_mapping = dict()
        self.class_cnt = 0
        
    def __format_dataline(self, xml_fpath: str,
                          im_dir: str = None):    
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
            bbox = dict()
            for child in item.xpath('./bndbox')[0].getchildren():
                bbox[child.tag] = str(round(float(child.text.strip())))
            # Ensuring that the order is always xmin, ymin, xmax, ymax 
            bbox = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
    
            obj_name = item.findtext('./name')
            # Object class-code is appended to the end of the 'bbox'
            if obj_name not in self.class_mapping:
                self.class_mapping[obj_name] = self.class_cnt
                bbox.append(str(self.class_mapping[obj_name]))
                self.class_cnt += 1
            else:
                bbox.append(str(self.class_mapping[obj_name]))
    
            bounding_boxes.append(",".join(bbox))
    
        # One line in the labelled-data textfile
        bounding_boxes = " ".join(bounding_boxes)   
        dataline = "{} {}".format(impath, bounding_boxes)
    
        return dataline
    
    def __call__(self, dir_path: str, op_fpath: str, im_dir: str = None):
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
            datalines.append(self.__format_dataline(xml_fpath, im_dir))
    
        # Writing to file
        with open(op_fpath, 'w') as f:
            f.write("\n".join(datalines))
    
    def sav_class_map(self, op_fpath: str):
        assert op_fpath.endswith('.txt'), "{} is not a valid text file path"\
            .format(op_fpath)
    
        # Sorting by class_cnt
        mapping = [[val, key] for key, val in self.class_mapping.items()]
        mapping.sort()
        mapping = [key for val, key in mapping]
    
        with open(op_fpath, 'w') as f:
            f.write("\n".join(mapping))

if __name__ == '__main__':
    dir_path = r"D:\Datasets\asl-voc\train"
    op_fpath = r"D:\Datasets\asl-voc\train.txt"
    cl_fpath = r"D:\Datasets\asl-voc\classes.txt"

    parser = VocParser()
    parser(dir_path, op_fpath)
    parser.sav_class_map(cl_fpath)
