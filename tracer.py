#!/usr/bin/env python
"""
Before running, go to terminal and issue the following command:
export PYTHONPATH=/home/tsmo-tstorm/Documents/research/pysot:$PYTHONPATH
"""

import os
import argparse
import sys
import time
from alive_progress import alive_bar
import jetson.inference
import jetson.utils
import getch

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from captureROI import *

from commander import adjust

from torch2trt import torch2trt

torch.set_num_threads(24)


parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()


def main():
    
        
    
    ############################################################
    target = False
    with alive_bar(10) as bar:
        print("---------------------------")
        print("-----BEGIN INITIALIZATION-----")
        print("---------------------------")

        cfg.merge_from_file(args.config)
        bar()
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        bar()
        device = torch.device('cuda' if cfg.CUDA else 'cpu')
        bar()
        model = ModelBuilder()
        bar()
        model.load_state_dict(torch.load(args.snapshot,
            map_location=lambda storage, loc: storage.cpu()))
        bar()
        model.eval().to(device)
        bar()
        tracker = build_tracker(model)
        bar()
        
        net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)     #Initialize net for object detection
        
        while not target:
            first_frame = True
            if args.video_name:
                video_name = args.video_name.split('/')[-1].split('.')[0]
                frame = get_frames(args.video_name)
                init_rect = cv2.selectROI(video_name, frame)
                lockStart = time.time()
            else:
                video_name = 'webcam'
                frame, init_rect = captureROI()
                
            
            if contains_person(jetson.utils.cudaFromNumpy(frame), net):
                target = getch.getch()
                target = (target == '\n')
            else : target = True
        lockStart = time.time()
        bar()
        print("----------------------------------------")
        print("------------BEGIN TARGET LOCK-----------")
        print("----------------------------------------")
        bar()
        tracker.init(frame, init_rect)
        print(f"------------Lock Time: {round(time.time() - lockStart, 3)}------------")
        print("----------------------------------------")
        bar()
        
        
    for frame in get_frames(args.video_name): 
        #frame = cv2.flip(frame, 1)  
        outputs = tracker.track(frame)
        #print(outputs)
        if 'polygon' in outputs:
            polygon = np.array(outputs['polygon']).astype(np.int32)
            cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                          True, (0, 255, 0), 3)
            mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
            mask = mask.astype(np.uint8)
            mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
            frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
        else:
            bbox = list(map(int, outputs['bbox']))
            #print(center := (bbox[0] + int(bbox[2]/2)), (bbox[1] + int(bbox[3]/2)))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)
            
            center = ((bbox[0] + int(bbox[2]/2)), (bbox[1] + int(bbox[3]/2)))
            #print(adjust(center, frame.shape))
            adjust(center, frame.shape)
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
       
    


if __name__ == "__main__":
    main()


