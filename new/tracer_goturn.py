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
import asyncio

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from captureROI import *

from commander import adjust
from mavsdk import System
#from mavsdk import (OffboardError, PositionNedYaw)

#from torch2trt import torch2trt

torch.set_num_threads(24)
buffer = {'x':[], 'y':[]}
FONT = cv2.FONT_HERSHEY_SIMPLEX
fps = 100

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()


async def run():
#-----------------------------------------------------------------------
#Initialize Tracking Background System 
#-----------------------------------------------------------------------

    target = False
    print("---------------------------")
    print("-----BEGIN INITIALIZATION-----")
    print("---------------------------")
    
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    
    model = ModelBuilder()
    model.load_state_dict(torch.load(args.snapshot,
    map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)
    tracker = build_tracker(model)

    net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)     #Initialize net for object detection

    #-----------------------------------------------------------------------
    #Target Locking Sequence
    #-----------------------------------------------------------------------
    while not target:
        first_frame = True
        if args.video_name:
            video_name = args.video_name.split('/')[-1].split('.')[0]
            frame = get_frames(args.video_name)
            init_rect = cv2.selectROI(video_name, frame)
            lockStart = time.time()
        else:
            video_name = 'webcam'
            frame, init_rect = await captureROI()
		
        if await contains_person(jetson.utils.cudaFromNumpy(frame), net):
            target = getch.getch()
            target = (target == '\n')
        else : target = True

    lockStart = time.time()
    print("----------------------------------------")
    print("------------BEGIN TARGET LOCK-----------")
    print("----------------------------------------")
    tracker.init(frame, init_rect)
    print(f"------------Lock Time: {round(time.time() - lockStart, 3)}------------")
    print("----------------------------------------")

        
    for frame in get_frames(args.video_name): 
        frame_time = time.time()
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
            #frame = cv2.putText('gray', fps, (7,70), FONT, 3, (100,255,0), 3, cv2.LINE_AA)
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)
            
            center = ((bbox[0] + int(bbox[2]/2)), (bbox[1] + int(bbox[3]/2)))
            filtered_target = await moving_average_filter(center)
            #print(adjust(center, frame.shape))
            await adjust(filtered_target, frame.shape)
        fps = 1 / (time.time() - frame_time)
        
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
   


    
async def moving_average_filter(coord, points=3):
    """Applies Low-Pass Moving Average Filter to a pair of (x,y) coordinates"""

    #Append new coordinates to filter buffer
    buffer['x'].append(coord[0])
    buffer['y'].append(coord[1])

    #If the buffer is full, discard the oldest point
    if (len(buffer['x']) > points):
        buffer['x'] = buffer['x'][1:]
        buffer['y'] = buffer['y'][1:]

    #Get buffer size
    n = len(buffer['x'])

    #Sum each side of buffer
    x_sum = sum(buffer['x'])
    y_sum = sum(buffer['y'])

    #Compute averages
    x_filt = int(round(x_sum / n))
    y_filt = int(round(y_sum / n))

    #Back to the show
    return (x_filt, y_filt)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())


