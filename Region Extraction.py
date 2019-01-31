#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Rob Williams
contact: jayrobwilliams@gmail.com
created: 01/15/19
updated: 01/29/19

this script uses openCV to identify keypoints in images with SURF, save the
RootSIFT descriptors to .csv files, and extract and save the regions of interest
around each keypoint as images for further visual inspection
"""

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# rootSIFT https://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/
def feature_extract(img, bwd = 7500, eps=1e-7):
    
    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # instantiate SURF
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=bwd, upright=1)
    
    # compute keypoints
    kp = surf.detect(img_gray, None)
    
    # apply Gaussian blur to image
    img_gb = cv2.GaussianBlur(img_gray, (5,5), 0)
     
    # instantiate SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    
    # extract normal SIFT descriptors from keypoints
    (kp, desc) = sift.compute(img_gb, kp)
    
    # need to deal w/ all black or mainly monochrome frames w/ no keypoints
    if not kp:
        return(False, False)
    
    # L1 normalize vectors and take the square-root
    desc /= (desc.sum(axis=1, keepdims=True) + eps)
    desc = np.sqrt(desc)
    
    # return keypoints and descriptions
    return(kp, desc)
                 
# extract regions of interest around keypoints and save to specified directory
def keypoint_extract(img, kpts, img_name, output_dir, px = 16):
    
    # create output directory to hold regions of interest
    os.makedirs(output_dir, exist_ok=True)
    
    # get video prefix for filenames
    vid_prefix = output_dir.split("/")[-1]
    
    # iterate through keypoints
    for i, kp in enumerate(kpts):
    
        # convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # get keypoint coordinates
        kp_coord = (round(kp.pt[0]), round(kp.pt[1]))
        
        # extract area of image around keypoint
        kp_context = img_gray[kp_coord[1]-px:kp_coord[1]+px,
                              kp_coord[0]-px:kp_coord[0]+px]
        
        # write area around keypoint to directory
        cv2.imwrite(os.path.join(output_dir, "%s_%s_kp%d.jpg" %
                                 (vid_prefix, img_name, i)), kp_context)
        
    # save keypoints image for visual inspection
    plt.imsave(os.path.join(output_dir, "%s_%s_kp.jpg" % (vid_prefix, img_name)),
               cv2.drawKeypoints(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), kpts,
                                 None, (0,255,0), 0))

# get subdirectories in image directory, omitting hidden
dirs = [f for f in os.listdir('proc/vid_proc') if not f.startswith('.')]

# iterate through videos
for d in dirs:
    
    # get create filepath for reading in video frames
    vid_dir = os.path.join('proc/vid_proc', d)
    
    # create path for output files for video frames
    out_path = os.path.join('proc/kp_proc/', d)
    
    # create subdirectory for saved frames
    os.makedirs(out_path, exist_ok=True)
    
    # get filenames for frames in video directory
    frames = [f for f in os.listdir(vid_dir) if not f.startswith('.')]
    
    # creat empty array to hold keypoint descriptors
    desc_array = np.empty((0, 128))
    
    for f in frames:
        
        # get file name for output directory
        file_s = f.split('.')[0]
        
        # read in image
        img = cv2.imread(os.path.join(vid_dir, f))
        
        # detect keypoints with SURF and extract descriptors with RootSIFT
        kpts, descs = feature_extract(img)
        
        # skip to next iteration if no keypoints found in frame
        if not kpts:
            continue
        
        # concatenate RootSIFT descriptors to descriptor array
        desc_array = np.concatenate([desc_array, descs])
        
        # save area around each keypoint for visual labeling of clusters
        keypoint_extract(img, kpts, file_s, out_path, px = 16)
    
    # save RootSIFT descriptors from keypoints
    np.savetxt(os.path.join(out_path, "desc.csv"), desc_array,
                  delimiter=",")

# quit script
quit()