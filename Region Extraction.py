#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Rob Williams
contact: jayrobwilliams@gmail.com
created: 01/15/19
updated: 07/09/19

this script uses openCV to identify keypoints in images with SURF, save the
RootSIFT descriptors to .csv files, save images with keypoints identified,
and extract and save the regions of interest around each keypoint as images
for further visual inspection
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