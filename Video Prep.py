# -*- coding: utf-8 -*-
"""
author: Rob Williams
contact: jayrobwilliams@gmail.com
created: 12/05/18
updated: 07/08/19

this script uses openCV to split a video into a number of by taking every
x-th frame, defined as a user specified interval in seconds
"""

import os
import cv2

# video capture https://stackoverflow.com/questions/22704936/reading-every-nth-frame-from-videocapture-in-opencv
def vid_split(file, outdir, interval):
    """
    function to split a video into images and save the images to an outdir
    """
    
    # get file name minus any directories
    file_n = file.split('/')[-1]
    
    # get directory structure before file name
    file_d = '/'.join(file.split('/')[:-1])
    
    # get file name for output directory
    file_s = file_n.split('.')[0]
    
    # create path for output files for video
    out_path = outdir + '/' + file_s + '/'
    
    # create subdirectory for saved frames
    os.makedirs(out_path, exist_ok=True)
    
    # open video connection
    cap = cv2.VideoCapture(file_d + '/' + file_n)
    
    # get frame rate
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    # read first frame
    ret, frame = cap.read()
    
    while(ret):
        
        # get frame number
        frame_no = int(round(cap.get(1)))
        
        # read next frame
        ret, frame = cap.read()
        
        # write frame as image every interval seconds
        if frame_no % (interval * fps) == 0:
            cv2.imwrite(out_path + "%dsecs.jpg" % (frame_no / fps), frame)
    
    # release video connection
    cap.release()