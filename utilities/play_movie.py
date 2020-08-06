#!/usr/bin/env python

"""
Simple program to read in and animate a movie

Nick Walkden, May 2015
Stan Pamela, Aug 2020
"""
import os.path
from os import path
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pyFastcamTools.operation.movie_reader import mraw_reader 

def play_movie(moviefile):
    
    # Check movie file actually exists
    if (not path.exists(moviefile)):
        print('Filename does not exit! Aborting...')
        sys.exit()
    
    # prepare open-cv window
    cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
    
    # read movie
    vid = mraw_reader(moviefile)
    
    # display movie
    while(1):
        # get frames
        ret,frame,header = vid.read()
        # scaling for pixel bit size
        frames = frame[::-1].T
        frames_scale = 255. / frames.max()
        frames = np.array(frames_scale * frames, dtype=np.uint8)
        # show image
        cv2.imshow('Video',frames)
        # keyboard listener: exit is esc key is pushed
        k = cv2.waitKey(20) & 0xff
        if k == 27:
            break
    
    # clear up
    cv2.destroyAllWindows()	
    vid.release()


