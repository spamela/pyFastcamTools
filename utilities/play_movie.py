#!/usr/bin/env python

"""
Simple program to read in and animate a movie

Nick Walkden, May 2015
Stan Pamela, Aug 2020
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pyFastcamTools.operation.movie_reader import mraw_reader 

cv2.namedWindow('Video',cv2.WINDOW_NORMAL)

moviefile = '29786/C001H001S0001/C001H001S0001-04.mraw'
vid = mraw_reader(moviefile)

while(1):
    ret,frame,header = vid.read()
    frames = frame[::-1].T
    frames_scale = 255. / frames.max()
    frames = np.array(frames_scale * frames, dtype=np.uint8)
    cv2.imshow('Video',frames)
    k = cv2.waitKey(20) & 0xff
    #Exit is esc key is pushed 
    if k == 27:
        break

cv2.destroyAllWindows()	
vid.release()


