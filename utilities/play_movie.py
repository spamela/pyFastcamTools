#!/usr/bin/env python

"""
Simple program to read in and animate a movie

Nick Walkden, May 2015
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
#Create a frame generator
#vid = cv2.VideoCapture('/home/nwalkden/python_tools/pySynthCam/29852_movie.mpg')
#vid = cv2.VideoCapture('Test_mraw.mraw')
from pyFastcamTools.operation.movie_reader import mraw_reader 
#from pyFastcamTools.background_subtractor import backgroundSubtractorMin


cv2.namedWindow('Video',cv2.WINDOW_NORMAL)



skip = 1

#moviefile = '/Volumes/SAMSUNG/SA1/29576/C001H001S0001/C001H001S0001-03.mraw'
#moviefile = '29840/C001H001S0001/C001H001S0001-00.mraw'
moviefile = '29786/C001H001S0001/C001H001S0001-04.mraw'
vid = mraw_reader(moviefile)
vid.set_frame_number(3200)
Nbgsub = 20
gamma = 1.0
use_bgsub = False
#bgsub = backgroundSubtractorMin(Nbgsub)
#for i in np.arange(Nbgsub):
ret,frame,_ = vid.read()
#	if use_bgsub: frame = bgsub.apply(frame)


while(1):
	#for i in np.arange(skip):
    ret,frame,header = vid.read()
    print(header)
	#if use_bgsub: frame = bgsub.apply(frame)	
    #cv2.imshow('Video',50*frame[::-1].T)
    frames = frame[::-1].T
    frames_scale = 255. / frames.max()
    frames = np.array(frames_scale * frames, dtype=np.uint8)
    cv2.imshow('Video',frames)
    #im.set_data(frame.T[::-1]**gamma)
    #plt.draw()
	#Display at 25fps (40ms = 1/25)
    k = cv2.waitKey(20) & 0xff
	
	#Exit is esc key is pushed 
    if k == 27:
        break

cv2.destroyAllWindows()	
vid.release()


#import pyAutoGit as git
#

#A = git.init()

#A.commit()

#def play_movie_frames(frames):
#    cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
#    for frame in frames:
#    	#for i in np.arange(skip):
#        #ret,frame,header = vid.read()
#        #print(header)
#    	##if use_bgsub: frame = bgsub.apply(frame)	
#        cv2.imshow('Video',20*frame.T[::-1])
#        #im.set_data(frame.T[::-1]**gamma)
#        #plt.draw()
#    	#Display at 25fps (40ms = 1/25)
#        k = cv2.waitKey(20) & 0xff
#	
#    	#Exit is esc key is pushed 
#        if k == 27:
 #           break
    
