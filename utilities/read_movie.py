#!/usr/bin/env python
from __future__ import print_function
import os.path
from os import path
import sys
from datetime import datetime
import cv2
import numpy as np
from pyFastcamTools.utilities.frame_history import frame_history
from pyFastcamTools.operation.movie_reader import ipx_reader,mraw_reader,imstack_reader
import logging
from subprocess import call,Popen,PIPE


def read_movie(filename,
               Nframes=None,
               stride=1,
               startpos=0,
               endpos=-1,
               verbose=0,
               startframe=None,
               endframe=None,
               starttime=None,
               endtime=100.0,
               transforms=[],
               trigger_time = -0.1,
               save_frames=False,
               frames_directory=None):

    """
    Function to read in a movie file using openCV and store as a frameHistory

    Arguments:
        filename	-> 	name of the movie file to read from
                    OR
                    MAST shot number to read

    keywords:
        Nframes		-> 	Number of frames to read	Default: None, read entire movie
        stride		->	Read frames with a stride	Default: 1, read every frame
        startpos	-> 	Relative position to start	Default: 0, start from beginning of movie
                                reading from, 0 = start,
                                1 = end, 0.xx is xx% through the movie
        endpos		->	Relative position to end 	Default: 1, read until end of movie reading
        verbose         ->      printout progress (=0,1,2)      Default: 0 (no printout)
        transforms	->	Apply any or all of the following transforms to the image data
                        'reverse_x' : reverse the x dimension
                        'reverse_y' : reverse the y dimension
                        'transpose' : transpose the image (flip x and y)

    Example:

        frames = readMove('myMovie.mov',Nframes=100,stride=2,startpos=0.3,endpos=0.5)

        This will read every 2nd frame of a movie from the file 'myMovie.mov' starting from 30% into the movie and
        ending after 100 frames, or when it reaches 50% through, whichever comes first

    """
    # Check movie file actually exists
    if (not path.exists(filename)):
        print('Filename does not exit! Aborting...')
        sys.exit()

    # read frames
    frames = frame_history(descriptor="File: "+filename)

    # Frames output
    if (save_frames):
        if (frames_directory==None):
            now = datetime.now()
            date_and_time = now.strftime("%Y.%m.%d_%H.%M.%S")
            frames_directory = 'frames_output_'+date_and_time
        call('mkdir -p '+frames_directory, shell=True)
        print('Frames will be saved to directory "./'+frames_directory+'/"')

    # deal with ipx, mraw or unspecified formats
    if '.' not in filename or filename.split('.')[-1] == 'ipx' or filename.split('.')[-1] == 'mraw':
        # get start/end frames
        # for ipx format
        if filename.split('.')[-1] == 'ipx':
            vid = ipx_reader(filename=filename)
            
            if startpos is not None and startframe is None:
                startframe = int(startpos*vid.file_header['numFrames'])
            if endpos is not None and endframe is None:
                endframe = int(np.abs(endpos)*vid.file_header['numFrames'])
            elif endframe is -1:
                endframe = int(1.0*vid.file_header['numFrames'])
        # for mraw of unspecified
        else:
            if filename.split('.')[-1] == 'mraw':
                vid = mraw_reader(filename=filename)
            else:
                vid = imstack_reader(directory=filename)
            if startpos is not None and startframe is None:
                startframe = int(startpos*int(vid.file_header['TotalFrame']))
            if endpos is not None and endframe is None:
                print(vid.file_header['TotalFrame'])
                if endpos > -1:
                    endframe = int(endpos*int(vid.file_header['TotalFrame']))
                else:
                    endframe = int(1.0*int(vid.file_header['TotalFrame']))

        # number of frames
        if Nframes is None:
            Nframes = endframe - startframe + 1
        
        # starting frame
        vid.set_frame_number(startframe)
        if starttime is not None:
            vid.set_frame_time(starttime)
        
        # loop over frames
        N = 0
        for i in np.arange(Nframes*stride):
            print( 'Reading frame {} out of {}\r'.format(i, len(np.arange(Nframes*stride))), end="")
            ret,frame,header = vid.read(transforms=transforms)
            # scaling for pixel bit size
            frame_tmp = frame[::-1].T
            frame_scale = 255. / frame_tmp.max()
            frame = np.array(frame_scale * frame_tmp, dtype=np.uint8)
            if ret and (not N + startframe > endframe)  and (not float(header['time_stamp']) > endtime):
                if i % stride == 0:
                    if (verbose > 0): print("Reading movie frame {} at time {}".format(header['frame_number'],header['time_stamp']))
                    frames.add_frame(frame,header['time_stamp'],header['frame_number'])
                    if (save_frames):
                        cv2.imwrite(frames_directory+"/frame%d.jpg" % N, frame)
                    N += 1
            else:
                break
        pass
    #Reading a non-ipx file with openCV
    else:

        vid = cv2.VideoCapture(filename)
        #Set the starting point of the video
        vid.set(2,startpos)

        times = []
        frameNos = []

        for i in np.arange(Nframes*stride):
            print( 'Reading frame {} out of {}\r'.format(i, len(np.arange(Nframes*stride))), end="")
            ret,frame = vid.read()
            if ret and not vid.get(2)>endpos:
                #Succesful read
                if i%stride==0:
                    #Only take every stride frames
                    frames.add_frame(frame,vid.get(0),vid.get(1))
                    if (verbose > 0): print("Reading movie frame "+str(i), end='\r')
            else:
                break
    vid.release()
    return frames


