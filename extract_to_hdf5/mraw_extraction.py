#!/usr/bin/env python3


import sys
import io
from PIL import Image
import logging
from pathlib import Path
from multiprocessing import Pool, RawArray
from functools import partial

import numpy as np
import h5py
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyIpx.movieReader import mrawReader
from pyIpx.utils import make_iterable, transform_image

logger = logging.getLogger(__name__)
logger.propagate = False
verbose = False
process_verbose = True









# -------------------
# --- MPI Subroutines
# -------------------


# --- Disctionary pointer for mpi shared data
shared_data_pointer = {}



# --- Initialise shared data buffer with a copy of the camera data
def init_worker(data, data_shape):
    shared_data_pointer['data'] = data
    shared_data_pointer['data_shape'] = data_shape




# --- Camera data extraction into images, to be run in parallel
def extract_image(data_broadcast,i_img):
    # --- Gather shared data
    data = np.frombuffer(shared_data_pointer['data']).reshape(shared_data_pointer['data_shape'])
    # --- Take the frame we want to extract
    frame_extract = data[i_img]
    # --- Retrieve broadcast data
    n_beg       = data_broadcast['n_beg']
    n_end       = data_broadcast['n_end']
    extract_all = data_broadcast['extract_all']
    # --- Print MPI progression
    if (process_verbose and (i_img%100 == 0) ):
        print('mraw image extraction %d out of (%d-%d)' % (i_img,n_beg,n_end) ) ; sys.stdout.flush()
    # --- Print data info
    if (verbose): 
        print(data.shape)
        print(frame_extract)
        print('Plotting data...')
        sys.stdout.flush()
    # --- Prepare figure to be extracted
    fig, ax = plt.subplots()
    ax.imshow(frame_extract, origin='upper', interpolation='none', cmap='gray')
    ax.axes.set_axis_off()
    plt.tight_layout()
    plt.axis('off')
    # --- Save image and exit
    if (extract_all):
        # --- Save image as array
        io_buffer = io.BytesIO()
        plt.savefig(io_buffer, bbox_inches='tight',pad_inches = 0)
        io_buffer.seek(0)
        img = Image.open(io_buffer)
        if (img.mode == 'RGBA'): img = img.getchannel(0) # just one channel in gray scale (less disk space)
        np_format = np.asarray(img)
        np_format = np_format.astype('uint8') # to save memory
        plt.close()
        return np_format
    else:
        image_output = 'tmp_image.jpg'
        plt.savefig(image_output, bbox_inches='tight',pad_inches = 0)
        plt.close()
        # --- Debug: print HDF5 as well (NEEDS TO COMMENT PREVIOUS LINE OF plt.close !!!)
        #io_buffer = io.BytesIO()
        #plt.savefig(io_buffer, bbox_inches='tight',pad_inches = 0)
        #io_buffer.seek(0)
        ##np_format = np.asarray(Image.open(io_buffer))
        #img = Image.open(io_buffer)
        #print('mode???',img.mode)
        #img = img.getchannel(0)
        #np_format = np.asarray(img)
        #plt.close()
        #filename = './tmp_channels.h5'
        #n_img = 1
        #with h5py.File(filename, 'w') as h5file: 
        #    h5file.create_dataset('n_images', data=[n_img])
        #    h5file.create_dataset('images', data=[np_format])
        return 0
  









# ---------------------------
# --- Camera MRAW Subroutines
# ---------------------------








def get_mraw_file_info(path_fn, transforms=()):
    """Get meta data from mraw file

    :param path_fn:     File path format string for a set of mraw files eg .../C001H001S0001-{{n:02d}}.mraw where n is
                        the mraw file index
    :param transforms:  List of transformation to apply to returned movie data eg ('transpose', 'reverse_y')
    :return: movie_meta - dictionary of movie meta data
    """
    if (verbose): print('  Starting meta data')
    movie_meta = {'movie_format': '.mraw'}
    mraw_files = pd.DataFrame({'StartFrame': []})
    # Get frame range in each mraw file
    if (verbose): print('  Getting frame range...')
    n = 0
    path_save = ''
    while Path(str(path_fn).format(n=n)).is_file():
        if (verbose): print('    Inside loop: file ',str(path_fn).format(n=n))
        if (str(path_fn).format(n=n) == path_save): break
        path_save = str(path_fn).format(n=n)
        vid = mrawReader(filename=path_fn.format(n=n))
        header = vid.file_header
        start_frame = int(header['StartFrame'].strip())
        mraw_files.loc[n, 'StartFrame'] = start_frame
        for key in ['TotalFrame', 'StartFrame']:  # 'OriginalTotalFrame', 'CorrectTriggerFrame', 'ZeroFrame']:
            mraw_files.loc[n, key] = int(header[key].strip())
        # Get time ranges for each file
        mraw_files.loc[n, 'StartTime'] = np.round(vid.set_frame_number(0).read()[2]['time_stamp'], decimals=6)
        mraw_files.loc[n, 'EndTime'] = np.round(
            vid.set_frame_number(int(header['TotalFrame'].strip())).read()[2]['time_stamp'], decimals=6)
        if (len(mraw_files) > 1):
            if (mraw_files.loc[n, 'StartTime'] < mraw_files.loc[n - 1, 'EndTime']):
                # If same start time as previous file, add on to previous end time
                mraw_files.loc[n, 'StartTime'] += mraw_files.loc[n - 1, 'EndTime']
            if (mraw_files.loc[n, 'EndTime'] <= mraw_files.loc[n - 1, 'EndTime']):
                # If same start time as previous file, add on to previous end time
                mraw_files.loc[n, 'EndTime'] += mraw_files.loc[n - 1, 'EndTime']
        vid.release()
        n += 1
    assert n > 0, 'No mraw files read'
    # Calcuate time interval stored in each file
    if (verbose): print('  Getting time interval...')
    mraw_files.loc[:, 'TotalTime'] = mraw_files.loc[:, 'EndTime'] - mraw_files.loc[:, 'StartTime']
    # Mraw movie frames don't start at zero so record start frame and offset by it so start at 0
    if (verbose): print('  Getting time index offset...')
    movie_meta['frame0'] = int(mraw_files.loc[0, 'StartFrame'])
    mraw_files.loc[:, 'StartFrame'] -= movie_meta['frame0']
    mraw_files.loc[:, 'EndFrame'] = mraw_files.loc[:, 'StartFrame'] + mraw_files.loc[:, 'TotalFrame'] - 1
    # Reset column types and order
    if (verbose): print('  Setting columns types...')
    mraw_files = mraw_files.astype({'StartFrame': int, 'EndFrame': int, 'TotalFrame': int})
    mraw_files = mraw_files[['StartFrame', 'EndFrame', 'TotalFrame', 'StartTime', 'EndTime', 'TotalTime']]
    # Get additional meta data eg frame shape
    if (verbose): print('  Getting additional info...')
    movie_meta['mraw_files'] = mraw_files
    movie_meta['mraw_header'] = header
    movie_meta['frame_range'] = [int(mraw_files.loc[0, 'StartFrame']), int(mraw_files.loc[n - 1, 'EndFrame'])]
    movie_meta['t_range'] = [mraw_files.loc[0, 'StartTime'], mraw_files.loc[n - 1, 'EndTime']]
    frame_shape = (int(header['ImageHeight'].strip()), int(header['ImageWidth'].strip()))
    frame_shape = transform_image(np.empty(frame_shape), transforms).shape
    movie_meta['frame_shape'] = frame_shape
    movie_meta['trasforms'] = transforms
    movie_meta['fps'] = int(header['RecordRate(fps)'].strip())
    logger.info('Readimg mraw movie file {}:\n{}'.format(path_fn, mraw_files))
    if (verbose): print('  Finished, returning from meta data info...')
    return movie_meta




def get_mraw_file_number(movie_meta, n):
    mraw_files = movie_meta['mraw_files']
    frame_range = movie_meta['frame_range']

    mask = (mraw_files['StartFrame'] <=  n) & (n <= mraw_files['EndFrame'])
    if np.sum(mask) != 1:
        raise ValueError('Frame number {} is outside of mraw file frame range {}'.format(n, frame_range))
    file_number = mraw_files.loc[mask].index.values[0]
    file_info = mraw_files.loc[file_number, :].to_dict()
    return file_number, file_info




def read_mraw_movie(fn_mraw, n=None, transforms=()):
    # Initialise array for data to be read into
    movie_meta = get_mraw_file_info(fn_mraw, transforms=transforms)
    if n is None:
        # Loop over frames from start frame, including those in the frame set
        # TODO: change to only loading files that haven't already been loaded?
        frames = np.arange(*movie_meta['frame_range'])
    else:
        frames = make_iterable(n)
    n = frames[0]
    end = frames[-1]
    i_data = 0
    data = np.zeros((len(frames), *movie_meta['frame_shape']))
    logger.debug('Reading {} frames from mraw movie file'.format(len(frames)))

    file_number, file_info = get_mraw_file_number(movie_meta, n=n)
    vid = mrawReader(filename=fn_mraw.format(n=file_number))
    vid.set_frame_number(n - file_info['StartFrame'])
    if (verbose): print('Starting data extraction...')
    while n <= end:
        # If reached end of current file, switch to next file
        if n > file_info['EndFrame']:
            vid.release()
            file_number, file_info = get_mraw_file_number(movie_meta, n=n)
            vid = mrawReader(filename=fn_mraw.format(n=file_number))
            vid.set_frame_number(n - file_info['StartFrame'])
        if n in frames:
            # frames are read with 16 bit dynamic range, but values are 10 bit!
            ret, frame, header = vid.read(transforms=transforms)
            data[i_data, :, :] = frame
            # self._meta.loc[n, 'set'] = True
            i_data += 1
        else:
            vid._skip_frame()
            # ret, frame, header = vid.read(transforms=self._transforms)
        n += 1
    vid.release()
    return data






# ----------------
# --- Main Program
# ----------------

def main():
    # --- Get input arguments
    if (verbose): print('Starting...')
    if (len(sys.argv) > 6):
        pulse = int(sys.argv[1])
        base_directory = sys.argv[2]
        sub_directory = sys.argv[3]
        video_file = sys.argv[4]
        extract_all = int(sys.argv[5])
        if (extract_all == 0):
            extract_all = False
        else:
            extract_all = True
        n_MPI = int(sys.argv[6])
    else:
        pulse = 29852
        base_directory = '/marconi_work/FUA36_MHD/spamela/MAST_fast_camera/all_SA1'
        sub_directory = 'C001H001S0001'
        video_file = 'C001H001S0001-04.mraw'
        extract_all = False
        n_MPI = 1
    # --- Film info
    fn_mraw = f'{base_directory}/{pulse}/{sub_directory}/{video_file}'
    transforms = ('transpose', 'reverse_y')

    # --- Get movie info
    if (verbose): print('Getting meta data...')
    meta_data = get_mraw_file_info(fn_mraw, transforms=transforms)
    if (verbose): print(meta_data['mraw_files'])

    # --- Read the whole movie
    n = None
    if (verbose): print('Reading data...')
    data = read_mraw_movie(fn_mraw, n=n, transforms=transforms)
    if (verbose): print(data.shape)

    # --- Share data between processes (faster because "data" can be very large)
    print('Copying data to shared buffer...') ; sys.stdout.flush()
    data_shape  = data.shape
    shared_data = RawArray('d', data_shape[0] * data_shape[1] * data_shape[2])
    shared_np   = np.frombuffer(shared_data).reshape(data_shape)
    np.copyto(shared_np, data)

    # --- Extract all images or just sample in the middle?
    n_mid = int( float(len(data)) / float(2) )
    if (extract_all):
        n_beg = 0
        n_end = len(data)
    else:
        n_beg = n_mid
        n_end = n_mid+1
    # --- Data sent to each process (can't be too large, so can't include camera "data" itself)
    data_broadcast = {}
    data_broadcast['n_beg']       = n_beg
    data_broadcast['n_end']       = n_end
    data_broadcast['extract_all'] = extract_all
    # --- Functions specification for mpi processes
    parallel_function = partial(extract_image, data_broadcast)
    parallel_array = [i for i in range(n_beg,n_end)]
    # --- Set up MPI processes
    print('Starting mraw image extraction: total of %d, (%d to %d)' % (n_end-n_beg,n_beg,n_end) ) ; sys.stdout.flush()
    with Pool(processes=n_MPI, initializer=init_worker, initargs=(shared_data, data_shape)) as mpi_pool:
        data_return = mpi_pool.map(parallel_function, parallel_array)
    full_movie = data_return
    print('Finished mraw image extraction!') ; sys.stdout.flush()

    # --- Save movie as HDF5
    if (extract_all):
        filename = './data_tmpworkdir/'+sub_directory+video_file+'.h5'
        n_img = len(full_movie)
        with h5py.File(filename, 'w') as h5file: 
            h5file.create_dataset('n_images', data=[n_img])
            h5file.create_dataset('images', data=full_movie)







# --- Main function execution as a script...        
if __name__ == '__main__':
    main()
