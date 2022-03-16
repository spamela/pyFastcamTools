#!/usr/bin/env python3
import sys
import os
import io
import glob
import h5py
import numpy as np
from PIL import Image, ImageOps
from multiprocessing import Pool, RawArray
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




# --------------
# --- Input data
# --------------

# --- Extract only one image per movie
extract_sample   = False
# --- Extract every image for each movie
extract_all      = False
# --- Post-process images (eg. resize/rotate) note rotation is automatic
transform_images = True
# --- Resize image
resize_sample    = True
resize_pixels    = 128 #64
# --- Convert tif images to jpg images
convert_tif2jpg  = False
# --- Convert png images to jpg images
convert_png2jpg  = False
# --- NOTE: Do not use convert_tif2jpg and convert_png2jpg together.
# ---       The main reason we need to convert tif and png is because
# ---       the python image library PIL is very unreliable depending
# ---       on the image format. For example, you can have the same
# ---       image in jpg and png, and pply the same PIL transformation
# ---       but the output will be different! The other option was CV2
# ---       (opencv-python), however, even though the transformation was
# ---       reliable for all formats, CV2 behaves very badly in parallel
# ---       processes, and leads to memory issues and MPI-freeze...
# ---       The last option was to use bash command "convert", but this
# ---       can be slower than staying in python, and the command is not
# ---       always available on parallel clusters...
# ---       Hence, PIL is still the best, but should always be used with jpg...

# --- Get progression of mpi processes when transforming images
process_verbose = True
# --- Standard verbose
verbose = True
# --- Only get info about content of pulse folders
info_only = False





# ---------------
# --- Subroutines
# ---------------


# --- MPI: Camera data extraction into images, to be run in parallel
def open_image_into_array(data_broadcast,i_img):
    # --- Retrieve broadcast data
    n_total = data_broadcast['n_total']
    all_files = data_broadcast['all_files']
    filename = all_files[i_img]
    # --- Print MPI progression
    if (process_verbose and (i_img%100 == 0) ):
        print('mraw image extraction %d out of %d' % (i_img,n_total) ) ; sys.stdout.flush()
    # --- Open image into array
    image = np.asarray(Image.open(filename))
    image = image.astype('uint8') # to save memory
    return image


# --- MPI: Disctionary pointer for mpi shared data
shared_data_pointer = {}

# --- MPI: Initialise shared data buffer with a copy of the camera data
def init_worker(data, data_shape):
    shared_data_pointer['data'] = data
    shared_data_pointer['data_shape'] = data_shape

# --- MPI: Image transformation (rotation/resize) to be executed in parallel
def convert_image(data_broadcast, i_img):
    # --- Get mpi-broadcast data
    n_img          = data_broadcast['n_img']
    single_sample  = data_broadcast['single_sample']
    transformation = data_broadcast['transformation']
    pulse          = data_broadcast['pulse']
    Extract_sample = data_broadcast['extract_sample']

    # --- Get data from buffer
    if (Extract_sample):
        # --- input/output images
        image_in = single_sample
        image_out = image_in
    else:
        # --- Gather shared data
        data = np.frombuffer(shared_data_pointer['data']).reshape(shared_data_pointer['data_shape'])
        # --- Take the frame we want to extract
        frame_extract = data[i_img]

    # --- MPI progress printout
    if (process_verbose and (i_img%100 == 0) ):
        print('image conversion %d out of %d' % (i_img,n_img) ) ; sys.stdout.flush()

    # --- Simple conversion for tif and png format
    if ( (transformation == "tif2jpg") or (transformation == "png2jpg") ):
        os.system('convert '+image_in+' '+image_out)
        return True

    # --- Image transformation: Load image
    if (Extract_sample):
        image = Image.open(image_in)
    else:
        image = Image.fromarray(frame_extract.astype('uint8'))

    # --- Rotate
    if (transformation['rotate'] != ''):
        rotation = int(transformation['rotate'])
        image = image.rotate(rotation, expand=True)

    # --- Resize
    if (transformation['resize'] != ''):
        desired_size = int(transformation['resize'])
        old_size = image.size
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        image = image.resize(new_size, Image.ANTIALIAS)
        #new_image = Image.new("RGB", (desired_size, desired_size))
        new_image = Image.new("L", (desired_size, desired_size))
        new_image.paste(image, ((desired_size-new_size[0])//2,
                                (desired_size-new_size[1])//2))
        image = new_image

    # --- Save image
    if (Extract_sample):
        image.save(image_out)
        return 0
    else:
        image = np.asarray(image)
        image = image.astype('uint8')
        return image






# --- Sanity check for user-input
def user_input_sanity_check():

    # --- Sanity check
    if (convert_tif2jpg):
        if (transform_image or resize_sample):
            sys.exit('cannot convert tif2jpg at the same time as rotation/resizing.\naborting...')
            
        if (not extract_all):
            sys.exit('cannot convert tif2jpg without extract_all option.\naborting...')

    # --- Sanity check
    if (convert_png2jpg):
        if (transform_images or resize_sample):
            sys.exit('cannot convert png2jpg at the same time as rotation/resizing.\naborting...')
        if (not extract_all):
            sys.exit('cannot convert png2jpg without extract_all option.\naborting...')

    # --- Sanity check
    if (convert_tif2jpg and convert_png2jpg):
        sys.exit('cannot convert tif2jpg and png2jpg at the same time, choose one.\naborting...')

    # --- Sanity check
    if (extract_sample and extract_all):
        sys.exit('cannot extract sample and all data at the same time, choose one.\naborting...')





# --- Create necessary folders
def create_working_folders():
    # --- The raw data is needed obviously, it should be in pulse sub-folders inside the ./data_raw/ folder
    if (not os.path.isdir('data_raw')):
        sys.exit('Cannot find the raw data! Should be in ./data_raw/ folder.\naborting...')
    # --- If we extract a single file per pulse
    os.system('mkdir -p data_samples')
    # --- The raw data will be extracted to HDF5 (one file per pulse)
    os.system('mkdir -p data_hdf5')
    # --- The HDF5 data can then be transformed (rotate/resize)
    os.system('mkdir -p data_hdf5_transformed')
    # --- We need a folder to work in
    os.system('mkdir -p data_tmpworkdir')
    os.system('rm data_tmpworkdir/*')





# --- Get transformation data (depends on pulse)
def get_transformation_per_pulse():
    transform = {}
    for pulse in range(20000,35000):
        if (convert_tif2jpg):
            transform[pulse] = 'tif2jpg'
        elif (convert_png2jpg):
            transform[pulse] = 'png2jpg'
        else:
            # --- We specify a rotation and a resize
            transform[pulse] = {}
            transform[pulse]['rotate'] = ''
            transform[pulse]['resize'] = ''
            # --- Rotation depends on pulse!!!
            if ( (20000 < pulse) and (pulse <= 29190) ): transform[pulse]['rotate'] = '+90'
            if ( (29190 < pulse) and (pulse <= 29543) ): transform[pulse]['rotate'] = '-90'
            if ( (29543 < pulse) and (pulse <= 29732) ): transform[pulse]['rotate'] = '+180'
            # --- Resize should be the same for all
            if (resize_sample):                          transform[pulse]['resize'] = str(resize_pixels)
    return transform




# --- Pulse folder preprocessing: we check what's inside a folder, .jpg? .mraw? .png? in multiple subfolders? etc.
def pulse_folder_info(pulse):
    # --- Ignore this pulse?
    ignore_pulse = False
    # --- Get list of all subfolders
    all_subfolders = sorted(glob.glob('./data_raw/'+str(pulse)+"/*/"))
    all_subfolders.append('./data_raw/'+str(pulse)+"/")
    n_subfolders = len(all_subfolders)
    # --- Initialise: Number of files found in each subfoder
    n_found = [0] * n_subfolders
    # --- Initialise: Total number of files in all subfolders
    n_total = 0
    # --- Initialise: Format of files found in each subfoder
    formats = [""] * n_subfolders
    # --- Initialise: Number of non-empty subfolders
    n_subs = 0
    # --- Initialise: Number of files in largest subfolder
    n_max = 0
    # --- Initialise: Index of subfolder from which we take a sample (the largest subfolder)
    i_sample = 0
    # --- Initialise: Copy of the data sample
    sample_data = ''
    # --- Initialise: Copy of the whole data set
    all_data = ''
    # --- Initialise: Copy of the files we need
    copy_files = ''
    # --- Go into each subfolder
    for i_sub in range(len(all_subfolders)):
        subfolder = all_subfolders[i_sub]
        # --- Check for all known formats directly here
        mraw_files = sorted(glob.glob(subfolder+"*.mraw"))
        jpg_files  = sorted(glob.glob(subfolder+"*.jpg"))
        tif_files  = sorted(glob.glob(subfolder+"*.tif"))
        png_files  = sorted(glob.glob(subfolder+"*.png"))
        # --- Number of files
        n_files = len(mraw_files) + len(jpg_files) + len(tif_files) + len(png_files)
        n_found[i_sub] = n_files
        n_total = n_total + n_files
        if (n_files > 0): n_subs = n_subs + 1
        # --- Record each present format and make a copy of the data
        present_formats = []
        n_present = []
        if (len(mraw_files) != 0):
            present_formats.append("mraw")
            formats[i_sub] = "mraw"
            n_present.append(len(mraw_files))
            copy_files = mraw_files
        if (len(jpg_files) != 0):
            present_formats.append("jpg")
            formats[i_sub] = "jpg"
            n_present.append(len(jpg_files))
            copy_files = jpg_files
        if (len(tif_files) != 0):
            present_formats.append("tif")
            formats[i_sub] = "tif"
            n_present.append(len(tif_files))
            copy_files = tif_files
        if (len(png_files) != 0):
            present_formats.append("png")
            formats[i_sub] = "png"
            n_present.append(len(png_files))
            copy_files = png_files
        # --- Check for data duplication (ie. multiple formats)
        if (verbose and (len(present_formats) > 1)):
            print("Warning on pulse %s : duplication found:" % (str(pulse))) ; sys.stdout.flush()
            for i in range(len(present_formats)):
                print("                      %s %d" % (present_formats[i],n_present[i]))
        # --- Make a copy of the data sample for the largest subfolder
        if (n_files > n_max):
            i_sample = i_sub
            n_max = n_files
            i_choose = int(float(len(copy_files))/2.0)
            sample_data = copy_files[i_choose]
            # --- When extracting all data, we want all subfolders...
            if (extract_all): all_data = subfolder
    # --- Sanity check
    file_format = formats[i_sample]
    if (extract_sample  and (sample_data == '') ): ignore_pulse = True
    if (extract_all     and (all_data    == '') ): ignore_pulse = True
    if (convert_tif2jpg and (file_format != "tif") ): ignore_pulse = True
    if (convert_png2jpg and (file_format != "png") ): ignore_pulse = True
    # --- Info warning
    if (verbose):
        if (n_subs > 1):
            print("Warning on pulse %s : data in multiple subfolders!" % (str(pulse))) ; sys.stdout.flush()
            for i_sub in range(n_subs):
                print("                      %s  %d  %s" % (all_subfolders[i_sub],n_found[i_sub],formats[i_sub])) ; sys.stdout.flush()
            print("You need to make sure everything is in a single folder before extracting all data!!!") ; sys.stdout.flush()
        # --- Warning if there is no data at all...
        if (n_total == 0):
            print("PULSE %s: NOTHING!" % (str(pulse))) ; sys.stdout.flush()
        print("PULSE %s: %d files with format .%s" % (str(pulse),n_max,file_format)) ; sys.stdout.flush()

    # --- Return necessary data
    return file_format,sample_data,n_max,ignore_pulse,copy_files










# ----------------
# --- Main Program
# ----------------




# --- Main program
def main():

    # --- MPI process specified from arguments
    if (len(sys.argv) > 1):
        n_MPI = int(sys.argv[1])
        if (len(sys.argv) > 2):
            pulse_folder = sys.argv[2]
        else:
            pulse_folder = ''
    else:
        n_MPI = 1
        pulse_folder = ''

    # --- Sanity check for user-input
    user_input_sanity_check()

    # --- Make sure all folders are here
    create_working_folders()

    # --- Transformation info
    transform = get_transformation_per_pulse()

    # --- Get list of all folders here
    if (pulse_folder == ''):
        all_folders = sorted(glob.glob("./data_raw/*"))
    else:
        all_folders = [pulse_folder]
    for full_folder in (all_folders):

        # --- Get pulse number (what we expect at least)
        folder_name = full_folder.split("/")
        folder = folder_name[len(folder_name)-1]
        # --- Check that folder has the name of pulse number
        try:
            pulse = int(folder)
        except:
            print("this folder: \"%s\" is not a pulse..." % (folder)) ; sys.stdout.flush()
            continue

        # --- Apply to some specific pulses?
        #if (pulse != 29272): continue # jpg example in subfolder
        #if (pulse != 29181): continue # jpg example in main folder
        #if (pulse != 29943): continue # mraw example in single folder (12 files)
        #if (pulse != 30008): continue # mraw example in single folder (3 files)
        #if ( (pulse != 29272) and (pulse != 29181) and (pulse != 30008) ): continue

        # --- MPI bug!!!
        #if (pulse != 29545): continue # strange problem with 29545...
        #if (pulse <= 29545): continue # strange problem with 29545...
        #if (pulse <= 29548): continue # strange problem with 29545...
        #if (pulse != 29548): continue # strange problem with 29545...

        #if (pulse != 29538): continue # example with 1 channel
        #if (pulse != 29547): continue # example with 4 channel
        #if ( (pulse != 29547) and (pulse != 29538) ): continue
        #if (pulse != 29547): continue

        # --- Apply to all known pulses?
        if ( (29000 < pulse) and (pulse < 41000) ):

            # --- Get pulse folder information
            file_format,sample_data,n_max,ignore_pulse,copy_files = pulse_folder_info(pulse)
            if (ignore_pulse): continue
            if (info_only):    continue

            # --- Extract movie data
            if (extract_all or extract_sample):
                if (file_format == "mraw"):
                    # --- Name of images
                    image_name = './tmp_image.jpg'
                    image_sample = './data_samples/'+str(pulse)+'.jpg'
                    # --- Depending on sample or full extraction, we loop over all folders or a single movie file
                    if (extract_all):
                        all_mraw = copy_files
                        extract_flag = str(1)
                    else:
                        all_mraw = [sample_data]
                        extract_flag = str(0)
                    # --- Loop over all mraw files (or just one sample)
                    for data_tmp in (all_mraw):
                        # --- Name/Path of movie file
                        film_name = data_tmp.split("/")
                        base_directory = os.getcwd() + "/data_raw/"
                        sub_directory = film_name[len(film_name)-2]
                        if (sub_directory == str(pulse)): sub_directory = '.'
                        film_name = film_name[len(film_name)-1]
                        # --- Extraction command
                        # --- Note: for extract_flag=0, code will write just one image named ./tmp_image.jpg
                        # ---       for extract_flag=1, code will extract all images in tmp directory named ./data_tmpworkdir/
                        my_command = './mraw_extraction.py '+str(pulse)+' '+base_directory+' '+sub_directory+' '+film_name+' '+extract_flag+' '+str(n_MPI)
                        if (verbose): print("mraw film: %s --- command: %s" % (film_name,my_command)) ; sys.stdout.flush()
                        # --- Execute command (note: maybe we just include mraw_extraction code as a subroutine eventually...?)
                        if (extract_sample or extract_all): os.system(my_command)
                        # --- For samples, we just copy the single image
                        if (extract_sample): os.system('mv '+image_name+' '+image_sample)
                    # --- For full data, we loop over each h5-file and merge them into one
                    if (extract_all):
                        if (verbose): print("merging all HDF5 together...") ; sys.stdout.flush()
                        # --- Initialise all images
                        n_all_img  = 0
                        all_images = []
                        # --- Get list of all HDF5 files to merge together
                        all_files  = sorted(glob.glob('./data_tmpworkdir/*.h5'))
                        for filename in (all_files):
                            h5file = h5py.File(filename, 'r')
                            n_img  = h5file['n_images'][0]
                            images = h5file['images']
                            for i_img in range(n_img):
                                all_images.append(images[i_img])
                            n_all_img = n_all_img + n_img
                        # --- Save movie as HDF5
                        filename = './data_hdf5/'+str(pulse)+'.h5'
                        with h5py.File(filename, 'w') as h5file: 
                            h5file.create_dataset('n_images', data=[n_all_img])
                            h5file.create_dataset('images', data=all_images)
                        # --- Cleanup
                        os.system('rm ./data_tmpworkdir/*.h5')
                # --- Extract image data
                elif ( (file_format == "jpg") or (file_format == "tif") or (file_format == "png") ):
                    # --- We make a copy of the sample
                    image_name = sample_data
                    image_sample = './data_samples/'+str(pulse)+'.'+file_format
                    if (extract_sample): os.system('cp '+image_name+' '+image_sample)
                    # --- Export data to HDF5
                    if (extract_all):
                        if (verbose): print("extracting files as HDF5...") ; sys.stdout.flush()
                        # --- Get list of files to copy and those we've already copied
                        all_files  = np.asarray(copy_files)
                        n_all_img = len(all_files)
                        # --- Data sent to each process (can't be too large, so can't include camera "data" itself)
                        data_broadcast = {}
                        data_broadcast['n_total'] = n_all_img
                        data_broadcast['all_files'] = all_files
                        # --- Functions specification for mpi processes
                        parallel_function = partial(open_image_into_array, data_broadcast)
                        parallel_array = [i for i in range(n_all_img)]
                        # --- Set up MPI processes
                        print('Reading %d images as arrays...' % (n_all_img) ) ; sys.stdout.flush()
                        with Pool(n_MPI) as mpi_pool:
                            data_return = mpi_pool.map(parallel_function, parallel_array)
                        all_images = data_return
                        print('Finished image extraction!') ; sys.stdout.flush()
                        # --- Save movie as HDF5
                        filename = './data_hdf5/'+str(pulse)+'.h5'
                        with h5py.File(filename, 'w') as h5file: 
                            h5file.create_dataset('n_images', data=[n_all_img])
                            h5file.create_dataset('images', data=all_images)


            # --- Image post-processing (rotate/resize/convert)
            if (transform_images or convert_tif2jpg or convert_png2jpg):
                # --- Check if pulse requires transformation
                if (transform[pulse] != ''):
                    format_tmp = file_format
                    if (file_format == 'mraw'): format_tmp = 'jpg'
                    # --- With samples, only one file to read, easy
                    single_sample  = ''
                    if (extract_sample):
                        single_sample  = './data_samples/'+str(pulse)+'.'+format_tmp
                        n_all_img = 1
                        all_images = [0] # won't actually be used
                    # --- If we just extracted the data, it's already in memory, otherwise, need to read it now
                    if ( (not extract_all) and (not extract_sample) ):
                        filename = './data_hdf5/'+str(pulse)+'.h5'
                        if (not os.path.exists(filename)):
                            print('Warning, you did not extract the data for this pulse: %d' % (pulse))
                            print('Ignoring')
                            continue
                        h5file = h5py.File(filename, 'r')
                        n_all_img  = h5file['n_images'][0]
                        all_images = h5file['images']
                    # --- Share data between processes (faster because "data" can be very large)
                    print('Copying data to shared buffer...') ; sys.stdout.flush()
                    data_shape  = np.shape(all_images)#all_images.shape
                    full_size = 1
                    for i in range(len(data_shape)):
                        full_size = full_size * data_shape[i]
                    print('allocating buffer memory of size %d',full_size) ; sys.stdout.flush()
                    shared_data = RawArray('l', full_size)
                    print('allocation successful') ; sys.stdout.flush()
                    shared_np   = np.frombuffer(shared_data).reshape(data_shape)
                    print('copy successful') ; sys.stdout.flush()
                    np.copyto(shared_np, all_images)
                    print('copy successful') ; sys.stdout.flush()
                    # --- Prepare data broadcast for mpi execution
                    data_broadcast = {}
                    data_broadcast['n_img']          = n_all_img
                    data_broadcast['single_sample']  = single_sample     
                    data_broadcast['transformation'] = transform[pulse]
                    data_broadcast['pulse']          = pulse         
                    data_broadcast['extract_sample'] = extract_sample
                    # --- Prepare parallel functions
                    parallel_function = partial(convert_image, data_broadcast)
                    parallel_array = [i for i in range(n_all_img)]
                    # --- Set up MPI processes
                    with Pool(processes=n_MPI, initializer=init_worker, initargs=(shared_data, data_shape)) as mpi_pool:
                        data_return = mpi_pool.map(parallel_function, parallel_array)
                    new_images = data_return
                    print('Finished image extraction!') ; sys.stdout.flush()
                    # --- Save movie as HDF5
                    if (not extract_sample):
                        print('Writing HDF5 file...') ; sys.stdout.flush()
                        filename = './data_hdf5_transformed/'+str(pulse)+'.h5'
                        with h5py.File(filename, 'w') as h5file: 
                            h5file.create_dataset('n_images', data=[n_all_img])
                            h5file.create_dataset('images', data=new_images)


            # --- Flush out forgotten prints...
            sys.stdout.flush()

    # --- Finished
    print('Main program finished.') ; sys.stdout.flush()





# --- Main function execution as a script...        
if __name__ == '__main__':
    main()

