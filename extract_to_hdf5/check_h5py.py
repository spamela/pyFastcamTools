#!/usr/bin/env python3


import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if (len(sys.argv) > 1):
    filename = sys.argv[1]
else:
    filename = 'data_hdf5_transformed/29406.h5'


f = h5py.File(filename, 'r')
#print(list(f.keys()))
nn = f['n_images'][0]
dset = f['images']
#print(nn)
print(np.shape(dset))
#print(np.dtype(dset))


# --- Check why we have 4 channels
if (False):
    image = dset[0] 
    print(np.shape(image))
    print(np.min(image),np.max(image))
    matplotlib.image.imsave('testing_channels_all.jpg', image)#, cmap='gray')
    image_shape = np.shape(dset)
    if (len(image_shape) > 3):
        for ic in range(image_shape[3]):
            image_size = (image_shape[1],image_shape[2])
            image_tmp = np.zeros( image_size, dtype=np.uint8 )
            for ix in range(image_size[0]):
                for iy in range(image_size[1]):
                    image_tmp[ix][iy] = image[ix][iy][ic]
            matplotlib.image.imsave('testing_channels'+str(ic)+'.jpg', image_tmp)
    # --- Print everything?
    if (False):
        for ix in range(image_size[0]):
            for iy in range(image_size[1]):
                print(image[ix][iy][0],image[ix][iy][1],image[ix][iy][2],image[ix][iy][3])


# --- Print a single image?
if (False):
    img = dset[4000]
    matplotlib.image.imsave('testing_mpi.jpg', img, cmap='gray')

# --- Print every image?
#for i in range(nn):
#    if (i%100 == 0): print("image %d out of %d" % (i,nn) )
#    img = dset[i]
#    i_str = str(i).zfill(6)
#    filename = './tmp/pulse_30008_img'+i_str+'.jpg'
#    matplotlib.image.imsave(filename, img)



