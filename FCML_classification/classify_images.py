#!/usr/bin/env python3


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import sys
import os
import h5py
from PIL import Image, ImageOps
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool, RawArray
from functools import partial
import time




# --- Global variables
use_easy_model = False
verbose = True
pyplots = 1 # 0: no plots, 1: save figures, 2: show figures


# --- Location of Mast fast-camera images
fast_cam_path = '/m100_work/FUAC6_UKAEA_ML/spamela/MAST_fast_camera/data_hdf5_transformed'
fast_cam_pulse_class = '/m100_work/FUAC6_UKAEA_ML/spamela/MAST_fast_camera/pulse_classification_simple.txt'
fast_cam_pulse_midplane_details = '/m100_work/FUAC6_UKAEA_ML/spamela/MAST_fast_camera/pulse_classification_midplane.txt'









# --- Print only if verbose
def vprint(stuff):
    if (verbose): print(stuff)











# --------------------------------------------------
# --- Section: Plots
# --------------------------------------------------




# --- Plot multiple images in a table
# --- Usage for a single image:
# plot_image_table(1,1,[img],['random_name'],[0],'folder/example.png')
def plot_image_table(num_rows,num_cols,images,classes,labels,filename):
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(classes[labels[i]])
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight',pad_inches = 0.2)
    if (pyplots == 2):
        plt.show()
    plt.close()


# --- Plot multiple images in a table together with prediction/accuracy
def plot_image_table_accuracy(num_rows,num_cols,images,classes,labels,predictions,filename):
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], labels, images, classes)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], labels)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight',pad_inches = 0.2)
    if (pyplots == 2):
        plt.show()
 


# --- Old plot function
def plot_image(i, predictions_array, true_label, img, class_names):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

# --- Old plot function
def plot_value_array(i, predictions_array, true_label):
  n_labels = len(predictions_array)
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(n_labels))
  plt.yticks([])
  thisplot = plt.bar(range(n_labels), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')













# --------------------------------------------------
# --- Section: Input data
# --------------------------------------------------





# --- Tutorial image data
def load_tutorial_data():
    # --- Get tensor-flow test images (clothes)
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # --- Classes of clothes
    class_names = ['T-shirt/top', 'Trouser', 'Pullover',
                   'Dress', 'Coat', 'Sandal', 'Shirt',
                   'Sneaker', 'Bag', 'Ankle boot']
    return train_images, train_labels, test_images, test_labels, class_names







# --- MAST image data
def load_mast_data(n_cpu):

    # --- How many pulses for each class?
    n_pulses_max = 1000

    # --- Limit the number of images per pulse?
    n_max_per_pulse = 1000 # training + tests

    # --- Test set fraction
    test_fraction  = 0.2 # 20%

    # --- crop images?
    crop_images = True
    crop_x = 100
    crop_y = 100

    # --- Shuffle data? Important otherwise, if n_max_per_pulse is small, you will take images from the beginning of each pulse only
    shuffle_data = True

    # --- resize images?
    resize_images = False #True
    resize_x = 1000
    resize_y = 1000

    # --- Load the file with the pulse data
    filename = fast_cam_pulse_midplane_details #fast_cam_pulse_class
    pulse_list   = []
    class_list   = []
    comment_list = []
    with open(filename) as file_tmp:
        count = 0
        for line in file_tmp:
            count = count + 1
            if (count == 1): continue
            array_tmp = line.split()
            pulse_list.append(int(array_tmp[0]))
            class_list.append(int(array_tmp[1]))
            comment_list.append(array_tmp[len(array_tmp)-1])

    # --- Take pulses for class 0
    selected_pulses  = []
    class_names      = []
    selected_classes = []
    # --- Loop over all types of classes
    n_classes = np.amax(class_list)
    for ic in range(n_classes):
        count = 0
        for i in range(len(pulse_list)):
            if (class_list[i] == ic):
                selected_pulses.append(pulse_list[i])
                selected_classes.append(ic)
                if (count == 0): class_names.append(comment_list[i])
                count = count + 1
                if (count == n_pulses_max): break
    

    # --- Data sent to each process (can't be too large, so can't include camera "data" itself)
    data_broadcast = {}
    data_broadcast['pulse_list']      = selected_pulses
    data_broadcast['shuffle_data']    = shuffle_data
    data_broadcast['crop_images']     = crop_images
    data_broadcast['crop_x']          = crop_x
    data_broadcast['crop_y']          = crop_y
    data_broadcast['resizei_images']  = resize_images
    data_broadcast['resize_x']        = resize_x
    data_broadcast['resize_y']        = resize_y
    data_broadcast['n_max_per_pulse'] = n_max_per_pulse
    # --- Functions specification for mpi processes
    parallel_function = partial(load_pulse_data, data_broadcast)
    parallel_array = [i_pulse for i_pulse in range(len(selected_pulses))]
    # --- Set up basic MPI progress bar (other solution with tqdm didn't work...)
    vprint('Loading '+str(len(selected_pulses))+' pulses in parallel...')
    progress_print = '['
    for i in range(len(selected_pulses)):
        progress_print = progress_print + '.'
    progress_print = progress_print + ']'
    vprint("Total-pulses: " + progress_print)
    if (verbose): print("Processing  : [", end = '', flush=True)
    # --- Set up MPI processes
    with Pool(n_cpu) as mpi_pool:
        data_return = mpi_pool.map(parallel_function, parallel_array)
    all_pulses = data_return
    vprint("]")
    vprint("Finished.")

    # --- Construct full dataset
    fullset_images = []
    fullset_labels = []
    for i in range(len(selected_pulses)):
        n_pulse_img = len(all_pulses[i])
        pulse_images = all_pulses[i]
        for j in range(n_pulse_img):
            fullset_images.append(pulse_images[j])
            fullset_labels.append(selected_classes[i])
        # --- Free memory as we go along
        all_pulses[i] = None

    # --- Free memory of pulse data
    del all_pulses
    fullset_images = np.asarray(fullset_images)
    mem_tot = fullset_images.nbytes
    mem_human = round(mem_tot/1.e6,2)
    vprint("Total memory of dataset: " + str(mem_tot) + " == " + str(mem_human) + "MB")
    total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    print(total_memory, used_memory, free_memory, round(used_memory/1024 / 1024,2))


    # --- Shuffle everything
    fullset_images, fullset_labels = shuffle_images(fullset_images, fullset_labels)

    # --- Extract a test set from the whole set
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    n_test_max = int( test_fraction*float(len(fullset_images)) )
    for i in range(len(fullset_images)):
        if (i <= n_test_max):
            test_images.append(fullset_images[i])
            test_labels.append(fullset_labels[i])
        else:
            train_images.append(fullset_images[i])
            train_labels.append(fullset_labels[i])

    # --- Convert to numpy, although not sure that's needed
    train_images = np.asarray(train_images)
    train_labels = np.asarray(train_labels)
    test_images  = np.asarray(test_images)
    test_labels  = np.asarray(test_labels)

    # --- Only for debugging
    #img = train_images[len(train_images)-10000]
    #matplotlib.image.imsave('images_tmp/testing_files.jpg', img, cmap='gray')


    return train_images, train_labels, test_images, test_labels, class_names













# --------------------------------------------------
# --- Section: Image processing
# --------------------------------------------------




def shuffle_images(images,labels):
    n_shuffle = len(images)
    shuffle = np.arange(n_shuffle)
    np.random.shuffle(shuffle)
    shuffle_images = []
    shuffle_labels = []
    for i in range(n_shuffle):
        shuffle_images.append(images[shuffle[i]])
        shuffle_labels.append(labels[shuffle[i]])
    images = np.asarray(shuffle_images)
    labels = np.asarray(shuffle_labels)
    return images, labels






def resize_image(image_in, desired_size_x, desired_size_y):
    image = Image.fromarray(image_in.astype('uint8'))
    old_size = image.size
    ratio_x = float(desired_size_x)/old_size[0]
    ratio_y = float(desired_size_y)/old_size[1]
    new_size_x = int(old_size[0]*ratio_x)
    new_size_y = int(old_size[1]*ratio_y)
    new_size = tuple([new_size_x,new_size_y])
    image = image.resize(new_size, Image.ANTIALIAS)
    new_image = Image.new("L", (desired_size_x, desired_size_y))
    new_image.paste(image, ((desired_size_x-new_size[0])//2,
                            (desired_size_y-new_size[1])//2))
    image = new_image
    image = np.asarray(image)
    image = image.astype('uint8')
    return image


def resize_image_squared(image_in, desired_size):
    image = Image.fromarray(image_in.astype('uint8'))
    old_size = image.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    image = image.resize(new_size, Image.ANTIALIAS)
    new_image = Image.new("L", (desired_size, desired_size))
    new_image.paste(image, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))
    image = new_image
    image = np.asarray(image)
    image = image.astype('uint8')
    return image




def crop_images_centre(image_in,crop_x,crop_y):
    im = Image.fromarray(image_in)
    width, height = im.size   # Get dimensions
    left = (width - crop_x)/2
    top = (height - crop_y)/2
    right = (width + crop_x)/2
    bottom = (height + crop_y)/2
    im = im.crop((left, top, right, bottom))
    im = np.asarray(im)
    im = im.astype('uint8')
    return im












# --------------------------------------------------
# --- Section: Parallelisation functions
# --------------------------------------------------




def get_CPU_and_GPU_info():
    # --- Check how many cores we have
    n_cpu = multiprocessing.cpu_count()
    gpus = tf.config.list_physical_devices('GPU')
    n_gpu = len(gpus)
    vprint("Number of CPUs: " + str(n_cpu) )
    vprint("Number of GPUs: " + str(n_gpu) )
    return n_cpu, n_gpu




# --- MPI: Load camera data into array
def load_pulse_data(data_broadcast,i_pulse):
    pulse_list      = data_broadcast['pulse_list']
    shuffle_data    = data_broadcast['shuffle_data']   
    crop_images     = data_broadcast['crop_images']    
    crop_x          = data_broadcast['crop_x']         
    crop_y          = data_broadcast['crop_y']         
    resize_images   = data_broadcast['resizei_images'] 
    resize_x        = data_broadcast['resize_x']       
    resize_y        = data_broadcast['resize_y']       
    n_max_per_pulse = data_broadcast['n_max_per_pulse']
    pulse = pulse_list[i_pulse]
    filename = fast_cam_path + '/' + str(pulse) + '.h5'
    if (not os.path.exists(filename)):
        sys.exit('Warning, there is no data for this pulse: ' + str(pulse) )
    h5file = h5py.File(filename, 'r')
    n_pulse_img  = h5file['n_images'][0]
    pulse_images = h5file['images']
    pulse_images = np.asarray(pulse_images)
    # --- It's important to shuffle images before, otherwise, if n_max_per_pulse is small, you will take images from the beginning of each pulse only
    if (shuffle_data):
        tmp_labels = np.zeros(n_pulse_img, dtype='int')
        pulse_images, tmp_labels = shuffle_images(pulse_images, tmp_labels)
    count = 0
    pulse_images_select = []
    for j in range(n_pulse_img):
        if (np.amax(pulse_images[j]) == 0): continue # ignore empty images? (or not???)
        image_process = pulse_images[j]
        if (crop_images):
            image_process = crop_images_centre(image_process,crop_x,crop_y)
        if (resize_images):
            image_process = resize_image(image_process, resize_x, resize_y)
        pulse_images_select.append(image_process)
        count = count + 1
        if (count == n_max_per_pulse): break
    pulse_images_select = np.asarray(pulse_images_select)
    if (verbose): print(".", end = '', flush=True)
    return pulse_images_select



























# --------------------------------------------------
# --- Section: Main program
# --------------------------------------------------

def main():




    # --- We like a clean output... TF import always spits out stuff...
    vprint("\n\n\n")
    vprint("**************************************************************************************")
    vprint("******* TensorFlow code for image recognition applied to MAST fast-camera ************")
    vprint("**************************************************************************************")

    # --- Timer
    time_start = time.perf_counter()

    # --- Info
    vprint("Tensorflow version: " + str(tf.__version__) )

    # --- Hardware
    n_cpu, n_gpu = get_CPU_and_GPU_info()






    vprint("\n\n\n")
    vprint("**************************************************************************************")
    vprint("Loading images...")

    # --- Tutorial data
    #train_images, train_labels, test_images, test_labels, class_names = load_tutorial_data()
    # --- MAST fast-camera data
    train_images, train_labels, test_images, test_labels, class_names = load_mast_data(n_cpu)

    # --- Get image dimensions
    img_dim_x = train_images.shape[1]
    img_dim_y = train_images.shape[2]
    num_classes = len(class_names)

    # --- Print info about data
    vprint("training set dimensions: " + str(train_images.shape) )
    vprint("training labels: " + str(train_labels) )
    vprint("testing set dimensions: " + str(test_images.shape) )
    vprint(["Class names: ",class_names])

    # --- Sanity check
    if (len(train_images) != len(train_labels)):
        sys.exit("Warning: training set and labels have different sizes: "+str(len(train_images))+" and "+str(len(train_labels)) )
    if (len(test_images) != len(test_labels)):
        sys.exit("Warning: training set and labels have different sizes: "+str(len(test_images))+" and "+str(len(test_labels)) )

    # --- Only for debug
    #img = train_images[len(train_images)-10000]
    #matplotlib.image.imsave('images_tmp/testing_files.jpg', img, cmap='gray')

    # --- Timer
    time_load = time.perf_counter()
    time_exec = time_load-time_start
    vprint("Loading time: " + str(time_exec))





    vprint("\n\n\n")
    vprint("**************************************************************************************")
    vprint("Normalising images...")

    # --- Plot raw image
    plot_image_table(1,1,train_images,class_names,train_labels,'images_tmp/raw_image.png')

    # --- Image calibration (EXCEPT if you rescale within the model!!!)
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # --- Plot calibrated images
    plot_image_table(5,5,train_images,class_names,train_labels,'images_tmp/calib_image.png')













    vprint("\n\n\n")
    vprint("**************************************************************************************")
    vprint("Neural Network...")


    # --- Model definition
    vprint("Defining NN model...")
    if (use_easy_model):
        vprint("Using simple NN model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(img_dim_x, img_dim_y)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes)
        ])
    else:
        vprint("Using advanced NN model...")
        train_images = train_images.reshape(-1,img_dim_x,img_dim_y,1)
        test_images  = test_images.reshape(-1,img_dim_x,img_dim_y,1)
        model = tf.keras.Sequential([
            #tf.keras.layers.Rescaling(1./255., input_shape=(img_dim_x,img_dim_y,1)),
            #tf.keras.layers.Rescaling(1./255., input_shape=(img_dim_x,img_dim_y,1)),
            #tf.keras.layers.Flatten(input_shape=(img_dim_x, img_dim_y,1)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_dim_x,img_dim_y,1) ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(img_dim_x,img_dim_y,1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=(img_dim_x,img_dim_y,1)),
            tf.keras.layers.MaxPooling2D(),
            #tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', input_shape=(img_dim_x,img_dim_y,1)),
            #tf.keras.layers.MaxPooling2D(),
            #tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', input_shape=(img_dim_x,img_dim_y,1)),
            #tf.keras.layers.MaxPooling2D(),
            #tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', input_shape=(img_dim_x,img_dim_y,1)),
            #tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes)
        ])
    
    # --- Model compilation
    vprint("Compiling NN model...")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # --- Train the model on training images
    vprint("Training NN model...")
    model.fit(train_images, train_labels, epochs=10)

    # --- Timer
    time_end = time.perf_counter()
    time_exec = time_end-time_start
    vprint("Execution time: " + str(time_exec))





    vprint("\n\n\n")
    vprint("**************************************************************************************")
    vprint("Evaluate NN and make predictions...")

    # --- Evaluate model on test images
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    vprint("Accuracy on test-set: " + str(test_acc) )

    # --- Make predictions for test images
    # --- Note: the prediction is an array of arrays, each one with an entry for
    # ---       each class, each containing the probability distribution for that class
    vprint("Making predictions on test-set...")
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)

    # --- Plot the first X test images, their predicted labels, and the true labels.
    # --- Color correct predictions in blue and incorrect predictions in red.
    vprint("Plotting results...")
    plot_image_table_accuracy(5,3,test_images,class_names,test_labels,predictions,'images_tmp/predict_image.png')
    #plot_image_table_accuracy(15,5,test_images,class_names,test_labels,predictions,'images_tmp/predict_many_image.png')











    vprint("\n\n\n")
    vprint("**************************************************************************************")
    vprint("Evaluate NN on external images...")

    # --- Load some image from file
    vprint("Loading image")
    #img = plt.imread('images_tmp/t-shirt_stan2.png')
    #img = plt.imread('images_tmp/testing_files.jpg')
    img = plt.imread('images_tmp/fast_camera_half.png')
    img = img * 255.0 # re- and de-normalise before/after, because resize acts strangely with [0,1] images
    vprint([img.shape])
    img = resize_image(img, img_dim_x, img_dim_y)
    img = img / 255.0
    img = img.reshape(-1,img_dim_x,img_dim_y,1)
    # --- From BnW to WnB...
    img = 1.0 - img
    # --- Get prediction
    predictions_single = probability_model.predict(img)
    vprint("Confidence in prediction: " + str(max(predictions_single[0])) )
    # --- Plot result
    vprint("Plotting results...")
    plot_image_table_accuracy(1,1,[img[0]],class_names,test_labels,predictions_single,'images_tmp/predict_image_external.png')

















# --------------------------------------------------
# --- Section: Execute
# --------------------------------------------------


if __name__ == '__main__':
    main()


