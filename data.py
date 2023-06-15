import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from functools import partial
import pandas as pd
from sklearn.utils import shuffle
import os
#from reid_code.my_bees_augmentation_func import *
#from temp_folder.my_bees_augmentation_func import *
from data_augmentation import *
from IPython.display import Image
from PIL import Image as Image2

import ast
import time
from datetime import datetime

###########################################################################
#
# FUNCTIONS FOR LOADING AND FORMATTING DATASETS
# BASE CODE FROM https://github.com/jachansantiago/bee_reid
# WITH SOME MODIFICATIONS AND ADDITIONS
#
###########################################################################




###################################################################################################
# FUNCTION FOR LOADING IMAGES
#
# INPUTS
# 1) filepath: string, filename or path of image
# 2) norm_method: int, specifies data normalization method
#                      0: do nothing (pixel values in [0,255] range)
#                      1: divide by 225 (pixel values in [0,1] range)
#                      2: divide by 127.5 and subtract 1 (pixel values in [-1,1] range)
#                      3) divide by 255 and center using provided per channel means and variance
# 3) mean_list: (optional) float list, list of mean pixel values per channel (required when norm_method==3)
# 4) std_list: (optional) float list, list of std pixel values per channel (required when norm_method==3)
# 5) channels: int, number of channels
#
# OUTPUTS
# 1) img: image as TF tensor
#
@tf.function 
def load_image(file_path, norm_method=0, mean_list=None, std_list=None, channels=3):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=channels)
    if norm_method == 1:
        # turns img from integer values to floats in (0,1) range
        img = tf.image.convert_image_dtype(img, tf.float32)
    elif norm_method == 2:
        # map pixels to [-1,1] range (i.e., divide by 127.5 and subtract 1)
        img = tf.cast(img, dtype=tf.float32)
        img = tf.math.divide(img, 127.5)
        img = tf.math.subtract(img, 1.0)
    elif norm_method == 3:
        # divide by 255 then center using provided per channel stats (mean and std)
        img/=255
        img = tf.math.subtract(img, mean_list)
        img = tf.math.divide(img, std_list)
        
    return img
###################################################################################################


###################################################################################################
# FUNTION FOR CROPPING IMAGES
#
# INPUTS
# 1) image: image as TF tensor
# 2) h_range: int list, specifies first and last height coordinate for cropping
# 3) w_range: int list, specifies first and last width coordinate for cropping
#
# OUTPUTS
# cropped image
#
def crop_image(image, h_range, w_range):
    return image[h_range[0]:h_range[1], w_range[0]:w_range[1], :]
###################################################################################################


###################################################################################################
# FUNCTION FOR SPLITTING DATASET INTO TRAIN AND VALIDATION
# MAKES SPLITS ALONG LABLES
#
# INPUTS
# 1) df: pandas dataframe, containing dataset
# 2) label_col: string, name of column containing labels
# 3) train_frac: float, percent of samples to be assigned to training set
#
# OUTPUTS
# 1) train_df: pandas dataframe, containing training examples
# 2) valid_df: pandas dataframe, containing validation examples
#
def train_valid_split_df(df, label_col, train_frac=0.8):
    # get list of labels
    labels = df[label_col].unique()
    # split by labels
    # get number of training labels
    train_num = int(len(labels)*train_frac)
    # permute sample indices
    rand_labels = np.random.permutation(labels)
    # choose training labels
    train_labels = rand_labels[:train_num]
    # split df into train/val
    train_df = df[df[label_col].isin(train_labels)]
    valid_df = df[~df[label_col].isin(train_labels)]
    return train_df, valid_df
###################################################################################################


###################################################################################################
# FUNCTION FOR PREPARING DATASET FOR TRIPLET LOSS FUNCTION
#
# INPUTS
# 1) df: pandas dataframe, containing dataset
# 2) label_col: string, name of column containing labels
# 3) fname_col: string, name of column containing filenames or paths
#
# OUTPUTS
# 1) tdf: pandas dataframe, contains only filename and label coloumns
#
def prepare_for_triplet_loss(df, label_col, fname_col):
    sdf = df.sort_values(label_col)
    labels = sdf[label_col].values
    filename = sdf[fname_col].values
    if labels.shape[0] % 2:
        labels = labels[1:]
        filename = filename[1:]
    pair_labels = labels.reshape((-1, 2))
    pair_filename = filename.reshape((-1, 2))
    ridx = np.random.permutation(pair_labels.shape[0])
    labels = pair_labels[ridx].ravel()
    filename = pair_filename[ridx].ravel()
    tdf = pd.DataFrame({"filename":filename, "label":labels})
    return tdf
###################################################################################################


###################################################################################################
# FUNCTION FOR EXTRACTING FILENAMES AND LABELS
#
# INPUTS
# 1) df: pandas dataframe, contains dataset
# 2) label_col: string, name of column containing labels
# 3) fname_col: string, name of column containing filenames or paths
#
# OUTPUTS
# 1) file_path: string list, list of filenames or paths
# 2) labels: list, list of labels
#
def extract_filenames_and_labels(df, label_col, fname_col):
    file_path = list()
    labels = list()
    # get unique label list
    ids_list = list(df[label_col].unique())
    for i, row in df.iterrows():
        filename = row[fname_col]
        # rename label by index in unique label list
        y = ids_list.index(row[label_col])
        file_path.append(filename)
        labels.append(y)
    return file_path, labels
###################################################################################################





###################################################################################################
# ADD PARAMS FOR LAST FEW AUG METHODS
# FUNCTION THAT APPLIES AUGMENTATION TECHNIQUES TO IMAGES
#
# INPUTS
# 1) images: TF dataset, contains list of images
# 2) data_config: dictionary, contains necessary arguments for augmenting data, including the following
#                 required: 'aug_p', 'aug_methods', list of applicable augmentations, which are 'r_rotate, 'g_blur', 'c_jitter', 'c_drop', 'r_erase', 'r_sat',
#                                'r_bright', 'r_contrast', 'occlusion'
#                 optional (as needed): 'gblur_kernel', 'gblur_sigmin', 'gblur_sigmax', 'jitter_s', 'erase_sh', 'erase_r1',
#                                       'erase_method', 'color_coeff', 'color_N', 'sat_lower', 'sat_upper', 'bright_delta',
#                                       'cont_lower', 'cont_upper', 'occlude_h_range', 'occlude_w_range', 'occlude_val'
# 3) num_parallel_calls: int, used by some tensorflow functions, specifies "the number of batches to compute asynchronously in parallel" (Tensorflow)
#
# OUTPUTS
# 1) images: augmented images dataset
#
def apply_augmentations(images, data_config, num_parallel_calls=10):

    
    if 'r_translate' in data_config['aug_methods']:
        images = images.map(lambda x: random_translation(x, p=data_config['aug_p'], height_range=data_config['translate_hrange'], 
                                                         width_range=data_config['translate_wrange']), num_parallel_calls=num_parallel_calls)
    # random rotation
    if 'r_rotate' in data_config['aug_methods']:
        #images = images.map(random_rotation, num_parallel_calls=num_parallel_calls)
        images = images.map(lambda x: random_rotation(x, p=data_config['aug_p'], minval=data_config['rotate_min'], maxval=data_config['rotate_max']), 
                            num_parallel_calls=num_parallel_calls)
    # gaussian blur
    if 'g_blur' in data_config['aug_methods']:
        images = images.map(lambda x: gaussian_blur(x, data_config['aug_p'], data_config['gblur_kernel'], data_config['gblur_sigmin'], data_config['gblur_sigmax']), 
                            num_parallel_calls=num_parallel_calls)
    # color jitter
    if 'c_jitter' in data_config['aug_methods']:
        images = images.map(lambda x: color_jitter(x, data_config['aug_p'], data_config['jitter_s']), num_parallel_calls=num_parallel_calls)
    # color drop
    if 'c_drop' in data_config['aug_methods']:
        images = images.map(lambda x: color_drop(x, data_config['aug_p']), num_parallel_calls=num_parallel_calls)
    # random erase
    if 'r_erase' in data_config['aug_methods']:
        images = images.map(lambda x: random_erasing(x, data_config['aug_p'], data_config['erase_sl'], data_config['erase_sh'], data_config['erase_r1'], 
                                                     data_config['erase_method']), num_parallel_calls=num_parallel_calls)
    # random saturation
    if 'r_sat' in data_config['aug_methods']:
        images = images.map(lambda x: random_saturation(x, data_config['aug_p'], data_config['sat_lower'], data_config['sat_upper']), 
                            num_parallel_calls=num_parallel_calls)
    # random brightness
    if 'r_bright' in data_config['aug_methods']:
        images = images.map(lambda x: random_brightness(x, data_config['aug_p'], data_config['bright_delta']), num_parallel_calls=num_parallel_calls)
    # random contrast
    if 'r_contrast' in data_config['aug_methods']:
        images = images.map(lambda x: random_contrast(x, data_config['aug_p'], data_config['cont_lower'], data_config['cont_upper']), 
                            num_parallel_calls=num_parallel_calls)
    # random occlusion
    if 'occlusion' in data_config['aug_methods']:
        images = images.map(lambda x: occlude_image(x, data_config['occlude_h_range'], data_config['occlude_w_range'], data_config['aug_p']), 
                            num_parallel_calls=num_parallel_calls)
    
    return images
###################################################################################################


###################################################################################################
# FUNCTION FOR CREATING TF DATASET FOR UCL MODEL
# 
# INPUTS
# 1) df: pandas dataframe, dataframe containing containing data (images, labels)
# 2) data_config: dictionary, contains necessary arguments for loading data
#                 required: 'fname_col', 'label_col', 'cropped' 'input_size', 'augmentation'
#                 optional (as needed): 'h_range', 'w_range', 'aug_methods', 'mean', 'std'
#                 See apply_augmentations() for more required info
# 5) num_parallel_calls: int, used by some tensorflow functions, specifies "the number of batches to compute asynchronously in parallel" (Tensorflow)
#
# OUTPUTS
# 1) dataset: a tensorflow dataaset
#
def load_tf_pair_dataset(df, data_config, num_parallel_calls=10):
    x1_path_list = []
    label_list = []
    counter = 0
    for fname, label in zip(df[data_config['fname_col']].values, df[data_config['label_col']]):
        x1_path_list.append(fname)
        label_list.append(counter)
        counter+=1
        
    x1_path_list = np.array(x1_path_list)
    label_list = np.array(label_list)
    assert x1_path_list.shape[0] == label_list.shape[0]
    # shuffle
    index_list = np.arange(x1_path_list.shape[0])
    np.random.shuffle(index_list)
    x1_path_list = x1_path_list[index_list]
    x1_path_list = x1_path_list.copy()
    x2_path_list = x1_path_list.copy()
    label_list = label_list[index_list]
    
    x1_path_list = tf.data.Dataset.from_tensor_slices(x1_path_list)
    x2_path_list = tf.data.Dataset.from_tensor_slices(x2_path_list)
    label_list = tf.data.Dataset.from_tensor_slices(label_list)
    
    #x1_images = x1_path_list.map(lambda x: load_image(x, backbone, finetune), num_parallel_calls=num_parallel_calls)
    x1_images = x1_path_list.map(lambda x: load_image(x, data_config['norm_method'], data_config['mean'], data_config['std'], data_config['input_size'][-1]), num_parallel_calls=num_parallel_calls)
    if data_config['cropped']:
        x1_images = x1_images.map(lambda x: crop_image(x, data_config['h_range'], data_config['w_range']), num_parallel_calls=num_parallel_calls)
    # resize image if required
    sample = next(iter(x1_images.batch(1)))
    if sample.shape[1:-1] != data_config['input_size'][:2]:
        x1_images = x1_images.map(lambda x: tf.image.resize(x, data_config['input_size'][:2]), num_parallel_calls=num_parallel_calls)
        
    # if simCLR, augment both x1 and x2
    if data_config['simCLR']: 
        x1_images = apply_augmentations(x1_images, data_config, num_parallel_calls)
    
    x2_images = x2_path_list.map(lambda x: load_image(x, data_config['norm_method'], data_config['mean'], data_config['std'], data_config['input_size'][-1]), num_parallel_calls=num_parallel_calls)
    if data_config['cropped']:
        x2_images = x2_images.map(lambda x: crop_image(x, data_config['h_range'], data_config['w_range']), num_parallel_calls=num_parallel_calls)
    # resize image if required
    sample = next(iter(x2_images.batch(1)))
    if sample.shape[1:-1] != data_config['input_size'][:2]:
        x2_images = x2_images.map(lambda x: tf.image.resize(x, data_config['input_size'][:2]), num_parallel_calls=num_parallel_calls)
    # augment x2_images
    x2_images = apply_augmentations(x2_images, data_config, num_parallel_calls)
    
    dataset = tf.data.Dataset.zip((x1_images, x2_images, label_list))
    return dataset
###################################################################################################


###################################################################################################
# FUNCTION FOR CONSTRUCTING TF DATASET FOR SCL MODEL
# 
# INPUTS
# 1) df: pabdas dataframe, dataframe containing containing data (images, labels)
# 2) data_config: dictionary, contains necessary arguments for loading data
#                 required: 'cropped', 'rescale_factor', 'image_size', 'augmentation'
#                 optional (as needed): 'h_range', 'w_range', 'aug_methods'
#                 If using augmentation, see apply_augmentations() for more required info
# 3) label_col: string, name of column containing labels
# 4) fname_col: string, name of column containing filenames or paths
# 5) backbone: string, specifies backbone of model to be trained with data
# 6) finetune: bool, whether model to be trained is finetuned
# 7) mean_list: (optional) float list, list of mean pixel values per channel (for finetuning, if using mean centering with pre-trained dataset stats)
# 8) std_list: (optional) float list, list of std pixel values per channel (for finetuning, if using mean centering with pre-trained dataset stats)
# 9) validation: bool, whether dataset is validation set
# 10) num_parallel_calls: int, used by some tensorflow functions, specifies "the number of batches to compute asynchronously in parallel" (Tensorflow)
#
# OUTPUTS
# 1) dataset: a tensorflow dataset
#
def load_tf_dataset(df, data_config, label_col, fname_col, validation=False, num_parallel_calls=10):
    
    # get lists of filenames and labels
    filenames, labels = extract_filenames_and_labels(df, label_col, fname_col)
    # make lists into TensorSliceDataset
    filenames = tf.data.Dataset.from_tensor_slices(filenames)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    # then make into MapDataset               
    images = filenames.map(lambda x: load_image(x, data_config['norm_method'], data_config['mean'], data_config['std'], data_config['input_size'][-1]), num_parallel_calls=num_parallel_calls)
    # crop images if specified
    if data_config['cropped'] == True and data_config['crop_before_aug'] == True:
        images = images.map(lambda x: crop_image(x, data_config['h_range'], data_config['w_range']), num_parallel_calls=num_parallel_calls)
    # add data augmentation if specified
    if data_config['augmentation'] == True and validation == False:
        images = apply_augmentations(images, data_config, num_parallel_calls)
    # crop images if specified
    if data_config['cropped'] == True and data_config['crop_before_aug']==False:
        images = images.map(lambda x: crop_image(x, data_config['h_range'], data_config['w_range']), num_parallel_calls=num_parallel_calls)
    # resize image if necessary
    images = images.map(lambda x: tf.image.resize(x, data_config['input_size'][:2]), num_parallel_calls=num_parallel_calls)
    dataset = tf.data.Dataset.zip((images, labels))
    return dataset
###################################################################################################


###################################################################################################
# INTERMEDIATE FUNCTION FOR CONSTRUCTING TF DATASET, TRAIN AND VALIDATION, FOR SCL MODEL
#
# INPUTS
# 1) train_df: pandas dataframe, dataframe containing training data
# 2) valid_df: pandas dataframe, dataframe containing validation data
# 3) data_config: dictionary, contains necessary arguments for loading data, see load_tf_dataset() for required info
# 3) label_col: string, name of column containing labels
# 4) fname_col: string, name of column containing filenames or paths
# 5) shuffle: bool, whether to shuffle the data
#
# OUTPUTS
# 1) train_dataset: train set as tensorflow dataset
# 2) valid_dataset: valid set as tensorflow dataset
#
def load_dataset(train_df, valid_df, data_config, label_col, fname_col, shuffle=True):
    
    train_dataset = load_tf_dataset(train_df, data_config, label_col, fname_col)
    valid_dataset = load_tf_dataset(valid_df, data_config, label_col, fname_col, validation=True)
    # shuffling if needed
    if shuffle:
        train_dataset = train_dataset.shuffle(len(train_df))
        valid_dataset = valid_dataset.shuffle(len(valid_df))    
    return train_dataset, valid_dataset
###################################################################################################


###################################################################################################
# FUNCTION TO LOAD DATA FOR SCL MODEL
# 
# INPUTS
# 1) data_config: dictionary, contains necessary arguments for loading data, including 'train_fname', 'valid_fname', 'label_col', 'fname_col', 'train_frac'
#                 see load_dataset() for more required info
# 2) verbose: bool, whether to print out messages
#
# OUTPUTS
# 1) train_dataset: train set as tensorflow dataset
# 2) valid_dataset: valid set as tensorflow dataset
#
def load_data(data_config, verbose=False):
    
    if verbose:
        print('Printing data_config:')
        for key, value in data_config.items():
            print(f'{key}: {value}')
    # split train data into train/validation if necessary
    if data_config['valid_fname'] is None:
        df = pd.read_csv(data_config['train_fname'])
        train_df, valid_df = train_valid_split_df(df, data_config['label_col'], train_frac=data_config['train_frac'])
    # else load previously split train/valid
    else:
        train_df = pd.read_csv(data_config['train_fname'])
        valid_df = pd.read_csv(data_config['valid_fname'])
    
    # resulting dfs have columns ['filename', 'label']
    train_df = prepare_for_triplet_loss(train_df, data_config['label_col'], data_config['fname_col'])
    valid_df = prepare_for_triplet_loss(valid_df, data_config['label_col'], data_config['fname_col'])
    
    train_dataset, valid_dataset = load_dataset(train_df, valid_df, data_config, 'label', 'filename', shuffle=False)
    
    return train_dataset, valid_dataset
###################################################################################################


###################################################################################################
# FUNCTION FOLR LOADING DATA FOR UCL MODEL
#
# INPUTS
# 1) data_config: dictionary, contains necessary arguments for loading data, including 'train_fname', 'valid_fname', 'label_col', 'fname_col', 'train_frac';
#                 see load_tf_pair_dataset() for more required info
# 2) verbose: bool, whether to print out messages
#
# OUTPUTS
# 1) train_dataset: train set as tensorflow dataset
# 2) valid_dataset: valid set as tensorflow dataset
#
def load_data_v2(data_config, verbose=False):

    if verbose:
        print('Printing data_config:')
        for key, value in data_config.items():
            print(f'{key}: {value}')

    # split train data into train/validation if necessary
    if data_config['valid_fname'] is None:
        df = pd.read_csv(data_config['train_fname'])
        train_df, valid_df = train_valid_split_df(df, data_config['label_col'], train_frac=data_config['train_frac'])
    # else load previously split train/valid
    else:
        train_df = pd.read_csv(data_config['train_fname'])
        valid_df = pd.read_csv(data_config['valid_fname'])

    
    train_df = train_df.sample(train_df.shape[0])
    valid_df = valid_df.sample(valid_df.shape[0])
    
    train_dataset = load_tf_pair_dataset(train_df, data_config)
    valid_dataset = load_tf_pair_dataset(valid_df, data_config)

    return train_dataset, valid_dataset
###################################################################################################




###################################################################################################
# FUNCTION FOR LOADING EVALUATION DATASET
# 
# INPUTS
# 1) filenames: string list, list of filenames or paths containing images
# 2) data_config: dictionary, contains necessary arguments for loading data, including 'norm_method', 'mean', 'std', 'input_size', 'cropped',
#                 'h_range', 'w_range';
#                 see get_data_for_MTL_ID_color() for required info
# 3) num_parallel_calls: int, used by some tensorflow functions, specifies "the number of batches to compute asynchronously in parallel" (Tensorflow)
#
# OUTPUTS
# 1) images: tensorflow dataset containing just images
#
def filename2image(filenames,data_config, num_parallel_calls=10):
    
    filenames = tf.data.Dataset.from_tensor_slices(filenames)
    images = filenames.map(lambda x: load_image(x, data_config['norm_method'], data_config['mean'], data_config['std'], data_config['input_size'][-1]), num_parallel_calls=num_parallel_calls)
    if data_config['cropped']:
        images = images.map(lambda x: crop_image(x, data_config['h_range'], data_config['w_range']), num_parallel_calls=num_parallel_calls)
    #func = lambda x: tf.image.resize(x, data_config['input_size'][:2])
    #images = images.map(func, num_parallel_calls=10)
    images = images.map(lambda x: tf.image.resize(x, data_config['input_size'][:2]), num_parallel_calls=num_parallel_calls)
    return images
###################################################################################################

