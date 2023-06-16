import os
import numpy as np
import tensorflow as tf
import pandas as pd
from data import filename2image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from IPython.display import Image
from PIL import Image as Image2
import time
from datetime import datetime


########################################################
#
# REFORMATTING EVALUATION FUNCTIONS
# BASE CODE FROM https://github.com/jachansantiago/bee_reid
# WITH SOME MODIFICATIONS AND ADDITIONS
#
########################################################



###################################################################################################
# FUNCTION TO PERFORM CMC GALLERY EVALUATION
#
# INPUTS
# 1) model: a TF model
# 2) model_type: string, specifies model type (e.g., 'SCL')
# 3) data_config: dictionary, contains necessary parameters to load images, including 'gallery_fname', 'fname_col', 'gallery_id',
#                             'iteration_id', 'image_id_col' and 'n_distractors'
# 4) verbose: bool, whether print out comments
#
# OUTPUTS
# 1) ranks: float list, cmc scores from top-1 to top-k
#
def evaluate_cmc(model, model_type, data_config, verbose=False):

    if verbose:
        print('Getting gallery images')
    df = pd.read_csv(data_config['gallery_fname'])
    df = df[df[data_config['image_id_col']] < data_config['n_distractors'] + 2]

    # get images
    queries_and_galleries = df[data_config['fname_col']].values
    images = filename2image(queries_and_galleries, data_config)
    
    # get embeddings for images
    predictions = model.predict(images.batch(32), verbose=False)
    # if model is a MTL, use the output of the first head (reid head)
    if model_type == 'MTL':
        predictions = predictions[0]
    if verbose:
        print('Finished embedding images')

    query_gallery_size = df[data_config['image_id_col']].max() + 1
    n_distractors = query_gallery_size - 2
    query_gallery_size, n_distractors

    # calculate total num of galleries across all iterations
    galleries_per_iteraration = len(df[data_config['gallery_id']].unique())
    iterations = df[data_config['iteration_id']].max() + 1
    total_galleries =  galleries_per_iteraration * iterations
    galleries_per_iteraration, iterations, total_galleries

    # get queries and galleries embedding
    queries_emb = predictions[::query_gallery_size]
    pred_idx = np.arange(0, len(predictions))
    galleries_emb = predictions[np.mod(pred_idx, query_gallery_size) != 0]
    queries_emb = queries_emb.reshape(total_galleries, 1, -1)
    galleries_emb = galleries_emb.reshape(total_galleries, n_distractors + 1, -1 )

    # Calculate distance
    cos_dist = tf.matmul(queries_emb, galleries_emb, transpose_b=True).numpy()
    euclid_dist = -(cos_dist - 1)

    # Calculate Rank
    r = np.argmin(np.argsort(euclid_dist), axis=2)
    r = np.squeeze(r)
    ranks = np.zeros(n_distractors)
    for i in range(n_distractors):
        ranks[i] = np.mean(r < (i + 1))

    return ranks
###################################################################################################



###################################################################################################
# FUNCTION TO CALCULATE LABEL CENTROIDS
# 
# INPUTS
# 1) embeddings_list: numpy array, containing images embedded in the learned representation space
# 2) label_list: list, containing labels
# 3) distance: string, specifies what distance metric to use from 'euclid', 'pseudo_cos', and 'cos' (cosine)
#                      'pseudo_cos' uses euclidean distance and the projects centroid back to the unit hypersphere
#
# OUTPUTS
# 1) dictionary containing 'label_centroids': numpy array representing the position of each label
#                          'index_label_map': dictionary mapping index in label_centroids to corresponding label
#                          'label_index_map': dictionary mapping label to corresponding index in label_centroids
#
def calculate_label_centroids(embedding_list, label_list, distance='euclid'):
    N_label = np.unique(label_list).shape[0]
    emb_dim = embedding_list.shape[1]
    label_centroids = np.zeros((N_label, emb_dim)) # for storing centroid vectors
    index_label_map = {} # for storing map from centroid index to label
    label_index_map = {} # for storing map from label to centroid index
    index = 0
    # for each label
    for label in np.unique(label_list):
        index_label_map[index] = label
        label_index_map[label] = index
        if distance == 'euclid':
            label_centroids[index] = np.mean(embedding_list[label_list==label], axis=0)
        # pseudo_cos calculates Euclidean centroid and then renormalizes centroid to be on unit hypersphere
        elif distance == 'pseudo_cos':
            label_centroids[index] = np.mean(embedding_list[label_list==label], axis=0)
            label_centroids[index]/= np.linalg.norm(label_centroids[index])
        elif distance == 'cos':
            print('ERROR - cosine centroid not yet coded')
            return -1
        else:
            print(f'ERROR - invalid choice of distance:{distance}')
            return -1
        index+=1
    return {'label_centroids':label_centroids, 'index_label_map':index_label_map, 'label_index_map':label_index_map}
###################################################################################################



###################################################################################################
# FUNCTION TO CALCULATE COSINE DISTANCE
def cosine_distance(X,Y):
    return 1. - np.dot(X,Y.T)
###################################################################################################



###################################################################################################
# FUNCTION TO MAKE LABEL PREDICTIONS BY CENTROID
#
# INPUTS
# 1) embeddings_list: numpy array, containing images embedded in the learned representation space
# 2) centroid_list: numpy array, containing the centroids of each label
# 3) metric: string, specifies distance metric to use, from 'euclid' and 'cos' (cosine)
#
# OUTPUTS
# 1) predictions: numpy array of label predictions
#
# predict label using label centroids
# consider using metric function as parameter instead of string choice
def predict_label_by_centroid(embedding_list, centroid_list, metric='euclid', index_label_map = None):

    # edit: changing for cases where labels are type string
    #predictions = np.zeros(embedding_list.shape[0])
    predictions = ['' for k in range(embedding_list.shape[0])]

    for k, embedding in enumerate(embedding_list):
        if metric == 'euclid':
            # calculate distance of embedding with every centroid
            distance_list = np.linalg.norm(embedding - centroid_list, axis=1)
        elif metric == 'cos':
            distance_list = cosine_distance(embedding, centroid_list)
        else:
            print(f'ERROR - invalid choice of metric: {metric}')
        distance_arg_sorted = np.argsort(distance_list)
        if index_label_map != None:
            predictions[k] = index_label_map[distance_arg_sorted[0]]
        else:
            # predict the index of the closest centroid
            predictions[k] = distance_arg_sorted[0]
    predictions = np.array(predictions)
    return predictions
###################################################################################################



###################################################################################################
# FUNCTION TO CALCULATE ACCURACY
#
# 1) predictions: numpy array, contains label predictions
# 2) true_labels: numpy array, contains true labels
# 3) by_class: bool, whether to return per label accuracy
#
# calculate accuracy of predictions
def calculate_accuracy(predictions, true_labels, by_class=False):
    if by_class:
        class_list = np.unique(true_labels)
        class_accuracy = {c:0.0 for c in class_list}
        for c in class_list:
            mask = true_labels==c
            class_accuracy[c] = np.sum(predictions[mask] == true_labels[mask])/true_labels[mask].shape[0]
        return class_accuracy
    else:
        return np.sum(predictions==true_labels)/predictions.shape[0]
###################################################################################################



###################################################################################################
# FUNCTION TO CALCULATE INTRALABEL STATISTICS
# For each label, calculates all distances of same label pairs
# and reports the statistics of mean, variance, min, median, and max
#
# INPUTS
# 1) embeddings_list: numpy array, containing images embedded in the learned representation space
# 2) label_list: numpy array, contains corresponding labels of embeddings_list
#
# OUTPUTS
# 1) spread_df: pandas dataframe, with columns [label, mean, var, min, median, max]
# 2) global_stats: dictionary, contains global distance statistics (mean, var, min, median, max)
#
def calculate_intralabel_stats(embedding_list, label_list):
    #label_spread = {}
    label_spread = []
    unique_labels = np.unique(label_list)
    global_distances = []
    for label in unique_labels:
        # calculate shortest and longest distance between any two points of the same label
        mask = np.where(label_list==label)[0]
        class_vectors = embedding_list[mask]
        distance_matrix = 1.0 - np.matmul(class_vectors, class_vectors.T)
        # remove diagonals
        mask = 1.0 - np.diag(np.ones(class_vectors.shape[0]))
        mask = mask.astype(bool)
        distances = distance_matrix[mask].flatten()
        global_distances+=list(distances)
        label_spread.append([label, np.mean(distances), np.var(distances), np.min(distances), np.median(distances),  np.max(distances)])

    spread_df = pd.DataFrame(label_spread)
    spread_df.columns = ['label', 'mean', 'var', 'min', 'median', 'max']
    global_stats = {'mean':np.mean(global_distances), 'var':np.var(global_distances), 'min':np.min(global_distances), 'median':np.median(global_distances), 'max':np.max(global_distances)}
    return spread_df, global_stats
###################################################################################################



###################################################################################################
# FUNCTION TO CALCULATE INTERLABEL STATS
# For each label, calculates all distances of (inlabel embedding, outlabel embedding) pairs
# and reports the statistics of mean, variance, min, median, and max
#
# INPUTS
# 1) embedding_list: numpy array, containing images embedded in the learned representation space
# 2) label_list: numpy array, contains corresponding labels of embeddings_list
#
# OUTPUTS
# 1) interlabel_stats_df: pandas dataframe, with columns [label_1, labels_2, mean, var, min, median, max]
# 2) global_stats: dictionary, contains global distance statistics (mean, var, min, median, max)
#
def calculate_interlabel_stats(embedding_list, label_list):
    unique_labels = np.unique(label_list)
    interlabel_stats = []
    global_distances = []
    # store mininum distances between each label
    for k in range(unique_labels.shape[0]-1):
        label = unique_labels[k]
        mask = label_list==label
        class_vectors = embedding_list[mask]
        for j in range(k+1, unique_labels.shape[0]):
            other_label = unique_labels[j]
            other_mask = label_list==other_label
            other_vectors = embedding_list[other_mask]
            distances = 1.0 - np.matmul(class_vectors, other_vectors.T)
            distances = distances.flatten()
            global_distances+=list(distances)
            row = [unique_labels[k], unique_labels[j], np.mean(distances), np.var(distances), np.min(distances), np.median(distances),  np.max(distances)]
            interlabel_stats.append(row)
    interlabel_stats_df = pd.DataFrame(interlabel_stats)
    interlabel_stats_df.columns = ['label_1', 'label_2', 'mean', 'var', 'min', 'median', 'max']
    global_stats = {'mean':np.mean(global_distances), 'var':np.var(global_distances), 'min':np.min(global_distances), 'median':np.median(global_distances), 'max':np.max(global_distances)}
    return interlabel_stats_df, global_stats
###################################################################################################



###################################################################################################
# FUNCTION TO OBTAIN DATE
#
def get_date():
    months = {'01':'JAN', '02':'FEB', '03':'MAR', '04':'APR', '05':'MAY', '06':'JUN', '07':'JUL', '08':'AUG', '09':'SEP', '10':'OCT', '11':'NOV', '12':'DEC'}
    now = datetime.now() # current date and time
    d = now.strftime('%y-%m-%d')
    d = d.split('-')
    date = d[0]+months[d[1]]+d[2]
    return date
###################################################################################################



###################################################################################################
# FUNCTION FOR CALCULATING EMBEDDING STATISTICS
# Calculates all intra- and inter-label statistics for train, test, and reference sets
# and stores them into csv files
#
# INPUTS
# 1) eval_config: dictionary, contains necessary parameters to perform evaluation, including 'model_fname',
#                             'model_type', 'results_dir'
# 2) data_config: dictionary, contains necessary parameters to load data, including 'train_fname', 'test_fname',
#                             'ref_fname'm 'fname_col', 'label_col'
# 3) model_key: string, specifies the model key as stored in models_db.csv
# 4) verbose: bool, whether to print comments
#
# OUTPUTS
#  No outputs; stats are stored in results_dir; filenames will contain data and model key
#
def embedding_statistics(eval_config, data_config, model_key, verbose=False):
    
    # get date
    date = get_date()
    
    # load model
    model = load_model(eval_config['model_fname'], custom_objects={'tf': tf})
    
    # get dataframes
    df_train = pd.read_csv(data_config['train_fname'])
    df_test = pd.read_csv(data_config['test_fname'])
    if data_config['ref_fname'] is not None:
        df_ref = pd.read_csv(data_config['ref_fname'])
    
    # get train data
    image_train = filename2image(df_train[data_config['fname_col']].values, data_config)
    embeddings_train = model.predict(images_train.batch(32), verbose=False)
    if eval_config['model_type'] == 'MTL':
        embedding_train = embeddings_train[0]
    labels_train = df_train[data_config['label_col']].values
    # get test data
    image_test = filename2image(df_test[data_config['fname_col']].values, data_config)
    if eval_config['model_type'] == 'MTL':
        embedding_test = embeddings_test[0]
    embeddings_test = model.predict(images_test.batch(32), verbose=False)
    labels_test = df_test[data_config['label_col']].values
    # get ref data
    if data_config['ref_fname'] is not None:
        image_ref = filename2image(df_ref[data_config['fname_col']].values, data_config)
        embeddings_ref = model.predict(images_ref.batch(32), verbose=False)
        if eval_config['model_type'] == 'MTL':
            embeddings_ref = embeddings_ref[0]
        labels_ref = df_ref[data_config['label_col']].values
    
    # get train stats
    df_intra_stats_train, global_intra_stats_train = calculate_intralabel_stats(embeddings_train, labels_train)
    fname = date + '_'.join([date, model_key, 'intralabel_train.csv'])
    df_intra_stats_train.to_csv(fname, index=False)
    df_inter_stats_train, global_inter_stats_train = calculate_interlabel_stats(embeddings_train, labels_train)
    fname = date + '_'.join([date, model_key, 'interlabel_train.csv'])
    df_inter_stats_train.to_csv(fname, index=False)    
    # get test stats
    df_intra_stats_test, global_intra_stats_test = calculate_intralabel_stats(embeddings_test, labels_test)
    fname = date + '_'.join([date, model_key, 'intralabel_test.csv'])
    df_intra_stats_test.to_csv(fname, index=False)
    df_inter_stats_test, global_inter_stats_test = calculate_interlabel_stats(embeddings_test, labels_test)
    fname = date + '_'.join([date, model_key, 'interlabel_test.csv'])
    df_inter_stats_test.to_csv(fname, index=False)
    if eval_config['ref_fname'] is not None:
        # get ref stats
        df_intra_stats_ref, global_intra_stats_ref = calculate_intralabel_stats(embeddings_ref, labels_ref)
        fname = eval_config['results_dir'] + '_'.join([date, model_key, 'intralabel_ref.csv'])
        df_intra_stats_ref.to_csv(fname, index=False)
        df_inter_stats_ref, global_inter_stats_ref = calculate_interlabel_stats(embeddings_ref, labels_ref)
        fname = date + '_'.join([date, model_key, 'intralabel_ref.csv'])
        df_inter_stats_ref.to_csv(fname, index=False)
###################################################################################################

        
        
###################################################################################################
# FUNCTION FOR PERFORMING MODEL EVALUATION
#
# INPUTS
# 1) model: a TF model
# 2) eval_config: dictionary, contains necessary parameters for evaluation, including the following:
#                             'methods' - string list, specifies which evaluation methods to use from ['gallery', 'knn', 'centroid', 'svm'] 
#                             'n_neighbors' - int, if using knn, specifies value for k
#                             'per_class' - bool, whether to report per class accuracies for knn, centroid and svm
#                             'conf_matrix' - bool, whether to store confusion matrices
#                             'model_type' - string, specifies model type (e.g., 'SCL')
# 3) data_config: dictionary, contains necessary parameters to load and process input to model, including 'train_fname', 'test_fname', 
#                             'fname_col', 'label_col'
#                             NOTE: for open set setting, use reference set as train_fname, and query set as test_fname
#
# OUTPUTS
# 1) results: dictionary, stores results for each method specified
#       
def model_evaluation(model, eval_config, data_config, verbose=False):

    # if using all methods, reformat methods arg    
    if eval_config['methods'] == 'all':
        methods = ['gallery', 'knn', 'centroid', 'svm']
    else:
        methods = eval_config['methods']

    if verbose:
        print('Evaluating model using the following methods:')
        print(methods)

    results = {}
    # performing gallery evaluation
    if 'gallery' in methods:
        if verbose:
            print('Evaluating model on galleries')
        ranks = evaluate_cmc(model, eval_config, data_config, verbose)
        results['top1'] = ranks[0]
        results['top3'] = ranks[2]
    

    # getting embeddings for other evaluations
    # load train images
    df_train = pd.read_csv(data_config['train_fname'])
    train_images = filename2image(df_train[data_config['fname_col']].values, data_config)

    # get image embeddings
    train_images = model.predict(train_images.batch(32), verbose=False)
    if eval_config['model_type'] == 'MTL':
        train_images = train_images[0]
    train_labels = df_train[data_config['label_col']].values

    # load test (or query) images
    df_test = pd.read_csv(data_config['test_fname'])
    test_images = filename2image(df_test[data_config['fname_col']].values, data_config)
    # get image embeddings
    test_images = model.predict(test_images.batch(32), verbose=False)
    if eval_config['model_type'] == 'MTL':
        test_images = test_images[0]
    test_labels = df_test[data_config['label_col']].values

    label_list = list(set(test_labels))
    label_dict = {val:k for k, val in enumerate(label_list)}
    results['label_list'] = label_list
    results['label_dict'] = label_dict

    if 'knn' in methods:
        # BUILD KNN MODEL AND PREDICT
        if verbose:
            print(f'Training kNN classifier with k={eval_config["n_neighbors"]}')
        my_knn = KNeighborsClassifier(n_neighbors=eval_config['n_neighbors'], metric='cosine')
        my_knn.fit(train_images, train_labels)
        knn_pred = my_knn.predict(test_images)
        knn_acc = np.round(np.sum([1 for pred, label in zip(knn_pred, test_labels) if pred == label])/test_labels.shape[0],4)
        # store results
        results['n_neighbors'] = eval_config['n_neighbors']
        results['knn'] = knn_acc
        if eval_config['per_class']:
            knn_class = np.zeros(len(label_list))
            for k, label in enumerate(label_list):
                mask = test_labels == label
                knn_class[k] = np.round(np.sum(knn_pred[mask]==test_labels[mask])/np.sum(mask),4)
            # store results
            results['knn_class'] = knn_class
        if eval_config['conf_matrix']:
            knn_conf = confusion_matrix(test_labels, knn_pred)
            results['knn_conf'] = knn_conf
        if verbose:
            print(f'kNN test accuracy: {knn_acc}')


    # maybe decomission this method; keep only knn and svm
    if 'centroid' in methods:
        # CALCULATE CENTROIDS AND PREDICT
        if verbose:
            print('Training Centroid classifier')
        my_centroids = calculate_label_centroids(train_images, train_labels, distance='pseudo_cos')
        centroid_pred = predict_label_by_centroid(test_images, my_centroids['label_centroids'], 'cos', my_centroids['index_label_map'])
        centroid_acc = np.round(np.sum([1 for pred, label in zip(centroid_pred, test_labels) if pred == label])/test_labels.shape[0],4)
        # store results
        results['centroid'] = centroid_acc
        if eval_config['per_class']:
            centroid_class = np.zeros(len(label_list))
            for k, label in enumerate(label_list):
                mask = test_labels == label
                centroid_class[k] = np.round(np.sum(centroid_pred[mask]==test_labels[mask])/np.sum(mask),4)
            # store results
            results['centroid_class'] = centroid_class
        if eval_config['conf_matrix']:
            centroid_conf = confusion_matrix(test_labels, centroid_pred)
            results['centroid_conf'] = centroid_conf
        if verbose:
            print(f'Centroid test accuracy: {centroid_acc}')


    if 'svm' in methods:
        # TRAIN SVM CLASSIFIER AND PREDICT
        if verbose:
            print('Training SVM classifier')
        my_svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        my_svm.fit(train_images, train_labels)
        svm_pred = my_svm.predict(test_images)
        svm_acc = np.round(np.sum([1 for pred, label in zip(svm_pred, test_labels) if pred == label])/test_labels.shape[0],4)
        results['svm'] = svm_acc
        if eval_config['per_class']:
            svm_class = np.zeros(len(label_list))
            for k, label in enumerate(label_list):
                mask = test_labels == label
                svm_class[k] = np.round(np.sum(svm_pred[mask]==test_labels[mask])/np.sum(mask),4)
            # store results
            results['svm_class'] = svm_class
        if eval_config['conf_matrix']:
            svm_conf = confusion_matrix(test_labels, svm_pred)
            results['svm_conf'] = svm_conf

        if verbose:
            print(f'SVM test accuracy: {svm_acc}')

    return results
###################################################################################################



###################################################################################################
# FUNCTION TO LOAD IMAGES FOR BASELINE - PIXEL
#
# INPUTS
# 1) fname: string, specifies filename or path for a csv file containing imaga filenames or paths
# 2) label_col: string, name of column containing labels
# 3) fname_col: string, name of column containing filenames or paths
# 3) input_size: int list, specifies dimensions of image input, channels last;
#                          if cropping, should match image size after crop
# 5) normalize: bool, specifies whether to normalize pixel values by dividing by 255
# 6) cropped: bool, specifies whether to crop images
# 7) h_range: int list, specifies height pixel position range, to be used if cropped == True
# 8) w_range: int list, specifies width pixel position range, to be used if cropped == True
#
# OUTPUTS
# 1) images: numpy array, contains images as 1D arrays
# 2) labels: numpy array, contains corresponding labels of the images
#
def baseline_load_images(fname, label_col, fname_col, input_size, normalize, cropped, h_range=None, w_range=None):
    df = pd.read_csv(fname)
    
    labels = df[label_col].values
    images = np.zeros((df.shape[0],input_size[0],input_size[1],input_size[2])).reshape((df.shape[0], input_size[0]*input_size[1]*input_size[2]))
    for k, filename in enumerate(df[fname_col].values):
        img = np.array(Image2.open(filename))
        if normalize:
            img = img/255
        if cropped:
            images[k] = img[h_range[0]:h_range[1], w_range[0]:w_range[1],:].flatten().copy()
        else:
            images[k] = img.flatten().copy()

    return images, labels
###################################################################################################



###################################################################################################
# FUNCTION TO LOAD IMAGES FOR BASELINE - PIXEL PCA
#
# INPUTS
# 1) fname: string, specifies filename or path for a csv file containing imaga filenames or paths
# 2) label_col: string, name of column containing labels
# 3) fname_col: string, name of column containing filenames or paths
# 3) input_size: int list, specifies dimensions of image input, channels last;
#                          if cropping, should match image size after crop
# 5) normalize: bool, specifies whether to normalize pixel values by dividing by 255
# 6) cropped: bool, specifies whether to crop images
# 7) h_range: int list, specifies height pixel position range, to be used if cropped == True
# 8) w_range: int list, specifies width pixel position range, to be used if cropped == True
# 9) n_components: int or float, for PCA; if int, specifies number of components; if float in range (0,1), specifies cummulative
#                                variance that the resulting components must account for
# 10) my_pca: sklearn PCA model, already trained; optional, used when loading test files
# 11) verbose: bool, whether to print out comments
#
# OUTPUTS
# 1) images: numpy array, contains images as 1D arrays
# 2) labels: numpy array, contains corresponding labels of the images
# 3) my_pca: sklearn PCA model
#
def baseline_load_images_pca(fname, label_col, fname_col, input_size, normalize, cropped, h_range=None, w_range=None, n_components=0.95, my_pca=None, verbose=False):

    images, labels = baseline_load_images(fname, label_col, fname_col, input_size, normalize, cropped, h_range, w_range)

    if my_pca is None:
        my_pca = PCA(n_components=n_components)
        my_pca.fit(images)
        if verbose:
            print('Printing explained variance ratio:')
            print(my_pca.explained_variance_ratio_)
            cum_var = np.cumsum(my_pca.explained_variance_ratio_)
            print('Printing explained variance ratio:')
            print(cum_var)

    images = my_pca.transform(images)

    return images, labels, my_pca
###################################################################################################



###################################################################################################
# FUNCTION TO PERFORM BASELINE CMC EVALUATION
#
# INPUTS
# 1) df: pandas dataframe
# 2) predictions: images as 1D numpy arrays (Pixels or Pixel PCA)
#
# OUTPUTS
# 1)
#
# adaptation of CMC function
def baseline_cmc(df, predictions, data_config, verbose=False):

    query_gallery_size = df[data_config['image_id_col']].max() + 1
    n_distractors = query_gallery_size - 2

    # calculate total num of galleries across all iterations
    galleries_per_iteraration = len(df[data_config['gallery_id']].unique())
    iterations = df[data_config['iteration_id']].unique().shape[0]
    
    total_galleries =  galleries_per_iteraration * iterations

    # get queries and galleries embeddings
    queries_emb = predictions[::query_gallery_size]
    pred_idx = np.arange(0, len(predictions))
    galleries_emb = predictions[np.mod(pred_idx, query_gallery_size) != 0]
    queries_emb = queries_emb.reshape(total_galleries, 1, -1)
    galleries_emb = galleries_emb.reshape(total_galleries, n_distractors + 1, -1 )

    # Calucluate distance
    cos_dist = tf.matmul(queries_emb, galleries_emb, transpose_b=True).numpy()
    euclid_dist = -(cos_dist - 1)

    # Calculate Rank
    r = np.argmin(np.argsort(euclid_dist), axis=2)
    r = np.squeeze(r)
    ranks = np.zeros(n_distractors)
    for i in range(n_distractors):
        ranks[i] = np.mean(r < (i + 1))

    return ranks
###################################################################################################



###################################################################################################
# FUNCTION FOR LOADING IMAGES FOR BASELINE CMC EVALUATION
#
# INPUTS
# 1) df: pandas dataframe, containing the image filenames or paths for gallery images
# 2) fname_col: string, name of column containin filenames or paths of images
# 3) input_size: int list, specifies dimensions of image input, channels last;
#                          if cropping, should match image size after crop
# 4) normalize: bool, specifies whether to normalize pixel values by dividing by 255
# 5) cropped: bool, specifies whether to crop images
# 6) h_range: int list, specifies height pixel position range, to be used if cropped == True
# 7) w_range: int list, specifies width pixel position range, to be used if cropped == True
#
# OUPTUTS
# 1) images: images as 1D arrays
#
def load_images_baseline_cmc(df, fname_col, input_size, normalize, cropped, h_range=None, w_range=None):
    
    images = np.zeros((df.shape[0],input_size[0],input_size[1],input_size[2])).reshape((df.shape[0], input_size[0]*input_size[1]*input_size[2]))
    for k, filename in enumerate(df[fname_col].values):
        img = np.array(Image2.open(filename))
        if normalize:
            img = img/255
        if cropped:
            images[k] = img[h_range[0]:h_range[1], w_range[0]:w_range[1],:].flatten().copy()
        else:
            images[k] = img.flatten().copy()

    return images
###################################################################################################



###################################################################################################
# FUNCTION FOR PERFORMING BASELINE CMC EVALUATION - MAIN
#
# INPUTS
# 1) df: pandas dataframe, contains galleries for cmc evaluation
# 2) eval_config: dictionary, contains all necessary parameters for evaluation; see evaluate_baseline_cmc
# 3) data_config: dictionary, contains all necessary parameters for loading data, including 'fname_col', 'input_size', 'normalize', 'cropped', 'h_range'
#                             and 'w_range'
#
def baseline_cmc_main(df, eval_config, data_config, my_pca=None, verbose=False):
    
    # load images
    img = load_images_baseline_cmc(df, data_config['fname_col'], data_config['input_size'], data_config['normalize'], data_config['cropped'], data_config['h_range'], data_config['w_range'])
    if my_pca is not None:
        img = my_pca.transform(img)
     # get ranks from cmc baseline
    ranks = baseline_cmc(df, img, data_config, verbose)
    return ranks
###################################################################################################


###################################################################################################
#
# 
#
def get_pca(eval_config, data_config, verbose=False):
    
    df_train = pd.read_csv(data_config['train_fname'])
    img_train = load_images_baseline_cmc(df_train,'filename', data_config['input_size'], True, data_config['cropped'], data_config['h_range'], data_config['w_range'])
    my_pca = PCA(n_components=0.95)
    my_pca.fit(img_train)
    if verbose:
        print('Printing explained variance ratio:')
        print(my_pca.explained_variance_ratio_)
        cum_var = np.cumsum(my_pca.explained_variance_ratio_)
        print('Printing explained variance ratio:')
        print(cum_var)
    return my_pca
###################################################################################################


###################################################################################################
# FUNCTION FOR PERFORMING BASELINE PIXEL
# Input images will be flattened to 1D arrays
#
# INPUTS
# 1) eval_config: dictionary, contains necessary parameters for performing evaluations, including 'methods', 'k_neighbors', 'per_class', 'conf_matrix'
# 2) data_config: dictionary, contains necessary parameters for loading data, including 'train_fname', 'test_fname', 'ref_name', 'label_col', 
#                             'fname_col', 'input_size', 'normalize', 'cropped', 'h_range', 'w_range', 'gallery_fname', 'iteration_id', 
#                             'iterations_per_calculations', 'n_distractors'
# 3) verbose: bool, whether to print out comments
#
# OUTPUTS
# 1) results: dictionary, contains scores for the evaluation methods performed
#
def baseline_pixel(eval_config, data_config, verbose=False):

    # if using all methods, reformat methods arg    
    if eval_config['methods'] == 'all':
        eval_config['methods'] = ['gallery', 'knn', 'centroid', 'svm']
        
    results = {}
        
    if 'gallery' in eval_config['methods']:
        if verbose:
            print(f'Calculating Gallery CMC scores')
        # running in batches and averaging to avoid exhausting memory resources
        gal_df = pd.read_csv(data_config['gallery_fname'])
        N_iterations = gal_df[data_config['iteration_id']].unique().shape[0]
        N_IPC = data_config['iterations_per_calculation']
        N_r = int(np.ceil(N_iterations/N_IPC))
        ranks_list = np.zeros((N_r, data_config['n_distractors']))
        for index, k in enumerate(range(0, N_iterations, data_config['iterations_per_calculation'])):
            currIterations = np.arange(k,min(N_iterations,k+N_IPC))
            try:
                df_sub = gal_df[gal_df[data_config['iteration_id']].isin(currIterations)]
                ranks = baseline_cmc_main(df_sub, eval_config, data_config, verbose=verbose)
                ranks_list[index] = ranks.copy()
            except Exception as e:
                print(f'ERROR - {e}')
                if verbose:
                    print(f'(On iterations: {currIterations})')
        results['cmc_ranks'] = np.round(ranks_list.mean(axis=0),4)
        if verbose:
            print(f'CMC ranks: {results["cmc_ranks"]}')
            

    # LOAD IMAGES AND LABELS
    train_images, train_labels = baseline_load_images(data_config['train_fname'], data_config['label_col'], data_config['fname_col'], data_config['input_size'], data_config['normalize'], data_config['cropped'], data_config['h_range'], data_config['w_range'])
    #train_images, train_labels = 
    test_images, test_labels = baseline_load_images(data_config['test_fname'], data_config['label_col'], data_config['fname_col'], data_config['input_size'], data_config['normalize'], data_config['cropped'], data_config['h_range'], data_config['w_range'])

    label_list = list(set(test_labels))
    results['label_list'] = label_list

    if 'knn' in eval_config['methods']:            
        # BUILD KNN MODEL AND PREDICT
        if verbose:
            print(f'Training kNN classifier with k={eval_config["n_neighbors"]}')
        my_knn = KNeighborsClassifier(n_neighbors=eval_config['n_neighbors'])
        my_knn.fit(train_images, train_labels)
    
        knn_pred = my_knn.predict(test_images)
        knn_acc = np.round(np.sum([1 for pred, label in zip(knn_pred, test_labels) if pred == label])/test_labels.shape[0],4)
        # store results
        results['n_neighbors'] = eval_config['n_neighbors']
        results['knn'] = knn_acc
        if eval_config['per_class']:
            knn_class = np.zeros(len(label_list))
            for k, label in enumerate(label_list):
                mask = test_labels == label
                knn_class[k] = np.round(np.sum(knn_pred[mask]==test_labels[mask])/np.sum(mask),4)
            # store results
            results['knn_class'] = knn_class
        if eval_config['conf_matrix']:
            knn_conf = confusion_matrix(test_labels, knn_pred)
            results['knn_conf'] = knn_conf
        if verbose:
            print(f'kNN test accuracy: {knn_acc}')

    if 'centroid' in eval_config['methods']:
        # CALCULATE CENTROIDS AND PREDICT
        if verbose:
            print('Training Centroid classifier')
        my_centroids = calculate_label_centroids(train_images, train_labels, distance='euclid')
        centroid_pred = predict_label_by_centroid(test_images, my_centroids['label_centroids'], 'euclid', my_centroids['index_label_map'])
        centroid_acc = np.round(np.sum([1 for pred, label in zip(centroid_pred, test_labels) if pred == label])/test_labels.shape[0],4)
        # store results
        results['centroid'] = centroid_acc
        if eval_config['per_class']:
            centroid_class = np.zeros(len(label_list))
            for k, label in enumerate(label_list):
                mask = test_labels == label
                centroid_class[k] = np.round(np.sum(centroid_pred[mask]==test_labels[mask])/np.sum(mask),4)
            # store results
            results['centroid_class'] = centroid_class
        if eval_config['conf_matrix']:
            centroid_conf = confusion_matrix(test_labels, centroid_pred)
            results['centroid_conf'] = centroid_conf
        if verbose:
            print(f'Centroid test accuracy: {centroid_acc}')


    if 'svm' in eval_config['methods']:
        # TRAIN SVM CLASSIFIER AND PREDICT
        if verbose:
            print('Training SVM classifier')
        my_svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        my_svm.fit(train_images, train_labels)
        svm_pred = my_svm.predict(test_images)
        svm_acc = np.round(np.sum([1 for pred, label in zip(svm_pred, test_labels) if pred == label])/test_labels.shape[0],4)
        results['svm'] = svm_acc
        if eval_config['per_class']:
            svm_class = np.zeros(len(label_list))
            for k, label in enumerate(label_list):
                mask = test_labels == label
                svm_class[k] = np.round(np.sum(svm_pred[mask]==test_labels[mask])/np.sum(mask),4)
            # store results
            results['svm_class'] = svm_class
        if eval_config['conf_matrix']:
            svm_conf = confusion_matrix(test_labels, svm_pred)
            results['svm_conf'] = svm_conf

        if verbose:
            print(f'SVM test accuracy: {svm_acc}')

    return results
###################################################################################################   



###################################################################################################
# FUNCTION FOR PERFORMING BASELINE EVALUATION USING PIXEL PCA INPUTS
# Input images will be flattened to 1D arrays, then transformed using PCA
# For open set setting, the function will use train set for PCA, and then build classifier models
# using the reference set
#
# INPUTS
# 1) eval_config: dictionary, contains necessary parameters for performing evaluations, including 'methods', 'n_compomemts', 'k_neighbors', 'per_class', 'conf_matrix'
# 2) data_config: dictionary, contains necessary parameters for loading data, including 'train_fname', 'test_fname', 'ref_name', 'label_col', 
#                             'fname_col', 'input_size', 'normalize', 'cropped', 'h_range', 'w_range', 'gallery_fname', 'iteration_id', 
#                             'iterations_per_calculations', 'n_distractors', ''
# 3) verbose: bool, whether to print out comments
#
# OUTPUTS
# 1) results: dictionary, contains scores for the evaluation methods performed
#
def baseline_pca(eval_config, data_config, verbose=False):

    # if using all methods, reformat methods arg    
    if eval_config['methods'] == 'all':
        eval_config['methods'] = ['gallery', 'knn', 'centroid', 'svm']

    # for open set setting, use ref(erence) set samples to train classifiers, not train set
    if data_config['ref_fname'] is None:
        use_ref = False
    else:
        if verbose:
            print('Using reference set')
        use_ref = True
        
    results = {}
    
    if 'gallery' in eval_config['methods']:
        if verbose:
            print(f'Calculating Gallery CMC scores')
        # running in batches and averaging to avoid exhausting memory resources
        gal_df = pd.read_csv(data_config['gallery_fname'])
        N_iterations = gal_df[data_config['iteration_id']].unique().shape[0]
        N_IPC = data_config['iterations_per_calculation']
        N_r = int(np.ceil(N_iterations/N_IPC))
        ranks_list = np.zeros((N_r, data_config['n_distractors']))
        my_pca = get_pca(eval_config, data_config, verbose)
        for index, k in enumerate(range(0, N_iterations, N_IPC)):
            currIterations = np.arange(k,min(N_iterations,k+N_IPC))
            try:
                df_sub = gal_df[gal_df[data_config['iteration_id']].isin(currIterations)]
                ranks = baseline_cmc_main(df_sub, eval_config, data_config, my_pca, verbose)
                ranks_list[index] = ranks.copy()
            except Exception as e:
                print(f'ERROR - {e}')
                if verbose:
                    print(f'(On iterations: {currIterations})')
        results['cmc_ranks'] = np.round(ranks_list.mean(axis=0),4)
        if verbose:
            print(f'CMC ranks: {results["cmc_ranks"]}')

    # LOAD IMAGES AND LABELS
    train_images, train_labels, my_pca = baseline_load_images_pca(data_config['train_fname'], data_config['label_col'], data_config['fname_col'], data_config['input_size'], data_config['normalize'], data_config['cropped'], data_config['h_range'], data_config['w_range'], eval_config['n_components'], verbose=verbose)
    if use_ref:
        ref_images, ref_labels, my_pca = baseline_load_images_pca(data_config['ref_fname'], data_config['label_col'], data_config['fname_col'], data_config['input_size'], data_config['normalize'], data_config['cropped'], data_config['h_range'], data_config['w_range'], eval_config['n_components'], my_pca, verbose=verbose)
        test_images, test_labels, my_pca = baseline_load_images_pca(data_config['test_fname'], data_config['label_col'], data_config['fname_col'], data_config['input_size'], data_config['normalize'], data_config['cropped'], data_config['h_range'], data_config['w_range'], eval_config['n_components'], my_pca, verbose=verbose)
    else:
        test_images, test_labels, my_pca = baseline_load_images_pca(data_config['test_fname'], data_config['label_col'], data_config['fname_col'], data_config['input_size'], data_config['normalize'], data_config['cropped'], data_config['h_range'], data_config['w_range'], eval_config['n_components'], my_pca, verbose=verbose)

    label_list = list(set(test_labels))
    results['label_list'] = label_list
    

    if 'knn' in eval_config['methods']:
        # BUILD KNN MODEL AND PREDICT
        if verbose:
            print(f'Training kNN classifier with k={eval_config["n_neighbors"]}')
        my_knn = KNeighborsClassifier(n_neighbors=eval_config["n_neighbors"])
        if use_ref:
            my_knn.fit(ref_images, ref_labels)
        else:
            my_knn.fit(train_images, train_labels)

        knn_pred = my_knn.predict(test_images)
        knn_acc = np.round(np.sum([1 for pred, label in zip(knn_pred, test_labels) if pred == label])/test_labels.shape[0],4)
        # store results
        results['n_neighbors'] = eval_config['n_neighbors']
        results['knn'] = knn_acc
        if eval_config['per_class']:
            knn_class = np.zeros(len(label_list))
            for k, label in enumerate(label_list):
                mask = test_labels == label
                knn_class[k] = np.round(np.sum(knn_pred[mask]==test_labels[mask])/np.sum(mask),4)
            # store results
            results['knn_class'] = knn_class
        if eval_config['conf_matrix']:
            knn_conf = confusion_matrix(test_labels, knn_pred)
            results['knn_conf'] = knn_conf
        if verbose:
            print(f'kNN test accuracy: {knn_acc}')


    if 'centroid' in eval_config['methods']:
        # CALCULATE CENTROIDS AND PREDICT
        if verbose:
            print('Training Centroid classifier')
        if use_ref:
            my_centroids = calculate_label_centroids(ref_images, ref_labels, distance='euclid')
        else:
            my_centroids = calculate_label_centroids(train_images, train_labels, distance='euclid')
        centroid_pred = predict_label_by_centroid(test_images, my_centroids['label_centroids'], 'euclid', my_centroids['index_label_map'])
        centroid_acc = np.round(np.sum([1 for pred, label in zip(centroid_pred, test_labels) if pred == label])/test_labels.shape[0], 4)
        # store results
        results['centroid'] = centroid_acc
        if eval_config['per_class']:
            centroid_class = np.zeros(len(label_list))
            for k, label in enumerate(label_list):
                mask = test_labels == label
                centroid_class[k] = np.round(np.sum(centroid_pred[mask]==test_labels[mask])/np.sum(mask),4)
            # store results
            results['centroid_class'] = centroid_class
        if eval_config['conf_matrix']:
            centroid_conf = confusion_matrix(test_labels, centroid_pred)
            results['centroid_conf'] = centroid_conf
        if verbose:
            print(f'Centroid test accuracy: {centroid_acc}')

    if 'svm' in eval_config['methods']:
        # TRAIN SVM CLASSIFIER AND PREDICT
        if verbose:
            print('Training SVM classifier')
        my_svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        if use_ref:
            my_svm.fit(ref_images, ref_labels)
        else:
            my_svm.fit(train_images, train_labels)
        svm_pred = my_svm.predict(test_images)
        svm_acc = np.round(np.sum([1 for pred, label in zip(svm_pred, test_labels) if pred == label])/test_labels.shape[0],4)
        results['svm'] = svm_acc
        if eval_config['per_class']:
            svm_class = np.zeros(len(label_list))
            for k, label in enumerate(label_list):
                mask = test_labels == label
                svm_class[k] = np.round(np.sum(svm_pred[mask]==test_labels[mask])/np.sum(mask),4)
            # store results
            results['svm_class'] = svm_class
        if eval_config['conf_matrix']:
            svm_conf = confusion_matrix(test_labels, svm_pred)
            results['svm_conf'] = svm_conf

        if verbose:
            print(f'SVM test accuracy: {svm_acc}')

    return results
###################################################################################################



###################################################################################################
# FUNCTION THAT GENERATES LATEX TABLE
#
# Generates table to display top1, top3, knn, centroid, svm results
# Make sure that df argument contains only rows you wish displayed
# column named 'key' should contain attribute to be used to give table rows their names
#
# INPUTS
# 1) df: pandas dataframe containing the following column names - (key, top1, top3, knn, centroid, svm)
# 2) n_neighbors: int, specifies the number of neighbors used for knn results
#
# OUTPUTS
# 1) table: string, defining table in latex
#
def get_latex_table(df, n_neighbors=3):

    table = '''\\begin{table}[h!]
                \centering
                \\begin{tabular}{|c  || c | c || c | c | c |} 
                \hline'''
    table = table + '\nModel & top1 & top3 & '+str(ne_neighbors)+'NN Acc & Centroid Acc & SVM Acc  \\\ \n\hline\n'


    #(key, date, dataset, model_folder, model_type, cropped, top1, top3, knn, centroid, svm)

    for index, row in df.iterrows():
        temp = row['key'] + '&' + row['top1'] + '&' + row['top3'] + '&' + row['knn'] + '&' + row['centroid'] + '&' + row['svm'] + '\\\ \n\hline\n'
        table = table + temp

    table = table + '''\hline
                       \end{tabular}
                       \label{}
                       %\caption{}
                       \end{table}'''
    return table




