import numpy as np
import pandas as pd
import tensorflow as tf
from evaluation import model_evaluation
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time
from datetime import datetime
import argparse
import yaml
import os
import pickle

#########################################################################
#
# REVISED CODE TO RUN MODEL EVALUATION
#
#########################################################################


def eval_function(config_file):
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        eval_config = config['eval_settings']
        data_config = config['data_settings']
        numpy_seed = config['numpy_seed']
        verbose = config['verbose']
    except Exception as e:
        print('ERROR - unable to open experiment config file. Terminating.')
        print('Exception msg:',e)
        return -1

    # get date of evaluation
    months = {'01':'JAN', '02':'FEB', '03':'MAR', '04':'APR', '05':'MAY', '06':'JUN', '07':'JUL', '08':'AUG', '09':'SEP', '10':'OCT', '11':'NOV', '12':'DEC'}
    now = datetime.now() # current date and time
    d = now.strftime('%y-%m-%d')
    d = d.split('-')
    date = d[0]+months[d[1]]+d[2]

    if verbose:
        print('Performing model evaluations')
    np.random.seed(numpy_seed)

    if eval_config['methods'] == 'all':
        eval_config['methods'] = ['gallery', 'knn', 'centroid', 'svm']

    results_dir = eval_config['results_dir']
    if not results_dir.endswith('/'):
        results_dir+='/'

        
    loss_plot_fname = 'lossplot_' + str(eval_config['key']) + '.png'
        
    # plot model loss
    history = pd.read_csv(eval_config['history_fname'])
    x = range(history.shape[0])
    plt.plot(x, history.loss, label='loss')
    plt.plot(x, history.val_loss, label='val_loss')
    plt.savefig(results_dir+loss_plot_fname)
    plt.close()


    # load model
    model = load_model(eval_config['model_folder'], custom_objects={'tf': tf})
    # perform evaluations
    results = model_evaluation(model, eval_config, data_config, verbose)

    if verbose:
        print('Preparing results to store...')
    df = pd.read_csv(eval_config['eval_file'])
    if eval_config['key'] in df.key.values:
        print(f'WARNING - evaluation results for model with key value {eval_config["key"]} already exist. Will result in duplication.')
        print('Please examine results table and remove any rows if necessary')
        
    # table format: (key, date, dataset, split_type, model_folder, model_type, cropped, top1, top3, knn, centroid, svm)
    
    #['key', 'date', 'dataset', 'split_type', 'model_fname', 'model_type', 'cropped', 'top1', 'top3', 'knn', 'centroid', 'svm']
    
    data = [eval_config['key'], date, data_config['dataset'], data_config['split_type'], eval_config['model_folder'], eval_config['model_type'], data_config['cropped']]
    # data = data + [results[m] for m in methods]
    if 'gallery' in eval_config['methods']:
        data+= [results['top1'], results['top3']]
    else:
        data+= [None, None]
    if 'knn' in eval_config['methods']:
        data.append(results['knn'])
    else:
        data.append(None)
    if 'centroid' in eval_config['methods']:
        data.append(results['centroid'])
    else:
        data.append(None)
    if 'svm' in eval_config['methods']:
        data.append(results['svm'])
    else:
        data.append(None)
    df_new = pd.DataFrame([data])
    df_new.columns = df.columns
    df_combined = pd.concat([df, df_new], ignore_index=True)
    df_combined.to_csv(eval_config['eval_file'], index=False)
    if verbose:
        print(f'{eval_config["eval_file"]} updated with new results')

        
    if 'gallery' in eval_config['methods']:
        eval_config['methods'] = eval_config['methods'][1:]

    # if per class, store results
    if eval_config['per_class']:
        X = {m:results[m+'_class'] for m in eval_config['methods']}
        X['label'] = results['label_list']
        df_class = pd.DataFrame(X)
        fname = results_dir + date + '_' + str(eval_config['key'])  + '_'  + '_class.csv'
        if verbose:
            print(f'storing pixel per class results in {fname}')
        df_class.to_csv(fname)

    # if conf matrix, store
    if eval_config['conf_matrix']:
        conf_M = {m:results[m+'_conf'] for m in eval_config['methods']}
        fname = results_dir + date + '_' + str(eval_config['key'])  + '_'  + '_conf_matrix.pkl'
        if verbose:
            print(f'storing pixel confusion matrices in {fname}')
        with open(fname, 'wb') as f:
            pickle.dump(conf_M, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print('Finished with model evaluations')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file", help="yaml file with baseline eval settings", type=str)
    args = parser.parse_args()

    # ADD PRINT OF DATE AND TIME
    now = datetime.now() # current date and time
    date_time = now.strftime("%y-%m-%d %H:%M")
    print(f'Date and time when this evaluation was started: {date_time}')
    eval_function(args.config_file)
    



