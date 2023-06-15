import numpy as np
import pandas as pd
import os
from evaluation import baseline_pixel, baseline_pca
import time
from datetime import datetime
import argparse
import yaml
import pickle

#########################################################################
#
# CODE TO RUN BASELINE EVALUATION USING BOTH PIXEL AND PCA REPRESENTATION
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

    # get data
    months = {'01':'JAN', '02':'FEB', '03':'MAR', '04':'APR', '05':'MAY', '06':'JUN', '07':'JUL', '08':'AUG', '09':'SEP', '10':'OCT', '11':'NOV', '12':'DEC'}
    now = datetime.now() # current date and time
    d = now.strftime('%y-%m-%d')
    d = d.split('-')
    date = d[0]+months[d[1]]+d[2]

    print('HERE')
    if verbose:
        print('Performing baseline evaluations')
    np.random.seed(numpy_seed)

    # specify eval method if 'all'
    if eval_config['methods'] == 'all':
        eval_config['methods'] = ['gallery', 'knn', 'centroid', 'svm']

    # performing PCA baseline
    if eval_config['pca_baseline']:
        if verbose:
            print('PCA baseline...')
        results_pca = baseline_pca(eval_config, data_config, verbose=verbose)
    # performing Pixel baseline
    if eval_config['pixel_baseline']:
        if verbose:
            print('Pixel baseline...')
        results_pixel = baseline_pixel(eval_config, data_config, verbose=verbose)

    # storing results
    if verbose:
        print('Preparing results to store...')
    df = pd.read_csv(eval_config['eval_file'])
    # create key for results
    key_init = df.shape[0]
    
    
    if 'gallery' in eval_config['methods']:
        methods = ['top1', 'top3']
        del methods['gallery']
    else:
        methods = []
    methods = methods + [m for m in eval_config['methods'] if m != 'gallery']
     
    # format results for saving
    data = []
    # PCA RESULTS
    if eval_config['pca_baseline']:
        pca_key = str(key_init)
        if len(pca_key) < 3:
            pca_key = '0'*(3-len(pca_key)) + pca_key
        pca_key = 'B'+ pca_key
        key_init+=1
        
        # row structure: ['key', 'date', 'baseline_type', 'dataset', 'split_type', 'cropped', 'top1', 'top3', 'knn', 'centroid', 'svm']
        data_pca =  [pca_key, date, 'pca', data_config['dataset'], data_config['split_type'], data_config['cropped']]
        data_pca = data_pca + [results_pca[m] for m in eval_config['methods']]
        data.append(data_pca)
    # PIXEL RESULTS
    if eval_config['pixel_baseline']:
        pixel_key = str(key_init)
        if len(pixel_key) < 3:
            pixel_key = '0'*(3-len(pixel_key)) + pixel_key
        pixel_key = 'B'+ pixel_key
        data_pixel = [pixel_key, date, data_config['dataset'], data_config['split_type'], data_config['cropped']]
        data_pixel = data_pixel + [results_pixel[m] for m in eval_config['methods']]
        data.append(data_pixel)
    df_new = pd.DataFrame(data)
    df_new.columns = df.columns
    df_combined = pd.concat([df, df_new], ignore_index=True)
    df_combined.to_csv(eval_file, index=False)
    if verbose:
        print(f'{eval_file} updated with new results')    

    if not results_dir.endswith('/'):
        results_dir+='/'

    # if per class, store results
    if eval_config['per_class']:
        if eval_config['ref_fname'] is None:
            X = {m:results_pixel[m+'_class'] for m in eval_config['methods']}
            X['label'] = results_pixel['label_list']
            df_class = pd.DataFrame(X)
            fname = results_dir + date + '_K' + str(key+1) + '_baseline_pixel_class.csv'
            if verbose:
                print(f'storing pixel per class results in {fname}')
            df_class.to_csv(fname)

        Y = {m:results_pca[m+'_class'] for m in eval_config['methods']}
        Y['label'] = results_pca['label_list']
        df_class_2 = pd.DataFrame(Y)
        fname_2 = results_dir + date + '_K' + str(key) + '_baseline_pca_class.csv'
        if verbose:
            print(f'storing pca per class results in {fname_2}')
        df_class_2.to_csv(fname_2)

    # if conf matrix, store
    if eval_config['conf_matrix']:
        if eval_config['ref_fname'] is None:
            conf_M = {m:results_pixel[m+'_conf'] for m in eval_config['methods']}
            fname = results_dir + date + '_K' + str(key+1) + '_baseline_pixel_conf_matrix.pkl'
            if verbose:
                print(f'storing pixel confusion matrices in {fname}')
            with open(fname, 'wb') as f:
                pickle.dump(conf_M, f, protocol=pickle.HIGHEST_PROTOCOL)

        conf_M_2 = {m:results_pca[m+'_conf'] for m in eval_config['methods']}
        fname_2 = results_dir  +  date + '_K' +  str(key)  + '_baseline_pca_conf_matrix.pkl'
        if verbose:
            print(f'storing pca confusion matrices in {fname_2}')
        with open(fname_2, 'wb') as f:
            pickle.dump(conf_M_2, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print('Finished with baseline evaluations')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file", help="yaml file with baseline eval settings", type=str)
    args = parser.parse_args()

    # ADD PRINT OF DATE AND TIME
    now = datetime.now() # current date and time
    date_time = now.strftime("%y-%m-%d %H:%M")
    print(f'Date and time when this evaluation was started: {date_time}')
    eval_function(args.config_file)



