import numpy as np
import pandas as pd
from models import *
from data import *
import pickle
import argparse
import yaml

###################################################################
#
# REVISED VERSION FOR TRAINING BASIC SINGLE MODELS ONCE
#
###################################################################


# function to load model
#    1) SCL: supervised contrastive learner, for reID using triplet loss with margin
def get_model(model_config, train_config, input_shape, norm_method, verbose):

    if model_config['model_type'] == 'SCL':
        if verbose:
            print('Building SCL...')
        model = build_simple_model(model_config, input_shape, norm_method)
    else:
        print(f'ERROR - invalid model type choice: {model_config["model_type"]}')
        print('Terminating code...')
        return -1
        model.compile(optimizer=tf.keras.optimizers.Adam(train_config['learning_rate']), loss=TripletSemiHardLoss(margin=train_config['margin']))

    return model

# function to load data
def get_data(data_config, model_type, verbose):

    if model_type == 'SCL':
        if verbose:
            print('Loading data for SCL')
        train_dataset, valid_dataset = load_data(data_config, verbose)
    else:
        print(f'ERROR - invalid model type choice: {model_type}')
        print('Terminating code...')
        return -1,-1
    return train_dataset, valid_dataset

# main function to train model
def train_function(config_file):
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        model_config = config['model_settings'] # settings for model building
        train_config = config['train_settings'] # settings for model training
        data_config = config['data_settings'] # settings for data loading
        config_dir = config['config_dir'] # directory where copy of config will be stored
        models_csv = config['models_csv'] # csv containing info about all trained models
        numpy_seed = config['numpy_seed']
        verbose = config['verbose']
    except Exception as e:
        print('ERROR - unable to open experiment config file. Terminating.')
        print('Exception msg:',e)
        return -1

    if verbose:
        # ADD PRINT OF DATE AND TIME
        now = datetime.now() # current date and time
        dt = now.strftime("%y-%m-%d %H:%M")
        print(f'Date and time when this experiment was started: {dt}')

    # yaml seems to read tuples as strings, so use lists instead and then convert to tuple
    data_config['input_size'] = tuple(data_config['input_size'])

    try:
        if verbose:
            print(f'Using numpy seed {numpy_seed}')
        np.random.seed(numpy_seed)
        if verbose:
            print('Getting data...')
        # if data not already split into train/valid dataframes, make sure valid_fname is None
        train_dataset, valid_dataset = get_data(data_config, model_config['model_type'], verbose)
        if verbose:
            print('Building and training model...')
        sample_batch = next(iter(valid_dataset.batch(32)))
        input_shape = sample_batch[0][0].numpy().shape
        if model_config['model_folder'] is None:
            model = get_model(model_config, train_config, input_shape, data_config['norm_method'], verbose)
            if model == -1:
                return -1
        else:
            # load previously trained model
            if model_config['model_folder'][-1] == "/":
                model_config['model_folder'] = model_config['model_folder'][:-1]
            model = load_model(os.path.join(model_config['model_folder'], "model.tf")) # tensorflow function
            _, model_name = os.path.split(model_config['model_folder'])
            model_name = "_".join(model_name.split("_")[1:])


        months = {'01':'JAN', '02':'FEB', '03':'MAR', '04':'APR', '05':'MAY', '06':'JUN', '07':'JUL', '08':'AUG', '09':'SEP', '10':'OCT', '11':'NOV', '12':'DEC'}
        now = datetime.now() # current date and time
        d,t = now.strftime('%y-%m-%d %H:%M').split()
        d,t = d.split('-'), t.split(':')
        date_time = d[0]+months[d[1]]+d[2]+'_' + t[0] + 'HR' + t[1] + 'MIN'

        # get dataframe containing info on past trained models
        df_models = pd.read_csv(models_csv)
        # construct key for current model
        model_key = str(df_models.shape[0])
        # zero padding
        if len(model_key) < 3:
            model_key = '0'*(3-len(model_key)) + model_key
        model_key = 'K'+model_key
        if verbose:
            print(f'model_key: {model_key}')

        # store config settings for future reference
        store_config = {'model_config':model_config, 'train_config':train_config, 'data_config':data_config, 'numpy_seed':numpy_seed}
        if config_dir[-1] != '/':
            config_dir+='/'
        config_copy_path = config_dir + model_key + '_config_copy.pkl'
        with open(config_copy_path, 'wb') as f:
            pickle.dump(store_config, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f'stored config settings in {config_copy_path}')
        
        # folders for storing results etc
        folder_name = date_time + '_' + model_key + '_' + model_config['model_type']
        finetuned = False
        if model_config['model_folder'] is not None:
            folder_name +=  '_finetuned'
            finetuned = True
        model_folder = os.path.join(model_config['models_dir'], folder_name)
        log_folder = os.path.join(model_folder, "logs")
        sm_folder = os.path.join(log_folder, "sensitivity_map")
        checkpoint_folder = os.path.join(model_folder, "checkpoints")
        metrics_filename = os.path.join(model_folder, "metrics.csv")
        checkpoint_format = os.path.join(checkpoint_folder, "{epoch:02d}-{val_loss:.2f}.hdf5")
        model_filename = os.path.join(model_folder, "model.tf")


        # Callbacks
        earlystop = tf.keras.callbacks.EarlyStopping(monitor=train_config['monitor'], patience=train_config['patience'], restore_best_weights=train_config['restore_best_weights'])
        checkpoints = tf.keras.callbacks.ModelCheckpoint(checkpoint_format, mode='auto', save_freq='epoch')
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_folder)
        metrics_loger = tf.keras.callbacks.CSVLogger(metrics_filename, separator=',', append=False)
        callbacks=[earlystop, checkpoints, tensorboard, metrics_loger]

        # Training
        start = time.time()
        history = model.fit(train_dataset.batch(train_config['batch_size']), validation_data=valid_dataset.batch(train_config['batch_size']), epochs=train_config['epochs'], callbacks=callbacks, verbose=2)
        stop = time.time()
        total_time = np.round((stop-start)/60.,2)
        
        if verbose:
            print(f'Finished Training')
            print(f'Training time: {total_time} minutes')

        # updating models_db with new trained model info
        #   format:  (key, datetime, model_type, finetune, model_folder, backbone, dataset, split_type, cropped, batch_size, augmentation, total_time, seed, config_copy_file)
        row_entry = [model_key, date_time, model_config['model_type'], finetuned, model_folder, model_config['backbone'], data_config['dataset'], data_config['split_type'], data_config['cropped'], train_config['batch_size'], data_config['augmentation'], total_time, config_copy_path]
        df_new = pd.DataFrame([row_entry])
        df_new.columns = df_models.columns
        df_combined = pd.concat([df_models, df_new], ignore_index=True)
        df_combined.to_csv(models_csv, index=False)

        if verbose:
            print('Storing model...')
        model.save(model_filename)
        if verbose:
            print('FINISHED.')


    except Exception as e:
        print('ERROR: not able to execute code completely')
        print('Exception msg:',e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file", help="yaml file with experiment settings", type=str)
    args = parser.parse_args()

    # ADD PRINT OF DATE AND TIME
    now = datetime.now() # current date and time
    date_time = now.strftime("%y-%m-%d %H:%M")
    print(f'Date and time when this experiment was started: {date_time}')
    train_function(args.config_file)
