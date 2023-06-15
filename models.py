import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial
from sklearn.utils import shuffle
import os
from tensorflow.keras.models import load_model
from tensorflow_addons.losses import TripletSemiHardLoss, TripletHardLoss
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential, load_model
import time
from datetime import datetime


###########################################################################
#
# FUNCTIONS FOR BUILDING MODELS
# BASE CODE FROM https://github.com/jachansantiago/bee_reid
# WITH SOME MODIFICATIONS AND ADDITIONS
#
###########################################################################




###################################################################################################
# FUNCTION TO GET INDEX OF A MODEL LAYER GIVEN ITS NAME
#
# INPUTS
# 1) model: a TF model
# 2) layer_name: string, the name of a layer in the model
#
# OUTPUTS
# 1) layer index if layer_name found, else -1
#
def get_layer_index(model, layer_name):
    for k, layer in enumerate(model.layers):
        if layer.name == layer_name:
            return k
    print(f'ERROR - Could not find specified layer in model: {layer_name}')
    return -1
###################################################################################################


###################################################################################################
# FUNCTION FOR SETTING THE TRAINABILITY OF LAYERS
# USED WHEN FINETUNING MODEL AND WHEN first_trainable_layer_name IS SET
# ALL BN LAYERS ARE SET TO UNTRAINABLE, AS WELL AS ALL LAYERS BEFORE FIRST_TRAINABLE_LAYER_NAME
#
# INPUTS
# 1) model: a TF model
# 2) first_trainable_layer: string, name of first trainable layer in model
#
# OUTPUTS
# 1) model: a TF model
#
def set_trainable_layers(model, first_trainable_layer_name):

    first_index = get_layer_index(model, first_trainable_layer_name)
    for k, layer in enumerate(model.layers):
        # keep all BN layers untrainable
        if layer.name[-2:] == 'bn':
            model.layers[k].trainable = False
        elif k < first_index:
            model.layers[k].trainable = False
        else:
            model.layers[k].trainable = True
    return model
###################################################################################################


###################################################################################################
# TF MODEL FOR UCL
#
# Requires base_model (backbone), and a temperature (with default value of 0.01)
#
class ContrastiveLearning(tf.keras.Model):
    def __init__(self, base_model, temperature=0.01):
        super(ContrastiveLearning, self).__init__()
        self.backbone = base_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.valid_loss_tracker = tf.keras.metrics.Mean(name="valid_loss")
        self.temperature = temperature
        self.model_name = "ConstrastiveLearning"

    def call(self, data):
        x = data
        x = self.backbone(x)
        return x

    def train_step(self, data):
        x1, x2, y = data

        with tf.GradientTape() as tape:
            # get embeddings
            x1 = self(x1, training=True)
            x2 = self(x2, training=True)
            # calculate similarities
            sim_matrix1 = tf.matmul(x1, x2, transpose_b=True)/ self.temperature
            sim_matrix2 = tf.transpose(sim_matrix1)
            # calculate loss
            loss1 = tfa.losses.npairs_loss(y_pred=sim_matrix1, y_true=y)
            loss2 = tfa.losses.npairs_loss(y_pred=sim_matrix2, y_true=y)
            loss = loss1 + loss2

        trainable_vars = self.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)
        self.loss_tracker.update_state(loss)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x1, x2, y = data

        x1 = self(x1, training=False)
        x2 = self(x2, training=False)

        sim_matrix1 = tf.matmul(x1, x2, transpose_b=True)/ self.temperature
        sim_matrix2 = tf.transpose(sim_matrix1)

        loss1 = tfa.losses.npairs_loss(y_pred=sim_matrix1, y_true=y)
        loss2 = tfa.losses.npairs_loss(y_pred=sim_matrix2, y_true=y)
        loss = loss1 + loss2

        self.valid_loss_tracker.update_state(loss)

        return {"loss": self.valid_loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.valid_loss_tracker]
###################################################################################################


###################################################################################################
# FUNCTION FOR BUILDING UCL MODEL
# 
# INPUTS
# 1) model_config: dictionary, contains the necessary arguments for building a model;
#                  see build_simple_model() for required info
# 2) input_shape: int list, input dimensions (channels last)
# 3) norm_method: int, specifies how data is normalized; see load_image() for more information
#
# OUTPUTS
# 1) contrastive_model: a TF model
#
def build_contrastive_model(model_config, input_shape, norm_method):
    # get model ()
    model = build_simple_model(model_config, input_shape, norm_method)
    contrastive_model = ContrastiveLearning(model, model_config['temperature'])
    return contrastive_model
###################################################################################################


###################################################################################################
# FUNCTION FOR GETTING MODEL BACKBONE
# CURRENTLY AVAILABLE: RESNET50V2, EFFICIENTNETB0, MOBILENET
#
# INPUTS
# 1) model_config: dictionary, contains the necessary arguments for building a model, including 'GAP', 'dropout', 'dropout_first', 'dropout_last', 'latent_dim',
#                  'output_layer_name', 'finetune', 'backbone'
# 2) input_shape: int list, input dimensions (channels last)
# 3) norm_method: int, specifies how data is normalized; see load_image() for more information
#                      if norm_method == 0, apply backbone specific pre-processing layer
#
# OUTPUTS
# 1) inputs: input layer to model
# 2) x: a TF model
# 3) model_name: string, name of model
#
def get_backbone(model_config, input_shape, norm_method):

    if norm_method > 0:
        inputs = Input(input_shape)
        x = inputs
    else:
        inputs = Input(input_shape)
    # get backbone from tensorflow keras with corresponding preprocessing layers
    try:
        if model_config['backbone'] == 'resnet50v2':
            print(input_shape)
            backbone = tf.keras.applications.ResNet50V2(include_top=False, input_shape=input_shape, weights=model_config['weights'])
            if norm_method == 0:
                x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
            model_name = 'ResNet50V2'
        elif model_config['backbone'] == 'efficientnetb0':
            backbone = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape, weights=model_config['weights'])
            if norm_method == 0:
                x = tf.keras.applications.efficientnet.preprocess_input(inputs)
            model_name = 'EfficientNetB0'
        elif model_config['backbone'] == 'mobilenet':
            backbone = tf.keras.applications.mobilenet.MobileNet(include_top=False, input_shape=input_shape, weights=model_config['weights'])
            if norm_method == 0:
                x = tf.keras.applications.mobilenet.preprocess_input(inputs)
            model_name = 'MobileNet'
    except Exception as e:
        print(f'ERROR - unable to load backbone {model_config["backbone"]}\nException msg: {e}\nTerminating code...')
        return -1

    if model_config['output_layer_name'] is not None:
        out_index = get_layer_index(backbone, model_config['output_layer_name'])
        if out_index >=0:
            backbone = Model(backbone.input, backbone.layers[out_index].output)
        else:
            print(f'WARNING - model layer {model_config["output_layer_name"]} not found. Using entire model as base.')

    # use value of None to specify that entire model is trainable
    if model_config['first_trainable_layer_name'] is None:
        x = backbone(x)
    # else, there is a specified first layer; make all layers before it untrainable
    # should be used with pre-trained weights
    else:
        backbone = set_trainable_layers(backbone, model_config['first_trainable_layer_name'])
        x = backbone(x)    

    return inputs, x, model_name
###################################################################################################




###################################################################################################
# FUNCTION FOR ADD REID HEAD TO BACKBONE
#
# INPUTS
# 1) x: a TF model (backbone)
# 1) model_config: dictionary, contains the necessary arguments for building a head, including 'GAP', 'dropout', 'dropout_first', 'dropout_last', 'latent_dim'
#
# OUTPUTS
# 1) x: a TF model
#
def add_model_head(x, model_config):
    
    # default head
    if model_config['head'] is None:
        # add trainable top to model
        if model_config['GAP']:
            x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        if model_config['dropout']:
            x = Dropout(model_config['dropout_first'])(x)
        x = Dense(model_config['latent_dim'])(x)
        if model_config['dropout']:
            x = Dropout(model_config['dropout_last'])(x)
        if model_config['l2_norm']:
            x = tf.keras.layers.Lambda(lambda y: tf.math.l2_normalize(y, axis=1))(x)
    else:
        print(f'ERROR - specified head type is not available: {model_config["head"]}')
        return -1
        
    return x
##################################################################################################



###################################################################################################
# FUNCTION FOR BUILDING SIMPLE MODEL FOR SCL
# COMPOSED OF A CNN BACKBONE PLUS A REID HEAD
#
# INPUTS
# 1) model_config: dictionary, contains the necessary arguments for building a model, including 'output_layer_name' and 'first_trainable_layer_name'
#                   see get_backbone() and add_model_head() for more required info
# 2) input_shape: int list, input dimensions (channels last)
# 3) norm_method: int, specifies how data is normalized; see load_image() for more information
#
# OUTPUTS
# 1) model: a TF model
#
def build_simple_model(model_config, input_shape, norm_method):

    # get backbone
    inputs, x, model_name = get_backbone(model_config, input_shape, norm_method)

    # add trainable top to model
    x = add_model_head(x, model_config)

    # change model name strategy
    if model_config['output_layer_name'] is not None:
        model_name = model_name + f'_trimmed_output_{model_config["output_layer_name"]}'
    if model_config['first_trainable_layer_name'] is not None:
        model_name = model_name + f'_firstTrainable_{model_config["first_trainable_layer_name"]}'

    model = Model(inputs, x, name=model_name)
    return model
###################################################################################################
