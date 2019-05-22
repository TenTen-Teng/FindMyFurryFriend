
# coding: utf-8

# # Helper
# 
# * **Project name:** cat and dog classifier
# * **Author:** Teng Li
# * ** Date:** 05.10.2019
# 
# This notebook is used to build some functions 

# In[1]:

import matplotlib.pyplot as plt
import itertools
import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.framework import graph_util, graph_io


# In[ ]:

def generate_image_path(input_path, output_path, 
                        col_names=['image_path', 'ground_truth'], 
                        save=True, 
                        shuffle=True):
    """
    Generate image paths from local path and convert to a csv file
    :param input_path - string: image paths
    :param output_path - string: output path
    :param col_names - list: a list of column names
    :param save - boolean: boolean value that determine save dataframe as a csv file to current path or not
    :param shuffle - boolean: boolean value that determine if shuffle or not
    
    :return: a df file that contains image paths and ground truth labels
    """

    assert os.path.exists(input_path), 'The input data source doesn\'t exist, please check input path'
    
    paths = []
    for dirpath, subdirs, files in os.walk(input_path):
        for x in files:
            if x.endswith((".jpg", ".png" , ".JPG")):
                paths.append((os.path.join(dirpath, x), dirpath.split('/')[-1]))

    if len(col_names) < 2:
        print('Use default column names: \"image_path\", \"ground_truth\"')
        col_names=['image_path', 'ground_truth']
        
    df = pd.DataFrame(paths, columns=col_names)
        
    if shuffle:
        df = df.sample(frac=1)
    
    if save:
        if output_path:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        else:
            output_path = './data'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
                
        df.to_csv(output_path + 'image_path.csv', sep='\t')

    return df


# In[31]:

def check_image_dim(df, col_image, output_path, save=True):
    """
    Check image dimension, remove images with only 2 dimensions
    
    :param df - dataframe: the original dataframe created from the data source directory
    :param col_image - str: the column name of image path in original df
    """
    
    image_paths = df[col_image]
    
    df_output = df.copy()
    
    print('checking wrong dimension images...')
    for path in tqdm(image_paths):        
        img = Image.open(path)
        img = np.array(img)
    
        if img.ndim == 2:
            df_output = df_output[df_output[col_image] != path]
    
    if save:
        if output_path:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        else:
            output_path = './data'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
                
        df_output.to_csv(output_path + 'image_path_checked.csv', sep='\t')
        
    return df_output


# In[ ]:

def prepare_data(df, output_path, ratio_train=0.8, ratio_val=0.1, ratio_test=0.1, save=True):
    """
    Seperate dataframe format data to train, val, and test dataset
    
    :param df_path - str: the directory of dataframe that contain image paths and ground truth labels
    :param output_path - str: output csv files directory
    :param ratio_train - float: the ratio of train dataset to total dataset, default value is 0.8
    :param ratio_val - float: the ratio of validation dataset to total dataset, default value is 0.1
    :param ratio_test - float: the ratio of test dataset to total dataset, default value is 0.1
    :param save - bool: the bool value for determine save the result csv files to output directory or not, default value is True
    
    return:
    df_train - dataframe: a training dataset dataframe
    df_val - dataframe: a validation dataset dataframe
    df_test - dataframe: a test dataset dataframe
    """

    # random select sample images from csv
    num_total = len(df)
    
    assert float(ratio_train)+float(ratio_val)+float(ratio_test) == 1, 'please check the dataset separation ratio, make sure the sum is equal to 1'
    
    num_train = int(num_total * float(ratio_train))
    num_val = int(num_total * float(ratio_val))
    num_test = int(num_total * float(ratio_test))

    df_train = df.sample(n=num_train, random_state=123)
    df_rest = df[~df.index.isin(df_train.index)]

    df_val = df_rest.sample(n=num_val, random_state=123)
    df_test = df_rest[~df_rest.index.isin(df_val.index)]

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)        
    df_test.reset_index(drop=True, inplace=True)
    
    if save:
        if output_path:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            df_train.to_csv(output_path + 'train_data.csv')
            df_val.to_csv(output_path + 'val_data.csv')
            df_test.to_csv(output_path + 'test_data.csv')
        else:
            output_path = '../data/csv_files/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            df_train.to_csv(output_path + 'train_data.csv')
            df_val.to_csv(output_path + 'val_data.csv')
            df_test.to_csv(output_path + 'test_data.csv')
    return df_train, df_val, df_test


# In[2]:

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)

    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.tight_layout()


# In[ ]:

def plot_results(path, history):
    """
    Plot accuracy and loss trendency graph
        
    :param path: -str, graph path
    :param history: model trained history
    """
    acc_name = 'accuracy.png'
    loss_name = 'loss.png'
    # Accuracy learning curves
    plt.figure(0)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(path+acc_name)
    plt.close()

    # Loss learning curves
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(path+loss_name)
    plt.close()

    print('training done!')


# In[3]:

def get_index(cat_dog_class, classes):
    if cat_dog_class in classes:
        index = classes.index(cat_dog_class)
    return index


# In[2]:

def keras_to_tensorflow(keras_model_path, tf_model_path, tf_output_name, num_output_layer, 
                        prefix='k2tf_output', write_ascii=False):
    """
    :param num_output_layer - int: the number of outputs in the model
    :prefix - str: The prefix for the output aliasing
    :write_ascii - boolean: 
    """
    
    '''load keras model and rename output'''
    K.set_learning_phase(0)
    model = load_model(keras_model_path)

    pred = [None] * num_output_layer
    pred_node_names = [None] * num_output_layer
    
    for i in range(num_output_layer):
        pred_node_names[i] = prefix+'_'+str(i)
        pred[i] = tf.identity(model.output[i], name=pred_node_names[i])
    print('Output nodes names are: ', pred_node_names)
    
    '''write graph definition in ascii'''
    sess = K.get_session()

    '''make path for saving trained model'''
    if not os.path.exists(tf_model_path):
        os.makedirs(tf_model_path)
        
    if write_ascii:
        f = 'graph_def.pb.ascii'
        tf.train.write_graph(sess.graph.as_graph_def(), tf_model_path, f, as_text=True)
        print('saved the graph definition in ascii format at: ', os.path.join(tf_model_path, f))
        
    '''convert variables to constants and save'''
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, tf_model_path, tf_output_name, as_text=False)
    print('saved the constant graph (ready for inference) at: ', os.path.join(tf_model_path, tf_output_name))

