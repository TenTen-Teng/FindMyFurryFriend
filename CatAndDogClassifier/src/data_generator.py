
# coding: utf-8

# # Data generator
# 
# * **Project name:** cat and dog classifier
# * **Author:** Teng Li
# * ** Date:** 05.10.2019

# In[1]:

import pandas as pd
import numpy as np
from PIL import Image


# In[17]:

"""Data generator class for generate data batch by batch based on the image path from CSV file"""
class DataGenerator:
    def __init__(self, 
                 df, 
                 crop_mode='center_crop',
                 short_edge_min=256, 
                 short_edge_max=384,
                 target_size=(224, 224), 
                 batch_size=16, 
                 x_col='image_path',
                 y_col='ground_truth',
                 shuffle=True):
        self.df = df
        self.short_edge_min = short_edge_min
        self.short_edge_max = short_edge_max
        self.crop_mode = crop_mode
        self.target_size = tuple(target_size)
        self.batch_size = batch_size    
        self.x_col = x_col
        self.y_col = y_col
        self.shuffle = shuffle
        self.classes = sorted(list(set(df[self.y_col].tolist())))
        self.nb_classes = len(self.classes)
        self.num = len(self.df)
        
        assert self.x_col in self.df.columns, 'please check image path column name in csv file'
        assert self.y_col in self.df.columns, 'please check ground truth column name in csv file'
        
    def _path_converter(self, df):
        '''
        The function is used to convert image path column to a list of image path and convert the label into a one hot vector
        
        :param df - dataframe: the dataframe that contains all the image path and ground truth
        
        return: a list of image paths and the ground truth one hot vector
        '''
        
        paths = df[self.x_col].tolist()
        
        ground_truths = df[self.y_col].tolist()
        ground_truth_one_hot_dict = {}
        
        for i, classification in enumerate(self.classes):
            # build the ohe arrays
            one_hot = [0]*i + [1] + [0]*(len(self.classes)-i-1)
            ground_truth_one_hot_dict[classification] = one_hot

        ground_truths_nhe = []
        for ground_truth in ground_truths:
            ground_truths_nhe.append(ground_truth_one_hot_dict[ground_truth])
            

        paths_gt_list = [paths, ground_truths_nhe]
        return paths_gt_list
    
    def _load_img_pair(self, 
                       paths, 
                       target_size=None, 
                       short_edge_min=None, 
                       short_edge_max=None, 
                       crop_mode='center_crop'):
        '''
        The function is used to load image, downsample crop the images
        
        :param paths - str: the path of image
        :param target_size - tuple: the target size after crop
        :param short_edge_min - int: a resize scale for downsample
        :param short_edge_max - int: a resize scale for downsample
        :param crop_mode - str: crop mode for cropping. Two options: center crop and random crop
        
        return img - np.array
        '''
        img = Image.open(paths)

        if target_size:
            img = img.resize(target_size)

        # decide short_edge_resize if short_edge_min has been assigned
        if short_edge_min:
            # if short_edge_max has been assigned, choose short_edge_resize from the range [short_edge_min, short_edge_max]
            if short_edge_max:
                short_edge_resize = np.random.randint(short_edge_min, short_edge_max)
            # otherwise use short_edge_min to specify short_edge_resize
            else:
                short_edge_resize = short_edge_min

            # resize the original image by computing the resizing ratio according to short_edge_resize
            width, height = img.size
            tmp = min(width, height)        
            width = int(width * short_edge_resize / tmp)
            height = int(height * short_edge_resize / tmp)
            img = img.resize((width, height))

        img = np.array(img)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        
        img = img[:, :, :3]
        
        if crop_mode == 'center_crop':
            # center crop image
            centerw, centerh = img.shape[1]//2, img.shape[0]//2
            halfw, halfh = target_size[1]//2, target_size[0]//2

            offsetw, offseth = 0, 0
            if target_size[0] % 2 == 1:
                offseth = 1
            if target_size[1] % 2 == 1:
                offsetw = 1
                
            img = img[centerh-halfh:centerh+halfh+offseth, centerw-halfw:centerw+halfw+offsetw, :]

        if crop_mode == 'random_crop':
            # set the offset of the left and right boundary
            rangew = (img.shape[1] - target_size[1]) // 2
            rangeh = (img.shape[0] - target_size[0]) // 2
            offsetw = 0 if rangew == 0 else np.random.randint(rangew)
            offseth = 0 if rangeh == 0 else np.random.randint(rangeh)

            img = img[offseth:offseth+target_size[1], offsetw:offsetw+target_size[0], :]
            
        if not crop_mode:
            return img
        
        return img
        
    def data_generator(self):
        '''
        The function is used to generate the data batch by batch.
        '''
        if self.shuffle:
            df_data = self.df.sample(n=len(self.df), random_state=123)
        else:
            df_data = self.df
        
        nb_batch = int(len(self.df)/self.batch_size)
        
        while 1:
            for i in range(nb_batch):
                if i == 0:
                    self.paths_labels = self._path_converter(df_data[:(i+1)*self.batch_size])

                self.paths_labels = self._path_converter(df_data[i*self.batch_size:(i+1)*self.batch_size])

                self.images = self.paths_labels[0]
                self.labels = self.paths_labels[1]

                x = []
                y = []
                for j in range(self.batch_size): 
                    img_paths = self.images[j]
                    x_img = self._load_img_pair(paths=img_paths,
                                                    target_size=self.target_size, 
                                                    short_edge_min=self.short_edge_min, 
                                                    short_edge_max=self.short_edge_max, 
                                                    crop_mode=self.crop_mode)

                    y_label = self.labels[j]

                    x.append(x_img)
                    y.append(y_label)

                x = np.asarray(x).reshape(self.batch_size, self.target_size[0], self.target_size[1], 3)
                y = np.asarray(y).reshape(self.batch_size, len(self.classes))

                yield (x, y)





