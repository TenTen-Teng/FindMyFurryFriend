
# coding: utf-8

# # training function
# 
# * **Project name:** cat and dog classifier
# * **Author:** Teng Li
# * ** Date:** 05.10.2019

# In[1]:

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.optimizers import SGD, Adam
import os
from time import time
from create_mdoel import create_model
from helper import plot_results


# In[2]:

def train(train_data, 
          val_data, 
          result_path,
          model_name,
          batch_size,
          optimizer='Adam',
          nb_epoch=50,
          lr=0.0001,
          lr_decay=0.0005,
          best_model_save='both',
          loss_function='categorical_crossentropy',
          reduce_lr=True,
          early_stopping=True,
          tensorboard=True,
          reduce_lr_monitor='val_acc',
          reduce_lr_factor=0.5,
          redue_lr_patience=5,
          early_stopping_monitor='val_acc',
          early_stopping_patience=10,
          multiprocessing=False,
          evaluate=True,
         ):
    """
    A training function
    
    :param train_data - DataGenerator: a data generator of training dataset
    :param val_data - DataGenerator: a data generator of validation dataset
    :param result_path - str: a path of result save
    :param model_name - str: the name of model
    :param batch_size - int: batch size
    :param optimizer - str: the optimizer function, options include ['Adam', 'SGD']
    :param nb_epoch - int: epoch number
    :param lr - float: learning rate
    :param lr_decay - float: learning rate decay
    :param best_model_save - string: save which best model in the callback: ["val_acc", "val_loss", "both", "None"]
    :param loss_function - string: loss function
    :param reduce_lr - boolean: using reduce learning rate or not
    :param early_stopping - boolean: uing early stopping or not
    :param tensorboard - boolean: using tensorboard or not
    :param reduce_lr_monitor - string: reduce learning rate monitor: ["val_acc", "val_loss"]
    :param reduce_lr_factor - float: the reduce factor of learning rate
    :param redue_lr_patience - int: the number of epoch that reduce learning rate
    :param early_stopping_monitor - string: early stopping monitor: ["val_acc", "val_loss"]
    :param early_stopping_patience - int: the number of epoch that early stopping    
    :param multiprocessing - bool: boolean value that determine use CPU multiprocessing for not, default values is False
    :param evaluate - bool: boolean value that determine evalue training dataset and test dataset or not, default values is True
    
    :retrun: trained model
    """
        
    '''set optimizer'''
    if optimizer == 'SGD':
        optimizer = SGD(lr=lr, decay=0.0005, momentum=0.9, nesterov=True)
    elif optimizer == 'Adam':
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=lr_decay)

    '''set model''' 
    nb_classes = val_data.nb_classes
    model = create_model(nb_classes=nb_classes, summary=False, save=True)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    
    '''make path for saving trained model'''
    model_path = result_path + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    '''set checkpoints'''
    callbacks_list = []
    
    # set the best model to save: best validation accuracy or best validation loss
    if best_model_save not in ['val_acc', 'val_loss', 'both']:
        best_model_save = 'None'
        
    if best_model_save == 'val_acc':
        checkpoint_val_acc_best = ModelCheckpoint(model_path+model_name+'_val_acc_best.h5', 
                                                  monitor='val_acc', 
                                                  verbose=1, 
                                                  save_best_only=True, 
                                                  mode='max')
        callbacks_list.append(checkpoint_val_acc_best)
    if best_model_save == 'val_loss':
        checkpoint_val_loss_best = ModelCheckpoint(model_path+model_name+'_val_loss_best.h5', 
                                                  monitor='val_loss', 
                                                  verbose=1, 
                                                  save_best_only=True, 
                                                  mode='min')
        callbacks_list.append(checkpoint_val_loss_best)
    if best_model_save == 'both':
        checkpoint_val_acc_best = ModelCheckpoint(model_path+model_name+'_val_acc_best.h5', 
                                                  monitor='val_acc', 
                                                  verbose=1, 
                                                  save_best_only=True, 
                                                  mode='max')
        checkpoint_val_loss_best = ModelCheckpoint(model_path+model_name+'_val_loss_best.h5', 
                                                  monitor='val_loss', 
                                                  verbose=1, 
                                                  save_best_only=True, 
                                                  mode='min')
        callbacks_list.append(checkpoint_val_acc_best)
        callbacks_list.append(checkpoint_val_loss_best)

    
    '''set reduce learning rate'''
    if reduce_lr:
        # set the reduce learning rate monitor: validation accuracy or validation loss
        if reduce_lr_monitor == 'val_acc':
            reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=reduce_lr_factor, patience=redue_lr_patience, mode='auto')
            callbacks_list.append(reduce_lr)
        if reduce_lr_monitor == 'val_loss':
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=redue_lr_patience, mode='auto')
            callbacks_list.append(reduce_lr)
        
    '''set early stopping'''
    if early_stopping:
        # set the early stopping monitor: validation accuracy or validation loss
        if early_stopping_monitor == 'val_acc':
            early_stopping = EarlyStopping(monitor='val_acc', patience=early_stopping_patience, verbose=1)
            callbacks_list.append(early_stopping)
        if early_stopping_monitor == 'val_loss':
            early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1)
            callbacks_list.append(early_stopping)
    
    '''set tensorboard'''
    if tensorboard:
        callbacks_list.append(TensorBoard(log_dir='./logs/{}'.format(time()), write_graph=True, write_images=True))


    '''model training'''
    if multiprocessing:
        history = model.fit_generator(
                    generator=train_data.data_generator(),
                    steps_per_epoch=train_data.num/batch_size,
                    validation_data=val_data.data_generator(),
                    validation_steps=val_data.num/batch_size,
                    epochs=nb_epoch,
                    workers=16,         # num of multiprocessing CPU codes
                    max_queue_size=32,      # max size of queue to hold the preprocessed images
                    pickle_safe=True,   # True for CPU multiprocessing, False for CPU multithreading
                    callbacks=callbacks_list)
    else:
        history = model.fit_generator(
                    generator=train_data.data_generator(),
                    steps_per_epoch=train_data.num/batch_size,
                    validation_data=val_data.data_generator(),
                    validation_steps=val_data.num/batch_size,
                    epochs=nb_epoch,
                    callbacks=callbacks_list)

    
    '''save final model'''
    model.save(model_path+model_name+'_final.h5')
    
    '''plot model training results'''
    plot_results(model_path, history)
    
    '''evaluate results'''
    if evaluate:
        print('evaluating training dataset...')
        train_score = model.evaluate_generator(generator=train_data.data_generator(), steps=train_data.num/batch_size)
        print('evaluating validation dataset...')
        val_score = model.evaluate_generator(generator=val_data.data_generator(), steps=val_data.num/batch_size)
        with open(model_name + '_evaluation.txt', 'w+') as f:
            f.write('validation loss: ' + str(val_score[0]) + '\n')
            f.write('validation accuracy: ' + str(val_score[1]) + '\n')
            f.write('train loss: ' + str(train_score[0]) + '\n')
            f.write('train accuracy: ' + str(train_score[1]) + '\n')
            
    return model
    


# In[ ]:



