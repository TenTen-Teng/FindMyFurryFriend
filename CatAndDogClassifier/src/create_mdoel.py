
# coding: utf-8

# # Create model: DenseNet121
# 
# * **Project name:** cat and dog classifier
# * **Author:** Teng Li
# * ** Date:** 05.10.2019
# 
# This script is used to create a DenseNet121 model for training

# In[ ]:

from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import Dense
import json


# In[ ]:

def create_model(nb_classes, summary=False, save=False):
    """
    The function is used to create a DenseNet121 model with softmax output layer
    
    :param nb_classes - int: the number of classes
    :param summary - bool: boolean value for determine show model summary or not, default value is False
    :param save - bool: boolean value for determine save model architure or not, default value is False
    
    :return model: a DenseNet121 model
    """
    
    print('creating model...')
    base_model = DenseNet121(weights='imagenet')
    base_outputs = base_model.get_layer('avg_pool').output
    
    # add softmax in the last layer
    new_outputs = Dense(nb_classes, activation='softmax')(base_outputs)
    
    model = Model(inputs=base_model.input, outputs=new_outputs)
    
    if summary:
        # show model summary
        model.summary()
    
    if save:
        # save model into JSON file
        print('save model to JSON file')
        with open('./model.json', 'w') as outfile:
            outfile.write(json.dumps(json.loads(model.to_json()), indent=3))
            
    print('load model!')
    return model

