
# coding: utf-8

# # Inference TensforFlow
# 
# * **Project name:** cat and dog classifier
# * **Author:** Teng Li
# * ** Date:** 05.10.2019
# 
# This notebook is used to predict the trained TensforFlow model 

# In[1]:

import argparse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


# In[20]:

def predict_image(image_path, model_path):
    """
    Make one image prediction using pretrained model
    
    :param image_path - str: the path where image stored
    :param model_path - str: the path where the trained model(.pb) stored
    
    output: the classification result label
    
    """
    short_edge_min = 256
    short_edge_max = 384
    center_crop_size = (224, 224)

    # open image
    img = Image.open(image_path)
    
    # downsample image
    width, height = img.size
    tmp = min(width, height)    
    short_edge_resize = np.random.randint(short_edge_min, short_edge_max)
    width = int(width * short_edge_resize / tmp)
    height = int(height * short_edge_resize / tmp)
    img = img.resize((width, height))
    img = np.array(img)
    
    # center crop image
    centerw, centerh = img.shape[1]//2, img.shape[0]//2
    halfw, halfh = center_crop_size[1]//2, center_crop_size[0]//2

    offsetw, offseth = 0, 0
    if center_crop_size[0] % 2 == 1:
        offseth = 1
    if center_crop_size[1] % 2 == 1:
        offsetw = 1

    img = img[centerh-halfh:centerh+halfh+offseth, centerw-halfw:centerw+halfw+offsetw, :]

    # expand image dimension to 4D
    img = np.expand_dims(img, axis=0)
    
    # load tensorflow graph
    with tf.Session() as sess:
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def)
    
            softmax_tensor = sess.graph.get_tensor_by_name('import/dense_1/Softmax:0')
            predictions = sess.run(softmax_tensor, {'import/input_1:0': img})
            
            
    # print predicted label
    predictions = np.squeeze(predictions)
    pred_labels = np.argmax(predictions)
    labels = ['cat', 'dog']

    for i in range(len(labels)):
        print(labels[i] + ': %f' %predictions[i])

    result = labels[int(pred_labels)]
    print('model predict result is %s' %result)
    


# In[ ]:

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument('--image_path', dest='image_path', required=True, help='(REQUIRED) test image path')
    parser.add_argument('--model_path', dest='model_path', required=True, help='(REQUIRED) tf model path')

    args = parser.parse_args()

    predict_image(image_path=args.image_path, 
                  model_path=args.model_path)

