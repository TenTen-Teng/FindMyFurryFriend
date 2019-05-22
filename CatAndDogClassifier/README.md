# Cat and Dog Classifier

This project is building a classifier to identify cat and dog. The dataset is using `Kaggle Cats and Dogs Dataset`. The model is `DenseNet121`. 

## Environment
* python: 3.6.6
* tensorflow: 1.4
* keras: 2.1.3

## Library
* json
* Pillow==5.4.1
* pandas
* numpy
* matplotlib
* itertools
* os
* tqdm
* sklearn
* argparse
* time

## Data
There are 25,000 colorful images totally, including cat and dog two classes and equally separate the dataset.

There are various of data size: 500*375, 300*281, 312*397, and so on. In that case, I need to unify the input image size. Also, since the data distribution is good, I don’t need to balance the dataset. Moreover, I think 25,000 images for binary classification is enough, so I didn’t use data augmentation in the project.

## How to run
### data
download data in the project folder and upzip data into a 'data' folder. Here's the [link](https://www.microsoft.com/en-us/download/details.aspx?id=54765)

### src
There are 4 steps to run the whole project. 
1. prepare_dataset.py
This script is used to convert image paths to dataframe format and save them to CSV file. After that, separate whole dataset to training, validation, and test dataset and save them. 
- check help 
In the terminal, under the file dirctory. 
```
python prepare_dataset.py -h
```
- run the code
```
python prepare_dataset.py --output_path=../models --ratio_train=0.8 --ratio_val=0.1 --ratio_test=0.1 --input_path=../data/kagglecatsanddogs_3367a/PetImages/ --col_names=image_path,ground_truth --save=True --shuffle=True 
```

* After this step, you will get 4 csv files in your folder, ```image_path.csv```, ```image_path_checked.csv```, ```train_data.csv```, ```val_data.csv```, ```test_data.csv``` 
These files are ready to use in the data generator function.
* Note: 
There are two images with zero byte in the original dataset. One is dog 11702.jpg and the other one is cat 666.jpg. I deleted them since they are the bad inputs. 
Delet them before you run the ```prepare_dataset.py```

2. model_training.ipynb
This scrip is used to train the model. 

* After running the code, we will get 3 trained models: ```densenet121_final.h5```, ```densenet121_val_acc_best.h5``` and ```densenet121_val_loss_best.h5```.
Also, a evaluation txt file would save into folder if ```eva=True``` where I saved the training and validation evaluation accuracy and loss in it. 

3. inference
- 3.1 using ```inference_keras.ipynb``` to evaluate test dataset
run inference_keras.ipynb in jupyter notebook. 

- 3.2 using ```inference_tf.py``` to evaluate tese dataset

4. deploy
4.1 Keras model to tensorflow model
- convert keras model to tensorflow model using ```keras2tf.ipynb```

* after running ```keras2tf.ipynb```,  you will get a `.pb` file 'densenet121_tf.pb' and a graph file 'graph_def.pb.ascii' which is available for visually check model architecture.

4.2 TensorFlow model to CoreML model
- convert tensorflow model to coreml model using ```tf2coreml.ipynb```

* after runnning ```tf2coreml.ipynb```, you will get a `.mlmodel` in the `models` folder. This file will be used in the IOS app. 












