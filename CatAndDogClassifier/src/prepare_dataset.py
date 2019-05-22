
# coding: utf-8

# # Prepare dataset
# 
# * **Project name:** cat and dog classifier
# * **Author:** Teng Li
# * ** Date:** 05.10.2019
# 
# This script is used to prepare dataset, including convert image paths to csv file and split dataset to training, validation and test dataset.

# In[ ]:

from helper import generate_image_path, prepare_data, check_image_dim
import argparse

# In[ ]:

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument('--output_path', dest='output_path', required=True, help='(REQUIRED) output csv files directory')
    parser.add_argument('--ratio_train', dest='ratio_train', required=True, help='(REQUIRED) the ratio of train dataset to total dataset(default: 0.8)')
    parser.add_argument('--ratio_val', dest='ratio_val', required=True, help='(REQUIRED) the ratio of validation dataset to total dataset(default: 0.1)')
    parser.add_argument('--ratio_test', dest='ratio_test', required=True, help='(REQUIRED) the ratio of test dataset to total dataset(default: 0.1)')
    parser.add_argument('--input_path', dest='input_path', required=True, help='(REQUIRED) data source directory')

    # optional
    parser.add_argument('--col_names', dest='col_names', default='image_path,ground_truth', type=str, required=False, help='a list of column names responding to dataframe columns(default: image_path, ground_truth)')
    parser.add_argument('--save', dest='save', default=True, type=bool, required=False, help='boolean value that determine save file or not (default: True)')
    parser.add_argument('--shuffle', dest='shuffle', default=True, type=bool, required=False, help='boolean value that determine if shuffle or not(default: True)')


    args = parser.parse_args()
    col_names = [item for item in args.col_names.split(',')]
    df = generate_image_path(input_path=args.input_path, 
                       output_path=args.output_path,
                       col_names=col_names,
                       save=args.save,
                       shuffle=args.shuffle)

    df_checked = check_image_dim(df=df, col_image=col_names[0], output_path=args.output_path, save=args.save)

    prepare_data(df=df_checked, 
                       output_path=args.output_path,
                       ratio_train=args.ratio_train,
                       ratio_val=args.ratio_val,
                       ratio_test=args.ratio_test,
                       save=args.save)

