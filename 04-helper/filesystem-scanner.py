# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 22:30:00 2021

@author:       Genocs
@description:  Python program to explain os.scandir() method
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# importing os module
import os

import pathlib

# Import pandas to read csv files
import pandas as pd


def build_csv_file(path):
    print("Files and Directories in '% s':" % path)

    obj = os.walk(path)

    for root, directories, filenames in obj:
        files = []

        for filename in filenames:
            path = pathlib.PurePath(root)
            files.append({'filename': filename, 'class_name': path.name})

        save_result(files)

    obj.close()


def save_result(file_list, filename='.\\all_data.csv'):

    df_start = pd.DataFrame(columns=['path', 'label'])

    for file_item in file_list:
        df_start = df_start.append(
            {
                'path': 'gs://My_Bucket/' + file_item['class_name'] + '/' + file_item['filename'],
                'label':  file_item['class_name'],
            },
            ignore_index=True)

    # random state is a seed value
    train = df_start.sample(frac=0.8, random_state=200)
    test = df_start.drop(train.index)

    df = pd.DataFrame(columns=['set', 'path', 'label'])

    if (os.path.exists(filename)):
        df = pd.read_csv(filename)

    for index, row in train.iterrows():
        df = df.append(
            {
                'set': 'TRAIN',
                'path': row['path'],
                'label':  row["label"],
            },
            ignore_index=True)

    for index, row in test.iterrows():
        df = df.append(
            {
                'set': 'TEST',
                'path': row['path'],
                'label':  row["label"],
            },
            ignore_index=True)

    df.to_csv(filename, index=False)


def main():
    # create_dataset(base_dir='E:\\Data\\UTU\\directinvoice_img')
    build_csv_file(path='E:\\Data\\UTU\\directinvoice_ts')


if __name__ == '__main__':
    main()
