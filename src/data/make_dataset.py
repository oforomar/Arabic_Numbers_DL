# -*- coding: utf-8 -*-
import click
import logging
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv


#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    TEST_PATH  = '../../data/raw/test/'
    TRAIN_PATH = '../../data/raw/train/'

    data_dir = Path(TRAIN_PATH)
    test_dir = Path(TEST_PATH)

    dfs = []
    total = 0

    for i in range(1,10):
        image_path = pd.Series(list(data_dir.glob(f'{i}/*'))).astype(str)
        
        labels = pd.Series(np.full((len(image_path)), i))
        
        total += len(image_path)
        print(len(image_path), total)
        
        dfs.append(pd.concat([image_path, labels], axis=1))
    
    df = pd.concat(dfs, axis=0, ignore_index=True)

    df.columns = ['image_path', 'label']

    df[['height', 'width']] = df.apply(lambda row: read_shape(row), axis=1)

    df['aspect ratio'] = df['width'] / df['height']

    df.to_csv('../../data/processed/df_train.csv')

    dfs = []
    total = 0

    for i in range(1,10):
        image_path = pd.Series(list(test_dir.glob(f'{i}/*'))).astype(str)

        labels = pd.Series(np.full((len(image_path)), i))

        total += len(image_path)
        print(len(image_path), total)

        dfs.append(pd.concat([image_path, labels], axis=1)) 

    test_df = pd.concat(dfs, axis=0, ignore_index=True)
    test_df.columns = ['image_path', 'label']

    test_df.to_csv('../../data/processed/df_test.csv')


def read_shape(row):
    # read image
    img = cv2.imread(row[0], -1)
    
    # get shape
    sh = img.shape
    
    return pd.Series(list(sh))



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
