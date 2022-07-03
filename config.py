import os
import sys


class Config:
    BASE_PATH = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = os.path.join(BASE_PATH, 'data')
    WEIGHTS_PATH = os.path.join(BASE_PATH, 'weights')

    DATASET_PATH = {
        'train': {
            'x': os.path.join(DATA_PATH, 'training.tif'),
            'y': os.path.join(DATA_PATH, 'training_groundtruth.tif')
        },
        'test': {
            'x': os.path.join(DATA_PATH, 'testing.tif'),
            'y': os.path.join(DATA_PATH, 'testing_groundtruth.tif')
        }
    }

    UNET_WEIGHTS = os.path.join(WEIGHTS_PATH, 'unet.tf')

    VALIDATION_SPLIT = 0.15
    INPUT_SHAPE = (256, 256, 1)
    OUTPUT_SHAPE = (164, 164, 1)
    LEVELS = 3
    INIT_CHANNELS = 64
    BATCH_SIZE = 2
    INIT_LEARNING_RATE = 1e-3
    EPOCHS = 50


CFG = Config()
