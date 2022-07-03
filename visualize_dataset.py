import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import cv2

from config import CFG

from modules.dataset_utils import load_tiff_images, visualize_labels
from modules.data_loader import init_2d_dataset


def from_files(subset):
    x = load_tiff_images(CFG.DATASET_PATH[subset]['x'])
    y = load_tiff_images(CFG.DATASET_PATH[subset]['y'])
    viz_frames = visualize_labels(x, y)
    while True:
        for img, viz_img in zip(x, viz_frames):
            cv2.imshow('training examples', img)
            cv2.imshow('labeled examples', viz_img)
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                return


def from_dataset(subset):
    if subset == 'train' or subset == 'val':
        train_dataset, val_dataset = init_2d_dataset(CFG.DATASET_PATH['train']['x'], CFG.DATASET_PATH['train']['y'], 256, 164, 1, validation_split_ratio=CFG.VALIDATION_SPLIT)
        dataset = train_dataset if subset == 'train' else val_dataset
    else:
        dataset = init_2d_dataset(CFG.DATASET_PATH['test']['x'], CFG.DATASET_PATH['test']['y'], 256, 164, 1)

    if subset == 'val':
        for x_patches, y_patches in dataset:
            x_patches = (x_patches[..., 0].numpy() * 255).astype(np.uint8)
            y_patches = (y_patches[..., 0].numpy() * 255).astype(np.uint8)
            for x, y in zip(x_patches, y_patches):
                viz_img = visualize_labels(x, y, single_frame=True)
                cv2.imshow('images', x)
                cv2.imshow('labels', y)
                cv2.imshow('labeled', viz_img)
                if cv2.waitKey(0) == 27:
                    cv2.destroyAllWindows()
                    return
    else:
        for x, y in dataset:
            x = (x[0, ..., 0].numpy() * 255).astype(np.uint8)
            y = (y[0, ..., 0].numpy() * 255).astype(np.uint8)
            viz_img = visualize_labels(x, y, single_frame=True)
            cv2.imshow('images', x)
            cv2.imshow('labels', y)
            cv2.imshow('labeled', viz_img)
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                return


if __name__ == '__main__':
    # from_files('train')
    from_dataset('val')
