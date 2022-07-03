import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2

from modules.model import UNet, InferenceModel
from modules.dataset_utils import load_tiff_images, visualize_labels
from config import CFG


def visualize_results(subset):
    model = UNet(CFG.INPUT_SHAPE, CFG.LEVELS, CFG.INIT_CHANNELS)
    frame_shape = (768, 1024)
    model = InferenceModel(model, CFG.UNET_WEIGHTS, CFG.INPUT_SHAPE, CFG.OUTPUT_SHAPE, frame_shape, 8)

    x = load_tiff_images(CFG.DATASET_PATH[subset]['x'])
    y = load_tiff_images(CFG.DATASET_PATH[subset]['y'])
    viz_frames = visualize_labels(x, y)
    for img, viz_img in zip(x, viz_frames):
        predicted_labels = model(img, norm_to_uint=True)
        viz_pred = visualize_labels(img, predicted_labels, single_frame=True)
        cv2.imshow('input frame', img)
        cv2.imshow('true labeles', viz_img)
        cv2.imshow('pred labeles', viz_pred)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            return


if __name__ == '__main__':
    visualize_results('test')
