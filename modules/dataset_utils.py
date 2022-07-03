import numpy as np
import cv2

from PIL import Image


def load_tiff_images(path):
    images = list()
    tiff_album = Image.open(path)
    for i in range(tiff_album.n_frames):
        tiff_album.seek(i)
        images.append(np.array(tiff_album))
    return np.array(images)


def visualize_labels(x, y, single_frame=False):
    if single_frame:
        x = x[np.newaxis, ...]
        y = y[np.newaxis, ...]
    viz_frames = list()
    for img, seg_map in zip(x, y):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_GRAY2BGR)
        seg_map[:, :, :2] = 0

        size_diff = (img.shape[0] - seg_map.shape[0]) // 2
        img_cutout = img[size_diff:seg_map.shape[0]+size_diff, size_diff:seg_map.shape[1]+size_diff]

        cutout = cv2.addWeighted(img_cutout, 0.8, seg_map, 0.2, 1.0)
        frame = img.copy()
        frame[size_diff:seg_map.shape[0]+size_diff, size_diff:seg_map.shape[1]+size_diff] = cutout
        viz_frames.append(frame)
    if single_frame:
        viz_frames = viz_frames[0]
    return np.array(viz_frames)
