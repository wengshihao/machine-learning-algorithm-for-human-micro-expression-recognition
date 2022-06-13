#!/usr/bin/env python
# coding: utf-8
import sys
import time

import numpy as np
import pandas as pd
import math
import cv2

# uniform LBP-TOP
# length height width:
from PIL import ImageDraw, Image
from matplotlib import pyplot as plt


def get_LBP_TOP(img_seq, uniform_dict, x_radius=1, y_radius=1, t_radius=4, xy_neighbor=8, xt_neighbor=8, yt_neighbor=8):
    length, height, width = img_seq.shape[:3]

    bins = 59
    pi = 3.1415926
    hist = [[0 for j in range(bins)] for i in range(3)]
    hist = np.array(hist).astype('float64')
    x_border = x_radius
    y_border = y_radius
    t_border = t_radius

    for ti in range(t_border, length - t_border):
        for yi in range(y_border, height - y_border):
            for xi in range(x_border, width - x_border):
                center = img_seq[ti][yi][xi]

                # in XY plane
                basic_LBP = 0
                fea_bin = 0
                for p in range(0, xy_neighbor):
                    x = math.floor(xi + x_radius * math.cos((2 * pi * p) / xy_neighbor) + 0.5)
                    y = math.floor(yi - y_radius * math.sin((2 * pi * p) / xy_neighbor) + 0.5)
                    # print(x, y)
                    current = img_seq[ti][y][x]
                    if current >= center:
                        basic_LBP = basic_LBP + 2 ** fea_bin
                    fea_bin = fea_bin + 1
                hist[0, uniform_dict[basic_LBP]] += 1

                # in XT plane
                basic_LBP = 0
                fea_bin = 0
                for p in range(0, xt_neighbor):
                    x = math.floor(xi + x_radius * math.cos((2 * pi * p) / xt_neighbor) + 0.5)
                    t = math.floor(ti + t_radius * math.sin((2 * pi * p) / xt_neighbor) + 0.5)
                    current = img_seq[t][yi][x]
                    if current >= center:
                        basic_LBP = basic_LBP + 2 ** fea_bin
                    fea_bin = fea_bin + 1
                hist[1, uniform_dict[basic_LBP]] += 1

                # in YT plane
                basic_LBP = 0
                fea_bin = 0
                for p in range(0, yt_neighbor):
                    y = math.floor(yi - y_radius * math.sin((2 * pi * p) / yt_neighbor) + 0.5)
                    t = math.floor(ti + t_radius * math.cos((2 * pi * p) / yt_neighbor) + 0.5)
                    current = img_seq[t, y, xi]
                    if current >= center:
                        basic_LBP = basic_LBP + 2 ** fea_bin
                    fea_bin = fea_bin + 1
                hist[2, uniform_dict[basic_LBP]] += 1

    # nomalize
    for i in range(3):
        hist[i] = hist[i] / sum(hist[i])

    return hist


def seq_divide(image_seq, t_times=4, y_times=4, x_times=4):
    length, height, width = image_seq.shape[:3]
    sub_length, sub_height, sub_width = (length // t_times, height // y_times, width // x_times)
    new_seq = image_seq[0:t_times * sub_length, 0:y_times * sub_height, 0:x_times * sub_width]

    sub_blocks = []
    for ti in range(t_times):
        for yi in range(y_times):
            for xi in range(x_times):
                t_up = min((ti + 1) * sub_length, length)
                y_up = min((yi + 1) * sub_height, height)
                x_up = min((xi + 1) * sub_width, width)
                tmp = np.array(new_seq[ti * sub_length:t_up, yi * sub_height:y_up, xi * sub_width:x_up])
                sub_blocks.append(tmp)
    sub_blocks = np.array(sub_blocks)
    print(sub_blocks.shape)
    return sub_blocks


# 只有LBP-TOP需要uniform_dict
# feature: LBP-TOP、3DHOG、HOOF
def get_ep_features(ep, uniform_dict=None, feature='LBP-TOP', t_times=4, y_times=4, x_times=4,
                    x_radius=1, y_radius=1, t_radius=4, xy_neighbor=8, xt_neighbor=8, yt_neighbor=8,
                    xy_bins=8, xt_bins=12, yt_bins=12,
                    bins=8):
    if len(ep.shape) == 4:
        gray_ep = []
        for image in ep:
            gray_ep.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        gray_ep = np.array(gray_ep)
    else:
        gray_ep = ep

    plt.show()
    sub_blocks = seq_divide(gray_ep, t_times=t_times, y_times=y_times, x_times=x_times)
    hist = []
    for cell in sub_blocks:
        print(cell.shape)
        if feature == 'LBP-TOP':
            hi = get_LBP_TOP(img_seq=cell, uniform_dict=uniform_dict, x_radius=x_radius, y_radius=y_radius,
                             t_radius=t_radius,
                             xy_neighbor=xy_neighbor, xt_neighbor=xt_neighbor, yt_neighbor=yt_neighbor)
            hist.append(hi)
    return np.array(hist).flatten()


def getVideoFrames(videoFilePath, startFrameNumber=-1, endFrameNumber=-1):
    frames = []
    vidcap = cv2.VideoCapture(videoFilePath)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    totalFrame = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    if startFrameNumber == -1:
        startFrameNumber = 0
    if endFrameNumber == -1:
        endFrameNumber = totalFrame - 1
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = color.rgb2yiq(image).astype(np.float32)

        if count < startFrameNumber:
            success, image = vidcap.read()
            count += 1
            continue
        elif count >= endFrameNumber:
            break
        else:
            frames.append(image)
        success, image = vidcap.read()
        count += 1
    frames = np.array(frames)

    return fps, frames


def get_uniform_dict(uniform_path):
    uniform = pd.read_csv(uniform_path, sep=' ')
    uniform.columns = ['default', 'uniform']
    uniform.head()
    uniform_dict = {}
    for i in range(len(uniform.default)):
        uniform_dict[uniform.default[i]] = uniform.uniform[i]
    return uniform_dict
def getVideoFrames(videoFilePath, startFrameNumber=-1, endFrameNumber=-1):
    """
    Loading video file between startFrameNumber and endFrameNumber.
    Each frame is converted to YIQ color space before saving to output matrix.

    Parameters
    ----------
    vidoFilePath: video file path
    startFrameNumber: start frame number to read. If it is -1, then start from 0
    endFrameNumber: end frame number to read. If it is -1, then set it to the total frame number

    Returns
    -------
    fps: frame rate of the video
    frames: frames matrix with the shape (frameCount, frameHeight, frameWidth, channelCount)
    """
    frames = []
    vidcap = cv2.VideoCapture(videoFilePath)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    totalFrame = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    if startFrameNumber == -1:
        startFrameNumber = 0
    if endFrameNumber == -1:
        endFrameNumber = totalFrame - 1
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = color.rgb2yiq(image).astype(np.float32)

        if count < startFrameNumber:
            success, image = vidcap.read()
            count += 1
            continue
        elif count >= endFrameNumber:
            break
        else:
            frames.append(image)
        success, image = vidcap.read()
        count += 1
    frames = np.array(frames)

    return fps, frames

def process(videopath,uniform_path,savepath):
    fps, frames = getVideoFrames(videopath, -1, -1)
    print("The video framerate is %.0f fps" % fps)
    print("Success read %d frames." % frames.shape[0], "We save them in a numpy array with shape:", frames.shape,
          ", which occupies %d bytes" % sys.getsizeof(frames))


    res=get_ep_features(frames, uniform_dict=get_uniform_dict(uniform_path), feature='LBP-TOP',
                                            t_times=2, x_times=2, y_times=2)

    np.save(savepath,res)

if __name__ == "__main__":
    #parameters
    videopath='EP18_01.avi'
    uniform_path = 'UniformLBP8.txt'
    savepath = 'feature.npy'


    process(videopath,uniform_path,savepath)

