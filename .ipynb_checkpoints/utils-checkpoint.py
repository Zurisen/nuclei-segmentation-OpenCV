import os
import cv2
import numpy as np
import glob
from PIL import Image, ImageEnhance
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from skimage import data, io, img_as_float, exposure
import matplotlib.pyplot as plt
from copy import deepcopy
import tifffile


#Dataset Loader
def plot_equalizations(img, clip_limit=0.03, contrast=0.7, brightness=0.7):
    '''
    Performs histogram normal equalization for the given input image
    Args:
        img (Numpy Array): 3 channel input image
    Doc: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html
    '''
    
    img[img>255]=255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    bright = ImageEnhance.Brightness(img)
    img = bright.enhance(brightness)
    cont = ImageEnhance.Contrast(img)
    img = cont.enhance(contrast)
    img = np.array(img)
    
    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=clip_limit)

    # Display results
    fig = plt.figure(figsize=(18, 18))
    axes = np.zeros((2, 2), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 2, 1)
    for i in range(1, 2):
        axes[0, i] = fig.add_subplot(2, 2, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    for i in range(0, 2):
        axes[1, i] = fig.add_subplot(2, 2, 3+i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title('Low contrast image')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 1])
    ax_img.set_title('Adaptive equalization')

    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()
    
    return img.astype(np.uint8), img_adapteq.astype(np.uint8)


#Dataset Loader
def process_channels(img, file_name, clip_limit=0.03, contrast=9, brightness=0.2):
    '''
    Performs histogram normal equalization for the given input image
    Args:
        img (Numpy Array): 3 channel input image
    Doc: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html
    '''
    img_channels = []
    
    img[img>255]=255
    
    for i in range(img.shape[2]):
        chan = img[:,:,i]
        chan = chan.astype(np.uint8)
        chan = Image.fromarray(chan)
        bright = ImageEnhance.Brightness(chan)
        chan = bright.enhance(brightness)
        cont = ImageEnhance.Contrast(chan)
        chan = cont.enhance(contrast)
        chan = np.array(chan)
        chan = chan.astype(np.uint8)

        if not os.path.exists(os.path.join("results", file_name)):
            os.mkdir(os.path.join("results", file_name))

        img_channels.append(chan)
        tifffile.imwrite(f'results/{file_name}/Channel_{i+1}.tiff', chan)
    
    
    
    return img_channels


def plot_img_and_hist(image, axes, bins=256):
    """
    Plot an image along with its histogram and cumulative histogram.

    """

    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

def extract_masks(image, sensitivity, dilation=True, plot=False):
    '''
    Get mask with the nuclei segmentations of the image.
    Args:
        image (Numpy Array): 3 channel input image.
        sensitivity (3 elements List): Sets the threshold for masking
        each RGB channel.
        dilation (Bool): Whether or not apply dilation on the segmentations,
        it slightly increases the area of each segmentation.
        plot (Bool): Plots the mask using matplotlib.
    '''
    
    ## options
    dilation=True
    ##

    size = image.shape[1]
    #imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, sensitivity, 255, 0)
    ## Optional dilation
    if dilation:
        kernal = np.ones((2, 2), np.uint8)
        dilation = cv2.dilate(thresh, kernal, iterations=1)
        final_mask = dilation
    else:
        final_mask = thresh
    ## Find contours

    contours, hierarchy = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    objects = str(len(contours))
    mask = np.zeros(final_mask.shape, dtype=np.int16)
    all_contours = cv2.drawContours(deepcopy(mask) , contours, -1, 255,-1)

    if plot:
        plt.subplot(1,2,1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Input image')
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(all_contours, cmap="gray",alpha=1)
        plt.title(f"Segmented mask")
        plt.axis('off')
        plt.show()
    return mask, contours, hierarchy