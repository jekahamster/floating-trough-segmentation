import os
import numpy as np

from datetime import datetime
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import backend as K


def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def dice_score(y_true, y_pred, smooth=1e-6):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_loss(y_true, y_pred, smooth=1e-6):
    return 1 - dice_score(y_true, y_pred, smooth)


def bce_dice_loss(y_true, y_pred):
    loss = keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred, smooth=1)
    return loss


def plot_training_history(history, metrics=None, save_dir=None):
    metrics = metrics or []

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    
    if history.history.get('val_loss'):
        plt.plot(history.history['val_loss'], label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if metrics:
        plt.subplot(1, 2, 2)
        for metric in metrics:
            plt.plot(history.history[metric], label=f'Training {metric}')

            if history.history.get(f'val_{metric}'):
                plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')

        plt.title('Training and Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()

    plt.tight_layout()
    
    if save_dir:
        strtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"training_history_{strtime}.png"
        plt.savefig(os.path.join(save_dir, fname))
        print(f"Figure saved to {os.path.join(save_dir, fname)}")

    plt.show()