import torch
import math
import numpy as np
import pandas as pd
from torch_geometric.nn import fps, radius
from torch_cluster import knn_graph
import matplotlib.pyplot as plt
from matplotlib import colors 
from matplotlib.ticker import PercentFormatter 
from typing import Union, Tuple, List

# Functions to write for 2, 4 (do later) batches
# 1. Test loss curves
# 2. Train loss curves
# 3. Model comparison curves
# 4. Angle error histograms
# 5. 2D histogram of true v/s reco opening angles
# 6. Average error as function of true angles

# Make a plot showing test loss curves
def plt_test_loss(files: List[str]):
    """
    Arg:
        files: List of files to generate test loss curves from

    Assumes same model, coordinate type, optimizer, and lr are used
    Assumes model in each file was ran for same number of epochs 
    
    Returns:
        Plot showing test loss curve for each file
    """

    plt.figure(figsize=(8,5))
    for i in range(len(files)):
        test_loss = pd.read_hdf(files[i], key='loss_hist')
        splits = files[i].split('_')
        batch_size = splits[-4].split('batch')[-1]
        plt.plot(test_loss['Angle Loss'], label=f'Batch {batch_size}')
        if i == len(files)-1:
            epochs = splits[-1].split('epochs')[0]
            model = splits[-2]
            print(model)
            coord_type = splits[-3]
            print(coord_type)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss ($^{\circ})$')
    plt.title(f'Test Loss Curves with Varied Batch Sizes \n Used: {coord_type}, {model}, {epochs} epochs, Adam, lr = 1e-4')
    plt.legend()

# Make a plot showing training loss curves
def plt_train_loss(files: List[str]):
    """
    Arg:
        files: List of files to generate test loss curves from

    Assumes same model, coordinate type, optimizer, and lr are used
    Assumes model in each file was ran for same number of epochs 
    
    Returns:
        Plot showing test loss curve for each file
    """

    plt.figure(figsize=(8,5))
    for i in range(len(files)):
        test_loss = pd.read_hdf(files[i], key='loss_hist')
        splits = files[i].split('_')
        batch_size = splits[-4].split('batch')[-1]
        plt.plot(test_loss['Training Loss'], label=f'Batch {batch_size}')
        if i == len(files)-1:
            epochs = splits[-1].split('epochs')[0]
            model = splits[-2]
            print(model)
            coord_type = splits[-3]
            print(coord_type)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss ($^{\circ})$')
    plt.title(f'Training Loss Curves with Varied Batch Sizes \n Used: {coord_type}, {model}, {epochs} epochs, Adam, lr = 1e-4')
    plt.legend()

# Make a 2D histogram depicting predicted v/s true opening angle distribution
def pred_vs_true_angle_2d_hist(true_vals, pred_vals, model=None, bins=None):
    """
    Args:
        true_vals: pandas df containing true angle values
        pred_vals: pandas df containing predicted angle values
        model (Optional): Name of model used, default is None
        bins: Bin size of 2D histogram, default is None

    Returns:
        2D histogram depicting predicted v/s true opening angle distribution
    """

    # Extract true and predicted angles as pandas df
    true_angles = true_vals['angle']
    pred_angles = pred_vals['angle']

    # Find min and max values of true values and pred values to make bins
    true_min = min(0, math.floor(min(true_angles)))
    true_max = max(45, math.ceil(max(true_angles)))
    pred_min = min(0, math.floor(min(pred_angles)))
    pred_max = 45#max(45, math.ceil(max(pred_angles)))
    
    # Create bins for plotting 2D histogram if none specified
    bins = [true_max-true_min, pred_max - pred_min] if bins == None else bins

    # Create 2D histogram
    model_name = '' if model == None else f'{str(model)}'
    plt.hist2d(true_angles, pred_angles, bins)
    plt.ylim(bottom=0, top=45)
    plt.xlabel('True Opening Angle ($^{\circ}$)')
    plt.ylabel('Predicted Opening Angle ($^{\circ}$)')
    #plt.title('Predicted vs. True Opening Angle Distribution (Degrees)' + model_name)
    plt.title(model_name)
    colorbar = plt.colorbar()
    colorbar.set_label('Frequency')
    plt.gca().set_aspect('equal')

def angle_error_hist(true_vals, pred_vals, model=None, bins=None):
    """
    Args:
        true_vals: pandas df containing true angle values
        pred_vals: pandas df containing predicted angle values
        model (Optional): Name of model used, default is None
        bins: Bin size of 2D histogram, default is None

    Returns:
        1D histogram of errors in angle prediction
    """

    true_angles = true_vals['angle']
    pred_angles = pred_vals['angle']
    errs = true_angles - pred_angles
    model_name = None if model == None else f'for {model}'

    true_min = math.floor(min(true_angles))# min(0, math.floor(min(true_angles)))
    true_max = math.ceil(max(true_angles))#max(45, math.ceil(max(true_angles)))
    pred_min = math.floor(min(pred_angles))#min(0, math.floor(min(pred_angles)))
    pred_max =  math.ceil(max(pred_angles))#max(45, math.ceil(max(pred_angles)))
    bins = [np.max([true_max, pred_max]) - np.min([true_min, pred_min])] if bins == None else bins

    plt.hist(errs)
    plt.xlabel('Count')
    plt.ylabel('Error')
    plt.title('Histogram of Angle Errors' + model_name)
