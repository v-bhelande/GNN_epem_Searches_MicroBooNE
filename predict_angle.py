###################################################################################################################################

"""IMPORT MODULES"""

import os
import sys
import importlib

import time
from datetime import date
from tqdm import tqdm

import math
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_scatter import scatter_max
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, global_mean_pool
from torch_geometric.nn.conv import PointConv
torch.set_default_dtype(torch.float32)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from torch_cluster import knn, knn_graph
from mpl_toolkits import mplot3d

import pandas as pd
import uproot as up
import h5py

# Trick to import functions from parent dir in fried_green_tomatoes
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gnn_models_beta import *

###################################################################################################################################

"""USER INPUTS"""

# 1. Type of data to load, specify batch size: 8, 16, 32, 64, and coordinate type: sp, pc
BATCH_SIZE = 8
COORD_TYPE = 'pc'
TRAIN_LOADER_PATH = f'/home/vbhelande/work/train_angle_loader_{COORD_TYPE}_batch{BATCH_SIZE}_no_feats.pt'
TEST_LOADER_PATH = f'/home/vbhelande/work/test_angle_loader_{COORD_TYPE}_batch{BATCH_SIZE}_no_feats.pt'

# 2. Choose model, optimizer, lr, and loss function
MODEL = PointNetMSG()
OPTIMIZER = 'Adam'  # TODO: Dynamically implement this, need to change it manually in next cell for now...
LR = 1e-5
LOSS_FN = torch.nn.MSELoss()

# 3. Device to use. GPUs 0-3 are available, choose -1 for CPdU
GPU_CORE = 0
device = torch.device('cpu') if GPU_CORE == -1 else torch.device(f'cuda:{GPU_CORE}')

# 4. Specify number of epochs model should run for
NUM_EPOCHS = 200

# 5. Specify if results should be saved to hdf5 file
STORE_RESULTS = True

# 6. Specify if model should be saved
SAVE_MODEL = True

# 7. Specify if loading a preexisiting model
LOAD_MODEL = False

# 8. Path to model being loaded, set to none if not loading a model
MODEL_PATH = '/home/vbhelande/work/2024-11-22_batch8_pc_1e-05_PointNetMSG2_100epochs.pt'

###################################################################################################################################

"""GENERATE ML MODEL AND SETTINGS"""

train_loader = torch.load(TRAIN_LOADER_PATH)
test_loader = torch.load(TEST_LOADER_PATH)

if LOAD_MODEL: model = torch.load(MODEL_PATH)
else: model = MODEL
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR) # lr=1e-4 seems optimal (maybe?) for MSG models

# Calculate total number of parameters ML model can train
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters for this PointNet++ w Tnet on pos : {total_params}')

today = date.today()
trained_epochs = 0
if LOAD_MODEL:
    print('Loaded exisitng model:', MODEL_PATH)
    splits = MODEL_PATH.split('_')
    trained_epochs = int(splits[-1].split('epochs')[0])
total_epochs = trained_epochs + NUM_EPOCHS

# Specify file path to store results in hdf5 file else leave unchanged
if STORE_RESULTS:
    if TEST_LOADER_PATH[-7:] == 'shifted':
        RESULTS_FILE_PATH = f'/home/vbhelande/work/gnn_mark/output/{today}_angle_pred_batch{BATCH_SIZE}_{COORD_TYPE}_shifted_{LR}_{str(model).split("(")[0]}_{total_epochs}epochs.hdf5'
    else:
        RESULTS_FILE_PATH = f'/home/vbhelande/work/gnn_mark/output/{today}_angle_pred_batch{BATCH_SIZE}_{COORD_TYPE}_{LR}_{str(model).split("(")[0]}_{total_epochs}epochs.hdf5'
    print('Storing results in:', RESULTS_FILE_PATH)

if SAVE_MODEL:
        MODELS_FILE_PATH = f'/home/vbhelande/work/gnn_mark/models/{today}_batch{BATCH_SIZE}_{COORD_TYPE}_{LR}_{str(model).split("(")[0]}_{total_epochs}epochs.pt'
        print('Saving model to:', MODELS_FILE_PATH)

###################################################################################################################################

"""TRAIN + TEST DEFINITION"""

# Defines 1 iteration of train and test
def train(model, optimizer, loader, scheduler=None):
    """
    Args:
        model: ML model being trained on data
        optimizer: Optimizer used to  updated weights, biases and other features of model
        loader: Dataset model is being trained on
        scheduler: Adapts the lr (learning rate) of the model, default is None

    Returns:
        avg_loss: Average training loss of model from 1 training epoch as defined by loss_fn
    """

    model.train()

    total_loss = 0
    for data in loader:

        data = data.to(device)
        true_angles = data.y

        optimizer.zero_grad()
        outputs = model(data)
        loss = LOSS_FN(true_angles, outputs.flatten())
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()

        total_loss += loss.item() * data.num_graphs

    # Return avg loss
    avg_loss = math.sqrt(total_loss/len(loader.dataset))    # If you're confused about this like I was, refer to: https://www.storyofmathematics.com/average-of-averages/
    return avg_loss

# Allows for faster computation by not tracking gradients since machine isn't learning now
@torch.no_grad()
def test(model, loader, last):
    """
    Args:
        model: ML model being trained on data
        loader: Test dataset for model
        last: If we are on last training epoch

    Angle refers to the opening angle between e+ and e- from the e+e- events (Unit; rad)
    Positon here refers to the vertex of the event in Euclidean coordinates [x, y, z] (Unit: cm)

    predictions (Predicted values) is of the form (from each batch of data):
    [[angle1],
    [angle2],
       ...
    [angle128]]

    Returns:
        true_values: True values from synthetic/experimental data
        pred_values: Predicted values by model
        angle_error: Total in error angles normalized by length of angles data
    """

    model.eval()

    # Lists to store true and pred results from ALL batches
    true_values_df = [['angle']]
    pred_values_df = [['angle']]

    total_loss = 0
    for data in loader:

        data = data.to(device)
        true_angles = data.y

        pred_values = model(data)
        pred_angles = torch.select(pred_values, 1, 0)
        loss = torch.square(true_angles - pred_angles).sum()
        total_loss += loss

        if last:
            true_values_df.extend(true_angles.cpu().numpy())
            pred_values_df.extend(pred_angles.cpu().numpy())

    avg_loss = math.sqrt(total_loss/len(loader.dataset))

    # Return true values + predicted values + errors at final epoch
    if last:
        true_values_df = pd.DataFrame(data=true_values_df[1:], columns=true_values_df[0])
        pred_values_df = pd.DataFrame(data=pred_values_df[1:], columns=pred_values_df[0])
        return true_values_df, pred_values_df, avg_loss
    else:
        return avg_loss
    
###################################################################################################################################

"""FULL CYCLE"""

train_error_list = []
val_error_list = []

# Timer to check how long avg epoch takes
time_start = time.time()

for epoch in tqdm(range(1, NUM_EPOCHS+1)):

    train_loss = train(model, optimizer, train_loader)
    last_epoch = True if epoch == NUM_EPOCHS else False

    if last_epoch: true_values, pred_values, val_error = test(model, test_loader, last_epoch)
    else: val_error = test(model, test_loader, last_epoch)
    val_error_list.append(val_error)
    train_error_list.append(train_loss)

    # Print training error, angle error, and vertex position error after every epoch
    print(f'Epoch: {epoch:02d} | Training Loss: {train_loss:.4f} | Validation Loss: {val_error:.4f}')

time_end = time.time()
avg_time = (time_end-time_start)/NUM_EPOCHS
print(f'Avg time for 1 epoch is: {avg_time:.2f} s/it')

loss_df = pd.DataFrame([val_error_list, train_error_list]).transpose()
loss_df.columns = ['Angle Loss', 'Training Loss']

###################################################################################################################################

"""STORE RESULTS"""

# Save all dfs to hdf5 file for future access
if STORE_RESULTS:
    true_values.to_hdf(RESULTS_FILE_PATH, key='true_values')
    pred_values.to_hdf(RESULTS_FILE_PATH, key='pred_values')
    loss_df.to_hdf(RESULTS_FILE_PATH, key='loss_hist')

# Save model
if SAVE_MODEL:
    torch.save(model, MODELS_FILE_PATH)

###################################################################################################################################
