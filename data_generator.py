import os
import sys

import numpy as np
import math
import h5py
import pandas as pd

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# Select GPU to use and set default torch datatype. 4 GPUs available: cuda: 1-4
print(torch.cuda.is_available())
print(torch.cuda.device_count())
torch.set_default_dtype(torch.float32)

# Choose a GPU core to use
GPU_CORE_NUM = 0
device = torch.device(f"cuda:{GPU_CORE_NUM}")

# Specify batch size
# Good rule of thumb is powers of 2 (why tho?)
BATCH_SIZE = 2

# Specify proportion of data to be allocated to test size
TEST_SIZE = 0.25

# Specify data path and coord type
DATA_PATH = "/home/vbhelande/work/gnn_mark/Isotropic_EpEm_Oct2024_data/all_data.pkl"
COORD_TYPE = 'pc' #'pc'

# Specify output path
TRAIN_LOADER_PATH = f"/home/vbhelande/work/train_angle_loader_{COORD_TYPE}_batch{BATCH_SIZE}_no_feats.pt"
TEST_LOADER_PATH = f"/home/vbhelande/work/test_angle_loader_{COORD_TYPE}_batch{BATCH_SIZE}_no_feats.pt"

# TODO: Implement functionality later
INCLUDE_CHARGE = False

# TODO: Implement fearures + cuts functionality at later date
def generate_loader(data_file, pos_type, device, features: list=None, sseed=123, include_charge=False, cut_rad=None, d2t=None, min_n=None):
    """
    Args:
        data_file: Directory containing data files
        pos_type: Picks out sp (spacepoints) or pc (point cloud) coordinates
        device: Device on which loaders are being generated
        features: Train GNN using charge, energy, etc
        sseed: Seed for generating random numbers on device

        # REST OF ARGS ARE FOR A LATER DATE!
        cut_rad: Maximum distance of points to vertex
        d2t: Maximum distance between the Pandora and true vertices
        min_n: Minimum number of points per event

        Data Objects are formatted as: [x, y, pos] where:
        1. x = features (Charge, energy, nothing, etc): Shape = [num_nodes, num_features]
        2. y = True angle [degrees]: Shape = [1]
        3. pos = sp/pc coordinates: Shape = [num_nodes, 3]
        Documentation: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html
        
    Returns:
        train_loader: Loader containing all training data for ML model
        test_loader: Loader containing all test data for ML model
    """

    # Set seed
    torch.manual_seed(sseed)
    torch.cuda.manual_seed(sseed)

    # Open file
    if data_file.endswith('.pkl'): data = pd.read_pickle(data_file)
    elif data_file.endswith('.hdf5'): data = pd.read_hdf(data_file)
    else: return 'Data must be stored as pkl or hdf5 file, exiting...'

    # List to store data objects
    all_data_objs = []

    # Iterate through all events in pandas df
    for index in range(len(data)):

        # Grab sp/pc (pos)
        if pos_type == 'sp':
            num_nodes = data.iloc[index]['nSP']
            if num_nodes == 0: continue
            x_coords = data.iloc[index]['sp_x']
            y_coords = data.iloc[index]['sp_y']
            z_coords = data.iloc[index]['sp_z']
            assert(num_nodes == len(x_coords) == len(y_coords) == len(z_coords))
        elif pos_type == 'pc':
            num_nodes = data.iloc[index]['nPC']
            if num_nodes == 0: continue
            x_coords = data.iloc[index]['pc_x']
            y_coords = data.iloc[index]['pc_y']
            z_coords = data.iloc[index]['pc_z']
            assert(num_nodes == len(x_coords) == len(y_coords) == len(z_coords))
        else: return 'pos_type must be sp (spacepoint) or pc (point cloud), exiting...'
        
        pos = np.concat((np.asarray(x_coords), np.asarray(y_coords), np.asarray(z_coords)))
        pos = np.reshape(pos, (len(x_coords), 3), order='F')
        pos = torch.as_tensor(pos, dtype=torch.float32)

        # TODO: Make data cuts once further progress on project is made!

        # Calculate true angle [degrees] using initial momenta (y)
        # e = electron, p = positron
        e_momentum = [data.iloc[index]['p1x'], data.iloc[index]['p1y'], data.iloc[index]['p1z']]
        p_momentum = [data.iloc[index]['p2x'], data.iloc[index]['p2y'], data.iloc[index]['p2z']]
        angle = np.arccos(np.dot(e_momentum, p_momentum)/(np.linalg.norm(e_momentum)*np.linalg.norm(p_momentum)))
        angle = [np.rad2deg(angle)]
        angle = torch.as_tensor(angle, dtype=torch.float32)

        # Include features (x)
        # TODO: Implement this later
        if features is not None: return 'features not implemented yet, exiting...'
        else: feats = torch.ones((num_nodes, 1))

        # Create Data object
        data_obj = Data(x=feats, y=angle, pos=pos)
        all_data_objs.append(data_obj)

    # Split data into training and test
    train_data_list, test_data_list = train_test_split(all_data_objs, test_size=TEST_SIZE, random_state=42)

    # Create train_loader and test_loader from respective lists, setting batch size as well
    # train_loader should be shuffled and both should drop last batch, which likely has uneven number of events
    train_loader = DataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data_list, batch_size=BATCH_SIZE, drop_last=True)

    # Print number of events in each and first Data object of train_data_list as sanity check
    print("Training Events:", len(train_data_list))
    print("Testing Events ", len(test_data_list))
    print(train_data_list[0])

    return train_loader, test_loader

# Make a new set of loaders
#trial = generate_loader(DATA_PATH, COORD_TYPE, device)
train_loader, test_loader = generate_loader(DATA_PATH, COORD_TYPE, device)

# Save loaders
torch.save(train_loader, TRAIN_LOADER_PATH)
torch.save(test_loader, TEST_LOADER_PATH)
