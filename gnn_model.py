import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, global_mean_pool
from torch.autograd import Variable
from torch_scatter import scatter_max
from plotting_scripts import PlotInputs

# Only taking in POS
# Modified from https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
class TNet3dpos(torch.nn.Module):
    def __init__(self):
        super(TNet3dpos, self).__init__()
        self.mlp1 = MLP([4, 64, 128, 1024])
        self.mlp2 = MLP([1024, 512, 256, 9])

    def forward(self, pos, x, batch):

        # Apply convolutional layer
        x = self.mlp1(torch.cat([pos, x], dim=1))

        # Max Pooling
        x, _ = scatter_max(x, batch, 0)

        # Fully Connected Layers (Linear)
        x = self.mlp2(x)
        x2 = x[batch].repeat(1,1)

        # Add identity matrix to output as bias
        batchsize = pos.size()[0]
        identity = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            dev2 = x.get_device()
            identity = identity.cuda(dev2)
        x = x2+identity
        x = x.view(-1, 3, 3)
        
        # Apply affine transformation to input points
        pos = torch.unsqueeze(pos, dim=1)
        pos = torch.bmm(pos, x)
        pos = torch.squeeze(pos, dim = 1)
        return pos

# The individual "SetAbstraction" that is repeated twice in PointNet++
class SAModuleMSG(torch.nn.Module):
    def __init__(self, radius_list, ratio, mlp_list):
        super(SAModuleMSG, self).__init__()
        self.radius_list = radius_list # Radius in which to group points around sampled ones
        self.ratio = ratio             # Ratio of points sampled to total points
        self.conv_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            self.conv_blocks.append(PointNetConv(MLP(mlp_list[i]), add_self_loops=False, aggr="max"))

    def forward(self, x, pos, batch):

        # Set abstraction consists of 3 steps: Sampling, Grouping and PointNetConvolution
        new_x_list = []

        # Sample
        idx = fps(pos, batch, self.ratio)
        x_dst = None if x is None else x[idx]

        # Loop over the ratios
        for i, conv in enumerate(self.conv_blocks):
            # Group - use a radius to grab neighbours up to a max
            row, col = radius(pos, pos[idx], self.radius_list[i], batch, batch[idx], max_num_neighbors=64)
            edge_index = torch.stack([col, row], dim=0)
            
            # Perform the convolutions
            new_x = conv((x, x_dst), (pos, pos[idx]), edge_index)
            new_x_list.append(new_x)
            
        # Concatenate features from each embedded layer
        new_x_concat = torch.cat(new_x_list, dim=1)
        
        return new_x_concat, pos[idx], batch[idx]

# The global set abstraction then learns from the various features at different scales learnt in the previous two
class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x, _ = scatter_max(x, batch, 0)
        return x
    
# PointNet++ MSG
class PointNetMSG(torch.nn.Module):
    def __init__(self):
        super().__init__()

        sseed = 330
        torch.manual_seed(sseed)
        torch.cuda.manual_seed(sseed)

        # T-Net module
        self.tnet = TNet3dpos()
        self.sa1_module = SAModuleMSG([1, 2, 4], 0.25, [[1 + 3, 32, 32, 64], [1 + 3, 64, 64, 128], [1 + 3, 64, 96, 128]])
        self.sa2_module = SAModuleMSG([2, 4, 8], 0.5, [[320 + 3, 64, 64, 128], [320 + 3, 128, 128, 256], [320 + 3, 128, 128, 256]])
        self.sa3_module = GlobalSAModule(MLP([640 + 3, 256, 512, 1024]))
        
        #A multilayer peceptron that returns the one singular regression parameter
        #modified to go to 32, and then a FC layer to a single value for regression
        #self.mlp = MLP([1024, 512, 256])
        self.mlp = MLP([1024, 512, 256, 1], dropout=0.5, norm=None)

    def forward(self, data):
        # Use T-Net to transform input point cloud
        tnet_out = self.tnet(data.pos, data.x, data.batch)
            
        # Feed into PointNet SetAbstraction layers
        sa1_out = self.sa1_module(data.x, tnet_out, data.batch)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        out = sa3_out        
        out = self.mlp(out)

        return out
