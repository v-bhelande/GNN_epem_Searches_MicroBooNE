# GNN_epem_Searches_MicroBooNE

This repository contains my GNN code and data analysis procedures from my internship at Los Alamos National Laboratory (LANL) through the Department of Energy Science Undergraduate Laboratory Internships (DOE SULI) program. The files shown here are forked from the original repo friedgreentomatoes owned by Professor Mark Ross-Lonergan at Columbia University, New York. A brief explanation of the 4 files are shown below:

1. predict_angle.py: Script that loads training and test data from data_generator.py which is used to train GNN from gnn_model.py. Records training and test model with additional features to save model and all output.
2. data_generator.py: Script that takes raw simulated particle events in MicroBooNE detector geometry and prepares batches for training and testing GNN.
3. gnn_model.py: PointNet++ based GNN model used to predict opening angles in epem decay events. The model attached can predict opening angles down to 4.3 degrees with some reliability, further hypertuning of parameters will be done to improve model performance.
4. plotting_scripts.py: Series of useful plotting functions to visualize and evaluate model performance.
