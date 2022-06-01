from tabnanny import verbose
from sympy import re
import wandb
from NN import NeuralNet
import torch
from dataset import GridDataset
from trainer import train
from torch.utils.data import DataLoader
from torch import autograd
import numpy as np
import random 
import wandb
from sklearn import preprocessing


# Fixed seed
seed = 11
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import dataset

def getAllForces():
    training_dataset = GridDataset(split = 'test')
    data = torch.stack([d[0] for d in training_dataset]).numpy()
    data_1 = torch.stack([d[1] for d in training_dataset]).view(-1,3).numpy()
    return data, data_1

def run():
    layer_sizes = [128,128,4]
    num_epochs = 256
    batch_size = 128
    batch_size_train = 512
    learning_rate = 0.01

    dict = {
        'learning_rate': learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size
    }

    use_existing_model = False

    data, data_1 = getAllForces()

    forces_scaler = preprocessing.StandardScaler().fit(data)
    coords_scaler = preprocessing.StandardScaler().fit(data_1)
    
    # Dataset
    train_dataset = GridDataset(force_scaler=forces_scaler, coords_scaler=coords_scaler)
    test_dataset = GridDataset(split="test",force_scaler=forces_scaler, coords_scaler=coords_scaler)
    validation_dataset = GridDataset(split="validation",force_scaler=forces_scaler, coords_scaler=coords_scaler)

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,  drop_last=True, num_workers=3, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=3, pin_memory=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=3, pin_memory=True)



    model = NeuralNet(layer_sizes, dropout = 0.1).to(device)
    if use_existing_model:
        model.load_state_dict(torch.load("./003model.pth")["state_dict"])
    wandb.init(project="FEM_case_cantilever", entity="master-thesis-ntnu", config=dict)
    wandb.watch(model, log='all', log_freq=10)
    train(model, num_epochs, batch_size, train_loader, test_loader, validation_loader, learning_rate=learning_rate, device=device)

    # Save model
    config = {
        "state_dict": model.state_dict()
    }

    torch.save(config, "model.pth")

if __name__ == "__main__":
    run()