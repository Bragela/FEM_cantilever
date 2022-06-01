from NN import NeuralNet
import torch
from dataset import GridDataset
from trainer import train
from torch.utils.data import DataLoader
from torch import autograd, dropout
import numpy as np
import random 
import wandb
from sklearn import preprocessing


# Seed
seed = 11
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

id = 'h8fi4hv3'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getAllForces():
    training_dataset = GridDataset()
    data = torch.stack([d[0] for d in training_dataset]).numpy()
    data_1 = torch.stack([d[1] for d in training_dataset]).view(-1,3).numpy()
    return data, data_1


def run():
    with wandb.init(project="FEM_case_cantilever", entity="master-thesis-ntnu", config=dict) as run:
        config = wandb.config

        num_epochs = config.epochs
        batch_size = 128
        batch_size_train = 128
        learning_rate = config.learning_rate
    
        data, data_1 = getAllForces()
    
        forces_scaler = preprocessing.StandardScaler().fit(data)
        coords_scaler = preprocessing.StandardScaler().fit(data_1)

        use_existing_model = False

        # Dataset
        train_dataset = GridDataset(force_scaler=forces_scaler, coords_scaler=coords_scaler)
        test_dataset = GridDataset(split="test",force_scaler=forces_scaler, coords_scaler=coords_scaler)
        validation_dataset = GridDataset(split="validation",force_scaler=forces_scaler, coords_scaler=coords_scaler)

        # Data loader
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,  drop_last=True, num_workers=3, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=3, pin_memory=True)
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=3, pin_memory=True)

        model = NeuralNet(config.layer_sizes+[4], dropout=config.dropout).to(device)
        if use_existing_model:
            model.load_state_dict(torch.load("model.pth")["state_dict"])
        #with autograd.detect_anomaly():
        wandb.init(project="FEM_case_cantilever", entity="master-thesis-ntnu", config=dict)

        train(model, num_epochs, batch_size, train_loader, test_loader, validation_loader, learning_rate=learning_rate, device=device)
        
        # Save model
        config = {
            "state_dict": model.state_dict()
        }

        torch.save(config, "model.pth")

if __name__ == "__main__":
    wandb.agent(id, project="FEM_case_cantilever", entity="master-thesis-ntnu", function=run, count=30)