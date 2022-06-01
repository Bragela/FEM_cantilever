from matplotlib import projections
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import tqdm
import wandb
from NN import NeuralNet
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
from sklearn import preprocessing
import math
from matplotlib import cm

colormap = cm.get_cmap('hsv', 24)

@torch.no_grad()
def no_grad_loop(data_loader, model, png_cnt, epoch=2, device="cuda", batch_size = 64):

    no_grad_loss = 0
    no_grad_disp_loss = 0
    no_grad_stress_loss = 0
    cnt = 0

    for i, (forces, coords, coords_original, FEM_stress, FEM_disp, vectors) in enumerate(data_loader):


        # transfer data to device
        forces = forces.to(device)
        FEM_stress = FEM_stress.to(device)
        FEM_disp = FEM_disp.to(device)
        coords = coords.to(device)
        coords_original = coords_original.to(device)
        vectors = vectors.to(device)

        with autocast():
            disp_pred, stress_pred = model(forces, coords, vectors)
            disp_loss = F.l1_loss(disp_pred, FEM_disp)
            stress_loss = F.l1_loss(stress_pred, FEM_stress)
            loss = disp_loss + stress_loss

        no_grad_disp_loss += disp_loss
        no_grad_stress_loss += stress_loss

        no_grad_loss += loss
        cnt += 1

        if i == len(data_loader) -1:

            case = 0
            fig = plt.figure(figsize=plt.figaspect(0.5))
            coords_original = coords_original[case]
            coords = coords[case]
            FEM_disp = FEM_disp[case]

            # FEM_plt

            coords_original = coords_original.cpu().squeeze().detach().numpy()
            coords = coords.cpu().squeeze().detach().numpy()
            FEM_disp = FEM_disp.cpu().squeeze().detach().numpy()*100
            max, min = coords_original.max().item(), coords_original.min().item()

            FEM_x = coords_original[:,0] + FEM_disp[:,0]
            FEM_y = coords_original[:,1] + FEM_disp[:,1]
            FEM_z = coords_original[:,2] + FEM_disp[:,2]



            stress_FEM = FEM_stress[case].cpu().squeeze().detach().numpy()
            max_a = stress_FEM.max().item()
            ax = fig.add_subplot(1,2,1, projection = '3d')
            a = ax.scatter(FEM_x, FEM_y, FEM_z, s = 10, c= stress_FEM, cmap = 'viridis', vmin= 0, vmax=max_a)
            fig.colorbar(a, pad=0.1, shrink=0.5, aspect=10)
            ax.set_title(f'FEM stresses.\n Max von Mises stress: {round(max_a,2)}')
            ax.set_xlim(min,max)
            ax.set_ylim(-1000,1000)
            ax.set_zlim(-1000,1000)

            # pred plot
            stress_pred = stress_pred[case].cpu().squeeze().detach().numpy()
            disp_pred = disp_pred[case].cpu().squeeze().detach().numpy()*100

            pred_x = coords_original[:,0] + disp_pred[:,0]
            pred_y = coords_original[:,1] + disp_pred[:,1]
            pred_z = coords_original[:,2] + disp_pred[:,2]

            ax = fig.add_subplot(1,2,2, projection = '3d')
            max_b = stress_pred.max().item()
            b = ax.scatter(pred_x, pred_y, pred_z, s = 10, c= stress_pred, cmap = 'viridis', vmin= 0, vmax=max_a)
            fig.colorbar(b, pad=0.1, shrink=0.5, aspect=10)
            ax.set_title(f'Predicted stresses.\n Max von Mises stress: {round(max_b,2)}')
            ax.set_xlim(min,max)
            ax.set_ylim(-1000,1000)
            ax.set_zlim(-1000,1000)

            if png_cnt != 100000000:
                plt.savefig(f'./GIFS/pngs/{png_cnt}.png')

            plt.savefig('train.png')
            wandb.log({"images": wandb.Image("train.png")}, commit=False)

            plt.close(fig)
            plt.close('all')

    return no_grad_loss/cnt, no_grad_disp_loss/cnt, no_grad_stress_loss/cnt


def train(model: NeuralNet, num_epochs, batch_size, train_loader, test_loader, validation_loader, learning_rate=1e-3, device="cuda"):
    no_grad_loss, no_grad_disp_loss, no_grad_stress_loss = no_grad_loop(validation_loader, model, png_cnt=0, epoch=0, device="cuda", batch_size=batch_size)
    curr_lr =  learning_rate

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=4)

    # training loop
    iter = 0
    png_cnt = 1
    training_losses = {
        "train": {},
        "valid": {}
    }
    scaler = GradScaler()
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        loader = tqdm.tqdm(train_loader)
        for forces, coords, coords_original, FEM_stress, FEM_disp, vectors in loader:

            # transfer data to device
            forces = forces.to(device)
            FEM_stress = FEM_stress.to(device)
            FEM_disp = FEM_disp.to(device)
            coords = coords.to(device)
            vectors = vectors.to(device)

            # Forward pass
            with autocast():
                disp_pred, stress_pred = model(forces, coords, vectors)
                disp_loss = F.l1_loss(disp_pred, FEM_disp)
                stress_loss = F.l1_loss(stress_pred, FEM_stress)
                loss = disp_loss + stress_loss

            loader.set_postfix(stress = loss.item())

            # Backward and optimize
            optimizer.zero_grad()               # clear gradients
            scaler.scale(loss).backward()       # calculate gradients

            # grad less than 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)

            scaler.step(optimizer)
            scaler.update()
            iter += 1

            training_losses["train"][iter] = loss.item()

            if (iter+1) % 5 == 0:

                # validation loop
                model = model.eval()
                valid_loss, valid_disp_loss, valid_stress_loss = no_grad_loop(validation_loader, model, png_cnt, epoch, device="cuda", batch_size=batch_size)
                png_cnt += 1
                scheduler.step(valid_loss)
                curr_lr =  optimizer.param_groups[0]["lr"]
                wandb.log({"valid loss": valid_loss.item(), "valid disp loss": valid_disp_loss.item(), "valid stress loss": valid_stress_loss.item(), "lr": curr_lr}, commit=False)
                model = model.train()
            wandb.log({"train loss": loss.item(), "train disp loss": disp_loss.item(), "train stress loss": stress_loss.item(),})

    # test loop
    test_loss, test_disp_loss, test_stress_loss = no_grad_loop(test_loader, model, png_cnt=100000000, device="cuda", batch_size=batch_size)
    print(f'testloss: tot={test_loss:.5f}')