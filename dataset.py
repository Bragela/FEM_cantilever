from tkinter import Y
from matplotlib.pyplot import axis
from sympy import Float
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import random
import math
import matplotlib.pylab as plt
from sklearn import preprocessing

def Angle(a):
    unit_x = [1,0,0]
    return 2*math.pi-np.arccos(np.dot(a, unit_x)/ (np.linalg.norm(a) * np.linalg.norm(unit_x)))


class GridDataset(Dataset):
    def __init__(self, root_dir="data", split="train", force_scaler=None, coords_scaler=None):
        self.data_path = f"{root_dir}/{split}" 
        self.data = [folder for folder in os.listdir(self.data_path)]
        self.split = split
        self.force_scaler = force_scaler
        self.coords_scaler = coords_scaler

    def __len__(self):
        return len(self.data)
        #return 1


    def __getitem__(self, idx):
        folder_name = self.data[idx]
        full_path = f"{self.data_path}/{folder_name}"

     
        forces = []
        with open(f'{full_path}/Input.txt','r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    F = line.rstrip('\n')
                    forces.append(float(F))
                else:
                    x, y, z = line.rstrip('\n').split(',')
                    vector = [float(x), float(y), float(z)]
                   
        
        phi = Angle(vector)
        rot_mat = np.array([[np.cos(phi), -np.sin(phi), 0],[np.sin(phi), np.cos(phi), 0],[0, 0, 1]])

        vector = vector @ rot_mat.T

        if self.force_scaler != None:
            forces = self.force_scaler.transform(np.array(forces).reshape(-1,1))
            forces = torch.from_numpy(forces).float().squeeze(0)
        else:
            forces = torch.tensor(forces)

        FEM_stress = []
        FEM_disp = []
        coords = []

        with open(f'{full_path}/Stress_and_Disp.txt','r') as f:
            for i, line in enumerate(f):
                ux, uy, uz, stress, x, y, z = line.rstrip('\n').split(',')
                FEM_stress.append(float(stress))
                pt = [float(x), float(y), float(z)]
                pt = pt @ rot_mat.T

                disp = [float(ux), float(uy), float(uz)]
                disp = disp @ rot_mat.T
                
                coords.append(pt)
                FEM_disp.append(disp)

        FEM_disp = np.array(FEM_disp)
        coords = np.array(coords)
        coords_original = torch.tensor(coords)
        coords = torch.tensor(coords)
        FEM_stress = torch.tensor(FEM_stress)
        FEM_disp = torch.tensor(FEM_disp)
        vector = torch.tensor(vector)


        if self.coords_scaler != None:
            coords = self.coords_scaler.transform(np.array(coords).reshape(-1,3))
            coords = torch.from_numpy(coords).float().squeeze(0)
            vector = self.coords_scaler.transform(np.array(vector).reshape(-1,3))
            vector = torch.from_numpy(vector).float().squeeze(0)


        return forces, coords, coords_original, FEM_stress, FEM_disp, vector

def main():

    dataset = GridDataset(split='validation')
    forces, coords, coords_original, FEM_stress, FEM_disp, vector = dataset[7]

    coords_original = coords_original.cpu().squeeze().detach().numpy()
    FEM_disp = FEM_disp.cpu().squeeze().detach().numpy()*100
    coords = coords.cpu().squeeze().detach().numpy()
    max, min = coords.max().item(), coords.min().item()

    FEM_x = coords_original[:,0] + FEM_disp[:,0]
    FEM_y = coords_original[:,1] + FEM_disp[:,1]
    FEM_z = coords_original[:,2] + FEM_disp[:,2]

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(FEM_x,FEM_y,FEM_z, s=25, c = FEM_stress, cmap='viridis')

    ax.set_xlim(min,max)
    ax.set_ylim(-1000,1000)
    ax.set_zlim(-1000,1000)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()