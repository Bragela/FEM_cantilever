from numpy import dtype
import torch.nn as nn
import torch
import torch.nn.functional as F


class NeuralNet(nn.Module):

    def __init__(self, layer_sizes, dropout):
        super(NeuralNet, self).__init__()
        layers = []
        input_features = 7
        for i, output_features in enumerate(layer_sizes):
            layers.append(nn.Linear(input_features, output_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p = dropout))
            input_features = output_features
        self.layers = nn.Sequential(*layers)


    def forward(self, forces, coords, vectors):             # [B,1] [B,P,3]
        
        B = forces.shape[0]
        P = coords.shape[1]
        vectors = vectors.view(B,1,3).repeat(1,P,1)         # [B,P,3]
        forces = forces.view(B,1,1).repeat(1,P,1)           # [B,P,1]

        a = torch.cat((forces,vectors,coords), dim=2)       # [B,P,7]

        out = self.layers(a)                                # [B,P,4]
        disp_pred = out[:,:,:3]                             # [B,P,3]
        stress_pred = out[:,:,3]                            # [B,P]
   
        return disp_pred, stress_pred
