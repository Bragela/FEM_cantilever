grid = CreateGrid()
x_list = []
y_list = []
s_list = []
for i in range(len(grid)):
    x = grid[i][0]
    y = grid[i][1]
    s = pred_grid_scores[i]
    x_list.append(x)
    y_list.append(y)
    s_list.append(s)

import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame({'x' : x_list, 'y' : y_list, 's' : s_list}) 
plt.figure(figsize=(10,10))
plt.scatter(df.x, df.y, s=10, c=df.s, cmap='terrain_r')

for pt in P:
    plt.plot(pt[0],pt[1], "b.")

plt.show()


meshPts = []
#print(pred_grid_scores.shape)
dx = (1.6*2) / 100
for i in range(round(pred_Ni)):
    # find min score and its index
    minPt, min_idx = torch.min(pred_grid_scores, dim=0)
    #print(min_idx)

    # translate from 1D to 2D, indexing in column and row
    col = min_idx % 100
    row = min_idx // 100 

    # find local domain of min value
    local_domain = pred_grid_scores.view(100, 100)[row-1:row+2, col-1:col+2]
    #print(space_around)
    # change values in local domain so that close pts are not slected next
    pred_grid_scores.view(100, 100)[row-1:row+2, col-1:col+2] = float('inf')
    #print(minPt)
    local_domain = pred_grid_scores.view(100, 100)[row-1:row+2, col-1:col+2]
    #print("after", space_around)

    pred_grid_scores = pred_grid_scores[:]
    local_domain = []