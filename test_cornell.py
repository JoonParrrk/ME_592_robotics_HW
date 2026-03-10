import matplotlib.pyplot as plt
import os
import numpy as np

if not hasattr(np, 'float'):
    np.float = float

from utils.dataset_processing.grasp import Grasp

archive_path = r'C:\Users\park2\Desktop\ME592_HW2\robotic-grasping\archive'
image_id = 'pcd0100'  
subfolder = '01'

rgb_path = os.path.join(archive_path, subfolder, image_id + 'r.png')

def load_cornell_rects(path):
    grasps = []
    if not os.path.exists(path):
        return grasps
        
    with open(path, 'r') as f:
        content = f.readlines()
        for i in range(0, len(content), 4):
            try:
                points = []
                for j in range(4):
                    points.append(list(map(float, content[i+j].strip().split())))
                
                # p is an array of [X, Y] coordinates
                p = np.array(points)
                
                center_x = p[:, 0].mean()
                center_y = p[:, 1].mean()
                
                # Feed it to the Grasp object as [Y, X] (Row, Col)
                center = np.array([center_y, center_x])
                
                dx = p[1, 0] - p[0, 0]
                dy = p[1, 1] - p[0, 1]
                angle = np.arctan2(dy, dx)
                
                width = np.linalg.norm(p[1] - p[0])
                height = np.linalg.norm(p[2] - p[1])
                
                # Create the object
                grasps.append(Grasp(center, angle, width, height))
            except Exception:
                continue
    return grasps

if not os.path.exists(rgb_path):
    print(f"Error: Path {rgb_path} not found. Check your 'archive' folder.")
else:
    rgb_img = plt.imread(rgb_path)
    pos_grasps = load_cornell_rects(os.path.join(archive_path, subfolder, image_id + 'cpos.txt'))
    neg_grasps = load_cornell_rects(os.path.join(archive_path, subfolder, image_id + 'cneg.txt'))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_img)

    for g in pos_grasps:
        g.plot(ax, color='lime')

    for g in neg_grasps:
        g.plot(ax, color='red')

    plt.title(f"Cornell Dataset: {image_id} (Green=Pos, Red=Neg)")
    plt.show()