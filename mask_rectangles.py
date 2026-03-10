import matplotlib.pyplot as plt
import os
import glob
import numpy as np
if not hasattr(np, 'float'):
    np.float = float

from utils.dataset_processing.grasp import Grasp

# 1. Setup Directories
base_dir = r'C:\Users\park2\Desktop\ME592_HW2\robotic-grasping'
archive_path = os.path.join(base_dir, 'archive')

output_dir = os.path.join(base_dir, 'annotated_images')
good_dir = os.path.join(output_dir, 'good')
bad_dir = os.path.join(output_dir, 'bad')

os.makedirs(good_dir, exist_ok=True)
os.makedirs(bad_dir, exist_ok=True)

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
                
                p = np.array(points)
                
                # Extract X and Y separately
                center_x = p[:, 0].mean()
                center_y = p[:, 1].mean()
                
                # Feed to Grasp object as [Row, Col] -> [Y, X]
                center = np.array([center_y, center_x])
                
                dx = p[1, 0] - p[0, 0]
                dy = p[1, 1] - p[0, 1]
                angle = np.arctan2(dy, dx)
                
                width = np.linalg.norm(p[1] - p[0])
                height = np.linalg.norm(p[2] - p[1])
                
                grasps.append(Grasp(center, angle, width, height))
            except Exception:
                continue
    return grasps

print("Scanning archive for images...")
image_files = glob.glob(os.path.join(archive_path, '*', '*r.png'))
total_images = len(image_files)
print(f"Found {total_images} images. Starting generation...")

for count, rgb_path in enumerate(image_files):
    folder = os.path.dirname(rgb_path)
    filename = os.path.basename(rgb_path)
    image_id = filename.replace('r.png', '')
    
    pos_path = os.path.join(folder, image_id + 'cpos.txt')
    neg_path = os.path.join(folder, image_id + 'cneg.txt')
    
    rgb_img = plt.imread(rgb_path)
    pos_grasps = load_cornell_rects(pos_path)
    neg_grasps = load_cornell_rects(neg_path)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(rgb_img)
    for g in pos_grasps:
        g.plot(ax, color='lime')
    plt.title(f"Good Grasps: {image_id}")
    ax.axis('off') 
    fig.savefig(os.path.join(good_dir, f"{image_id}_good.png"), bbox_inches='tight')
    plt.close(fig) #free up memory!!!
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(rgb_img)
    for g in neg_grasps:
        g.plot(ax, color='red')
    plt.title(f"Bad Grasps: {image_id}")
    ax.axis('off')
    fig.savefig(os.path.join(bad_dir, f"{image_id}_bad.png"), bbox_inches='tight')
    plt.close(fig) 
    
    if count % 50 == 0:
        print(f"Processed {count} / {total_images} images...")

print(f"DONE! Check the '{output_dir}' folder.")