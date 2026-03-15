import cv2
import numpy as np

def extract_motion_features(frames, cube_size=(10,10,5)):
    H, W, T = cube_size
    motion_features = []
    for y in range(0, frames[0].shape[0]-H+1, H):
        for x in range(0, frames[0].shape[1]-W+1, W):
            for t in range(0, len(frames)-T+1):
                cube = np.stack([frames[t+i][y:y+H, x:x+W] for i in range(T)], axis=-1)
                gx = cv2.Sobel(cube, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(cube, cv2.CV_64F, 0, 1, ksize=3)
                gz = np.diff(cube, axis=-1)
                grad = np.sqrt(gx**2 + gy**2 + gz**2).flatten()
                motion_features.append(grad)
    return np.array(motion_features)
