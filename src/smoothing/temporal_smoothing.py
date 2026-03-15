import cv2
import numpy as np

def smooth_scores(scores, kernel_size=5, sigma=1.0):
    scores = np.array(scores, dtype=np.float32)
    smoothed = cv2.GaussianBlur(scores, (kernel_size,1), sigma)
    return smoothed.flatten()
