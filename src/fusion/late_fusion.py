import numpy as np

def fuse_scores(motion_scores, appearance_scores):
    return np.mean([motion_scores, appearance_scores], axis=0)
