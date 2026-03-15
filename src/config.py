WINDOW_SIZE = 10       # number of frames per segment
STRIDE = 1             

NUM_LOOPS = 10         # number of unmasking iterations
TOP_FEATURES = 50      # number of top features to remove per iteration

NUM_BINS = (2, 2)      # 2x2 bins for motion and appearance features

CNN_MODEL = "vgg-f"    
FRAME_SIZE = (224, 224) 

CUBE_SIZE = (10, 10, 5) # (H, W, T)
FRAME_RESIZE = (160, 120) 

GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA = 1.0
