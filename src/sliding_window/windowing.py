def sliding_windows(frames, w=10, stride=1):
    windows = []
    for start in range(0, len(frames)-2*w+1, stride):
        windows.append(frames[start:start+2*w])
    return windows
