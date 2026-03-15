from ..features.motion_features import extract_motion_features
from ..features.appearance_features import extract_appearance_features
from ..unmasking.unmasking_loop import unmasking
from ..unmasking.anomaly_profile import compute_anomaly_score
from ..sliding_window.windowing import sliding_windows
from ..fusion.late_fusion import fuse_scores
from ..smoothing.temporal_smoothing import smooth_scores

def detect_anomalies(frames, w=10, stride=1, num_loops=10, top_features=50):
    anomaly_scores = []

    windows = sliding_windows(frames, w=w, stride=stride)
    for window in windows:
        ref_frames = window[:w]
        test_frames = window[w:]

        # Features
        motion_ref = extract_motion_features(ref_frames)
        motion_test = extract_motion_features(test_frames)
        motion_feat = np.vstack([motion_ref, motion_test])
        y = np.array([0]*len(motion_ref) + [1]*len(motion_test))
        motion_acc = unmasking(motion_feat, y, num_loops=num_loops, top_features=top_features)
        motion_score = compute_anomaly_score(motion_acc)

        appearance_ref = extract_appearance_features(ref_frames)
        appearance_test = extract_appearance_features(test_frames)
        appearance_feat = np.vstack([appearance_ref, appearance_test])
        y_app = np.array([0]*len(appearance_ref) + [1]*len(appearance_test))
        appearance_acc = unmasking(appearance_feat, y_app, num_loops=num_loops, top_features=top_features)
        appearance_score = compute_anomaly_score(appearance_acc)

        # Fusion
        score = fuse_scores(motion_score, appearance_score)
        anomaly_scores.append(score)

    # Temporal smoothing
    final_scores = smooth_scores(anomaly_scores)
    return final_scores
