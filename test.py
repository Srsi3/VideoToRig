import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_pose_model, inference_topdown_pose_model

import sys
sys.path.append('MotionBERT_RELEASE')
from common.utils.preprocessing import pre_normalization
from common.utils.vis import vis_keypoints_3d
from model.motionbert import MotionBERT
import torch

# ---------- Paths ----------
video_path = 'input_video.mp4'
output_json = '3d_pose_output.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- MMPose (2D) ----------
pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

det_model = init_detector(det_config, det_checkpoint, device=device)
pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)

# ---------- MotionBERT (3D) ----------
motionbert = MotionBERT(input_feat=17*2)
state_dict = torch.load('MotionBERT_RELEASE/checkpoints/motionbert-base-ft-h36m.pth', map_location=device)
motionbert.load_state_dict(state_dict['model'])
motionbert.to(device)
motionbert.eval()

# COCO → H36M mapping (17 keypoints)
COCO_TO_H36M = [11, 12, 5, 6, 0, 7, 8, 9, 10, 13, 14, 15, 16, 2, 1, 4, 3]

# ---------- 2D Inference ----------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

all_person_tracks = {}  # {person_id: [[x,y,conf]*17]*T}

print("🧠 Step 1: Estimating 2D poses...")
for frame_idx in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people
    det_result = inference_detector(det_model, frame)
    person_bboxes = [{'bbox': bbox} for bbox in det_result[0] if bbox[4] > 0.5]

    # Pose estimation
    pose_results = inference_topdown_pose_model(
        pose_model, frame, person_bboxes, bbox_format='xyxy')

    for person_idx, person in enumerate(pose_results):
        keypoints = person.pred_instances.keypoints[0].cpu().numpy()  # [17, 2]
        scores = person.pred_instances.keypoint_scores[0].cpu().numpy()  # [17]

        h36m_keypoints = []
        for idx in COCO_TO_H36M:
            x, y = keypoints[idx]
            conf = scores[idx]
            h36m_keypoints.append([x, y, conf])

        track_id = f'person_{person_idx}'
        if track_id not in all_person_tracks:
            all_person_tracks[track_id] = []
        all_person_tracks[track_id].append(h36m_keypoints)

cap.release()

# ---------- 3D Lifting ----------
print("🔼 Step 2: Lifting to 3D with MotionBERT...")

output_3d = {}  # {person_id: [[[x,y,z],...]*17]*T}

for pid, kps2d_seq in all_person_tracks.items():
    kp2d = np.array(kps2d_seq)  # [T, 17, 3]
    vis = kp2d[..., 2] > 0.3
    kp2d[..., :2] = pre_normalization(kp2d[..., :2])  # normalize x,y
    kp2d_input = torch.tensor(kp2d[..., :2].reshape(kp2d.shape[0], -1), dtype=torch.float32).to(device).unsqueeze(0)  # [1, T, 34]

    with torch.no_grad():
        preds = motionbert(kp2d_input).cpu().numpy()  # [1, T, 51]
    kp3d = preds[0].reshape(-1, 17, 3).tolist()  # [T, 17, 3]

    output_3d[pid] = kp3d

# ---------- Save JSON ----------
with open(output_json, 'w') as f:
    json.dump(output_3d, f, indent=2)

print(f"✅ Done. Saved 3D poses to: {output_json}")
