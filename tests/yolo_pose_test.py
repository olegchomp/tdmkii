"""
Standalone YOLO pose test: inference with pose model, draws boxes + keypoints.
Run: python tests/yolo_pose_test.py --image path/to/image.png
Output: tests/output_pose_test.png
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# COCO pose skeleton (pairs of keypoint indices to draw lines)
POSE_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # torso, legs
]


def letterbox_preprocess(image: torch.Tensor, engine_w: int, engine_h: int) -> torch.Tensor:
    """(C, H, W) -> (1, 3, engine_h, engine_w) with Ultralytics-style letterbox."""
    image = image[:3, :, :]
    src_h, src_w = image.shape[1], image.shape[2]
    if src_w <= 0 or src_h <= 0:
        image = image.unsqueeze(0)
        if image.max() > 1.0:
            image = image.clamp(0.0, 255.0) / 255.0
        return image
    ratio = min(engine_w / src_w, engine_h / src_h)
    new_w = round(src_w * ratio)
    new_h = round(src_h * ratio)
    image = F.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )
    dw = engine_w - new_w
    dh = engine_h - new_h
    pad_left = round((dw / 2) - 0.1)
    pad_right = dw - pad_left
    pad_top = round((dh / 2) - 0.1)
    pad_bottom = dh - pad_top
    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=114.0 / 255.0)
    if image.max() > 1.0:
        image = image.clamp(0.0, 255.0) / 255.0
    return image


def main():
    parser = argparse.ArgumentParser(description="YOLO pose test: boxes + keypoints")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument(
        "--engine",
        "-e",
        default=os.path.join(REPO_ROOT, "engines", "yolo", "yolo11s-pose_640x640_b1_fp16.engine"),
        help="Path to pose .engine file",
    )
    parser.add_argument("--output", "-o", default="tests/output_pose_test.png", help="Output image path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    image_path = args.image
    if not os.path.isabs(image_path):
        image_path = os.path.join(REPO_ROOT, image_path)
    if not os.path.isfile(image_path):
        print(f"Error: image not found: {image_path}")
        sys.exit(1)
    engine_path = args.engine
    if not os.path.isabs(engine_path):
        engine_path = os.path.join(REPO_ROOT, engine_path)
    if not os.path.isfile(engine_path):
        print(f"Error: engine not found: {engine_path}")
        sys.exit(1)

    from ultralytics import YOLO
    from ultralytics.utils import ops

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: could not load image {image_path}")
        sys.exit(1)
    orig_h, orig_w = img_bgr.shape[:2]
    print(f"Image: {orig_w}x{orig_h} (WxH)")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    if img_t.max() > 1.0:
        img_t = img_t.clamp(0, 255) / 255.0

    engine_w, engine_h = 640, 640
    img_letterbox = letterbox_preprocess(img_t, engine_w, engine_h)
    img_letterbox = img_letterbox.cuda()

    model = YOLO(engine_path, task="pose")
    results = model.predict(
        source=img_letterbox,
        verbose=False,
        save=False,
        conf=args.conf,
        iou=args.iou,
    )
    torch.cuda.synchronize()

    out = img_bgr.copy()
    boxes = results[0].boxes
    keypoints = getattr(results[0], "keypoints", None)

    img0_shape = (orig_h, orig_w)
    img1_shape = (engine_h, engine_w)

    if boxes is not None and len(boxes) > 0:
        xyxy_640 = boxes.xyxy
        if hasattr(xyxy_640, "cpu"):
            xyxy_640 = xyxy_640.cpu().numpy()
        xyxy_640 = np.asarray(xyxy_640, dtype=np.float32)
        xyxy_orig = ops.scale_boxes(img1_shape, xyxy_640.copy(), img0_shape)

        for i in range(len(xyxy_orig)):
            x1, y1, x2, y2 = xyxy_orig[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            conf = boxes.conf[i].item()
            cv2.putText(out, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"Boxes: {len(xyxy_orig)}")

    if keypoints is not None and len(keypoints) > 0:
        kpt_data = keypoints.data
        if hasattr(kpt_data, "cpu"):
            kpt_data = kpt_data.cpu().numpy()
        kpt_data = np.asarray(kpt_data, dtype=np.float32)
        kpt_xy = kpt_data[..., :2].copy()
        ops.scale_coords(img1_shape, kpt_xy, img0_shape, padding=True)

        for i in range(kpt_xy.shape[0]):
            # Draw skeleton
            for (a, b) in POSE_SKELETON:
                if a < kpt_xy.shape[1] and b < kpt_xy.shape[1]:
                    va = kpt_data[i, a, 2] if kpt_data.shape[-1] >= 3 else 1.0
                    vb = kpt_data[i, b, 2] if kpt_data.shape[-1] >= 3 else 1.0
                    if va > 0.5 and vb > 0.5:
                        pt1 = (int(kpt_xy[i, a, 0]), int(kpt_xy[i, a, 1]))
                        pt2 = (int(kpt_xy[i, b, 0]), int(kpt_xy[i, b, 1]))
                        cv2.line(out, pt1, pt2, (0, 255, 255), 2)
            # Draw keypoints
            for k in range(kpt_xy.shape[1]):
                v = float(kpt_data[i, k, 2]) if kpt_data.shape[-1] >= 3 else 1.0
                if v > 0.5:
                    x, y = int(kpt_xy[i, k, 0]), int(kpt_xy[i, k, 1])
                    cv2.circle(out, (x, y), 4, (0, 0, 255), -1)
                    cv2.putText(out, str(k), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        print(f"Keypoints: {kpt_xy.shape[0]} persons x {kpt_xy.shape[1]} pts")

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.normpath(os.path.join(REPO_ROOT, out_path))
    cv2.imwrite(out_path, out)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
