"""
Standalone YOLO coordinate test: validates letterbox + scale_boxes outside TouchDesigner.
Run: python tests/yolo_coord_test.py --image path/to/image.png --engine engines/yolo/yolo11n_640x640_b1_fp16.engine
Output: output_coord_test.png with boxes drawn in original resolution.
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
    parser = argparse.ArgumentParser(description="YOLO coord test: letterbox + scale_boxes validation")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument(
        "--engine",
        "-e",
        default=os.path.join(REPO_ROOT, "engines", "yolo", "yolo11n_640x640_b1_fp16.engine"),
        help="Path to .engine file",
    )
    parser.add_argument("--output", "-o", default="output_coord_test.png", help="Output image path")
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

    # Load image (BGR from cv2)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: could not load image {image_path}")
        sys.exit(1)
    orig_h, orig_w = img_bgr.shape[:2]
    print(f"Image: {orig_w}x{orig_h} (WxH)")

    # HWC -> CHW, BGR, float32 0-1
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    if img_t.max() > 1.0:
        img_t = img_t.clamp(0, 255) / 255.0

    engine_w, engine_h = 640, 640
    img_letterbox = letterbox_preprocess(img_t, engine_w, engine_h)
    # YOLO expects RGB; we have RGB
    img_letterbox = img_letterbox.cuda()

    model = YOLO(engine_path, task="detect")
    results = model.predict(
        source=img_letterbox,
        verbose=False,
        save=False,
        conf=args.conf,
        iou=args.iou,
    )
    torch.cuda.synchronize()

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print("No detections.")
        cv2.imwrite(args.output, img_bgr)
        print(f"Saved {args.output} (no boxes)")
        return

    xyxy_640 = boxes.xyxy
    if hasattr(xyxy_640, "cpu"):
        xyxy_640 = xyxy_640.cpu().numpy()
    xyxy_640 = np.asarray(xyxy_640, dtype=np.float32)

    img0_shape = (orig_h, orig_w)
    img1_shape = (engine_h, engine_w)
    xyxy_orig = ops.scale_boxes(img1_shape, xyxy_640.copy(), img0_shape)
    xywhn = ops.xyxy2xywhn(xyxy_orig, w=orig_w, h=orig_h, clip=True)
    if hasattr(xywhn, "cpu"):
        xywhn = xywhn.cpu().numpy()

    # Draw on original image
    out = img_bgr.copy()
    for i in range(len(xyxy_orig)):
        x1, y1, x2, y2 = xyxy_orig[i]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        conf = boxes.conf[i].item()
        cls_id = int(boxes.cls[i].item())
        cv2.putText(out, f"c{cls_id} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"  box {i}: xyxy_640={xyxy_640[i].tolist()} -> xyxy_orig=({x1},{y1},{x2},{y2}) xywhn={xywhn[i].tolist()}")

    # Validate coords
    for i in range(len(xyxy_orig)):
        x1, y1, x2, y2 = xyxy_orig[i]
        ok = 0 <= x1 <= orig_w and 0 <= x2 <= orig_w and 0 <= y1 <= orig_h and 0 <= y2 <= orig_h
        print(f"  box {i} in [0,{orig_w}]x[0,{orig_h}]: {'OK' if ok else 'OUT OF BOUNDS'}")

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    cv2.imwrite(out_path, out)
    print(f"Saved {out_path} ({len(xyxy_orig)} boxes in original {orig_w}x{orig_h})")


if __name__ == "__main__":
    main()
