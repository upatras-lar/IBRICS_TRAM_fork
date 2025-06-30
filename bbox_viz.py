#!/usr/bin/env python3
import json
import cv2
import os
import sys
import argparse
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(
        description="Draw COCO-style bounding boxes (including multiple per image)")
    p.add_argument("-a", "--annotations", required=True,
                   help="Path to COCO annotations JSON file")
    p.add_argument("-i", "--images", required=True,
                   help="Directory where the images live")
    p.add_argument("-o", "--output", default="images_with_bboxes",
                   help="Directory to save the visualized images")
    p.add_argument("-s", "--show", action="store_true",
                   help="Also pop up each image with cv2.imshow")
    return p.parse_args()

def main():
    args = parse_args()

    # Load annotations
    try:
        with open(args.annotations, "r") as f:
            coco = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load annotations: {e}", file=sys.stderr)
        sys.exit(1)

    # Build image‐ID → filename map
    img_map = {img["id"]: img["file_name"] for img in coco.get("images", [])}

    # Group all bounding boxes by image ID
    bboxes_per_img = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        bbox   = ann.get("bbox", [])
        if img_id is not None and bbox:
            bboxes_per_img[img_id].append(bbox)

    os.makedirs(args.output, exist_ok=True)

    # Draw all boxes for each image
    for img_id, bboxes in bboxes_per_img.items():
        fname = img_map.get(img_id)
        if not fname:
            print(f"[WARN] No filename for image_id {img_id}", file=sys.stderr)
            continue

        img_path = os.path.join(args.images, fname)
        if not os.path.isfile(img_path):
            print(f"[ERROR] File not found: {img_path}", file=sys.stderr)
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] cv2 failed to read: {img_path}", file=sys.stderr)
            continue

        # draw every box
        for bbox in bboxes:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # save result
        out_path = os.path.join(args.output, fname)
        cv2.imwrite(out_path, img)
        print(f"Wrote {out_path}")

        if args.show:
            cv2.imshow("boxes", img)
            cv2.waitKey(1)

    if args.show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
