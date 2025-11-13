import warnings
import cv2
import numpy as np
import torch
import time
import argparse

from ultralytics import YOLO
from nets import nn
from utils import util

warnings.filterwarnings("ignore")


def draw_line(image, x1, y1, x2, y2, index):
    w = 10
    h = 10
    color = (200, 0, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # Top left
    cv2.line(image, (x1, y1), (x1 + w, y1), color, 3)
    cv2.line(image, (x1, y1), (x1, y1 + h), color, 3)

    # Top right
    cv2.line(image, (x2, y1), (x2 - w, y1), color, 3)
    cv2.line(image, (x2, y1), (x2, y1 + h), color, 3)

    # Bottom right
    cv2.line(image, (x2, y2), (x2 - w, y2), color, 3)
    cv2.line(image, (x2, y2), (x2, y2 - h), color, 3)

    # Bottom left
    cv2.line(image, (x1, y2), (x1 + w, y2), color, 3)
    cv2.line(image, (x1, y2), (x1, y2 - h), color, 3)

    cv2.putText(image, f"ID:{index}", (x1, y1 - 2),
                0, 0.7, (0, 250, 0), 2)


def main(input_video_path, output_video_path, weights_path):

    print(f"\n📦 Loading YOLOv8 model: {weights_path}")
    model = YOLO(weights_path)

    reader = cv2.VideoCapture(input_video_path)
    if not reader.isOpened():
        print(f"❌ Error opening video: {input_video_path}")
        return

    source_fps = int(reader.get(cv2.CAP_PROP_FPS))
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, source_fps, (width, height))

    print(f"🎥 Processing: {input_video_path}")
    print(f"💾 Writing output to: {output_video_path}")

    bytetrack = nn.BYTETracker(source_fps)

    frame_count = 0
    total_time = 0
    warmup_frames = 20

    while True:
        ok, frame = reader.read()
        if not ok:
            break

        t0 = time.time()
        frame_count += 1

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # CRUCIAL: Give the tracker the frame for ReID crops
        bytetrack.last_frame = frame
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ----- YOLOv8 inference -----
        results = model.predict(frame, verbose=False, half=True)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = results.boxes.conf.cpu().numpy().astype(np.float32)
            classes = results.boxes.cls.cpu().numpy().astype(np.int32)

            # Track with ReID-enhanced BYTETracker
            outputs = bytetrack.update(boxes, confs, classes)

            if len(outputs) > 0:
                for det in outputs:
                    x1, y1, x2, y2 = map(int, det[:4])
                    track_id = int(det[4])
                    cls_id = int(det[6])

                    if cls_id == 0:   # person class only
                        draw_line(frame, x1, y1, x2, y2, track_id)

        t1 = time.time()

        if frame_count > warmup_frames:
            total_time += (t1 - t0)

        if frame_count % 50 == 0:
            avg_fps = (frame_count - warmup_frames) / max(1e-6, total_time)
            print(f"  → Frame {frame_count} | Avg FPS: {avg_fps:.2f}")

        writer.write(frame)

    # ---- Final summary ----
    reader.release()
    writer.release()

    useful_frames = max(0, frame_count - warmup_frames)
    if useful_frames > 0:
        final_fps = useful_frames / total_time
        print(f"\n✅ Done! Total frames: {frame_count}")
        print(f"⚡ Average Effective FPS: {final_fps:.2f}")
    else:
        print("\n⚠️ No frames processed after warmup.")

    print(f"🎉 Output saved to: {output_video_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_video", type=str, default="./output.mp4")
    parser.add_argument("--weights", type=str, default="yolov8s.pt")
    args = parser.parse_args()

    main(args.input_video, args.output_video, args.weights)
