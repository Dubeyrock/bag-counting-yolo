import os
import cv2
from ultralytics import YOLO
from src.counter import BagCounter
from src.utils import draw_boxes

# -------------------- CONFIG --------------------
VIDEO_DIR = "data/videos/"
OUTPUT_DIR = "data/output/results/"
MODEL_NAME = "yolov8s.pt"          # Use a larger model (s, m, l, x)
CONF_THRESH = 0.15                  # Lower confidence
IOU_THRESH = 0.5
TRACKER = "botsort.yaml"            # More stable tracker
# Correct COCO IDs: 24 (backpack), 26 (handbag), 28 (suitcase)
BAG_CLASSES = [24, 26, 28]          
DEBUG = True                         # Print detections per frame
# ------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model (automatically downloads if not present)
model = YOLO(MODEL_NAME)

# Get list of video files
video_files = [f for f in os.listdir(VIDEO_DIR)
               if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
video_files.sort()

for video_file in video_files:
    print(f"\nProcessing {video_file} ...")
    video_path = os.path.join(VIDEO_DIR, video_file)
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(OUTPUT_DIR, f"output_{video_file}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    counter = BagCounter()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"  Frame {frame_count}/{total_frames}", end='\r')

        # Run tracking on ALL classes to see what is being detected
        results = model.track(frame,
                              persist=True,
                              conf=CONF_THRESH,
                              iou=IOU_THRESH,
                              tracker=TRACKER,
                              # classes=BAG_CLASSES, # Removed filter for debugging
                              verbose=False)

        # Log ALL detections for the first 100 frames to see what YOLO sees
        if DEBUG and frame_count <= 100:
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                counts = {}
                for cid in cls_ids:
                    name = model.names[cid]
                    counts[name] = counts.get(name, 0) + 1
                if frame_count % 20 == 0:
                    print(f"\n  Frame {frame_count} Detections: {counts}")
            elif frame_count % 50 == 0:
                print(f"\n  Frame {frame_count} - No objects detected.")

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Filter for bags only in the counter update
            bag_track_ids = [tid for tid, cid in zip(track_ids, class_ids) if cid in BAG_CLASSES]
            counter.update(bag_track_ids)

            # Draw boxes for everything
            class_names = [model.names[cid] for cid in class_ids]
            colors = {tid: (hash(tid) % 256, (hash(tid)//256) % 256, (hash(tid)//65536) % 256)
                      for tid in track_ids}
            frame = draw_boxes(frame, boxes, track_ids, class_names, colors)

        # Display current count
        cv2.putText(frame, f"Bags counted: {counter.total_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nFinished {video_file}. Total distinct bags: {counter.total_count}")

print("\nAll videos processed. Results saved in:", OUTPUT_DIR)