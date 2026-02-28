# Bag Counting Project

This project implements a modular computer vision system for counting bags in video clips using YOLOv8 and ByteTrack.

## Project Structure

```
bag_counting_project/
│
├── data/
│   ├── videos/         # Input video files (Scenario1.mp4, etc.)
│   └── output/
│       └── results/    # Processed output videos with counts
│
├── models/
│   └── yolov8n.pt      # YOLOv8 nano model
│
├── src/
│   ├── detect.py       # Model loading and detector abstraction
│   ├── track.py        # Tracking logic
│   ├── counter.py      # Unique ID counting logic
│   └── utils.py        # Visualization helpers
│
├── main.py             # Application entry point
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place input videos in `data/videos/`.

3. Run the application:
   ```bash
   python main.py
   ```

## Key Features
- **Modular Design**: Separated concerns for detection, tracking, and counting.
- **Unique Counting**: Uses `seen_ids` to ensure each bag is counted only once per video.
- **Visual Feedback**: Bounding boxes and live total count displayed on output videos.


   


