class Tracker:
    def __init__(self, model):
        """Initialize with a loaded YOLO model."""
        self.model = model

    def track(self, frame, conf=0.3, iou=0.5, tracker="bytetrack.yaml", classes=None):
        """Run tracking on a single frame."""
        results = self.model.track(
            frame,
            persist=True,
            conf=conf,
            iou=iou,
            tracker=tracker,
            classes=classes,
            verbose=False
        )
        return results[0]
