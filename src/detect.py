from ultralytics import YOLO

class Detector:
    def __init__(self, model_path):
        """Initialize the YOLO model."""
        self.model = YOLO(model_path)

    def get_model(self):
        """Return the loaded model instance."""
        return self.model
