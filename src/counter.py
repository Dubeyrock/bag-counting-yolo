#simple counter class

class BagCounter:
    def __init__(self):
        self.seen_ids = set()
    def update(self, track_ids):
        """Add new track ID to the set."""
        for tid in track_ids:
            if tid is not None:
                self.seen_ids.add(tid)
    
    @property
    def total_count(self):
        return len(self.seen_ids)
    
    def reset(self):
        self.seen_ids.clear()
