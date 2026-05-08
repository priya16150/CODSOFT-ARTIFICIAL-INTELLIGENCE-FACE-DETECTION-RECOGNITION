import face_recognition

class FaceDetector:
    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold  # not used, kept for compatibility

    def detect_faces(self, image):
        """image is RGB numpy array. Returns list of (x1, y1, x2, y2)."""
        # face_recognition returns list of (top, right, bottom, left)
        face_locations = face_recognition.face_locations(image)
        # convert to (x1, y1, x2, y2)
        return [(left, top, right, bottom) for (top, right, bottom, left) in face_locations]