import face_recognition

class FaceDetector:
    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold 

    def detect_faces(self, image):
        """image is RGB numpy array. Returns list of (x1, y1, x2, y2)."""
        face_locations = face_recognition.face_locations(image)
        return [(left, top, right, bottom) for (top, right, bottom, left) in face_locations]
