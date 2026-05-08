import os
import cv2
import numpy as np
from deepface import DeepFace

class ArcFaceRecognizer:
    def __init__(self, model_name='ArcFace', detector_backend='opencv'):
        self.model_name = model_name
        self.detector_backend = detector_backend  # we already do detection ourselves, so we'll disable it
        self.db_embeddings = {}
        self.db_path = None

    def get_embedding(self, face_img_bgr):
        """Extract embedding from a face image (BGR format)."""
        # DeepFace expects RGB; convert if needed
        img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        # enforce_detection=False because we already provide a cropped face
        embedding = DeepFace.represent(img_path=img_rgb, model_name=self.model_name,
                                       detector_backend='skip', enforce_detection=False)
        if embedding:
            return np.array(embedding[0]["embedding"])
        return None

    def build_database(self, dataset_path):
        """Pre‑compute embeddings for all known faces."""
        self.db_path = dataset_path
        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person)
            if not os.path.isdir(person_path):
                continue
            embeddings = []
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = cv2.imread(os.path.join(person_path, img_file))
                    if img is None:
                        continue
                    emb = self.get_embedding(img)
                    if emb is not None:
                        embeddings.append(emb)
            if embeddings:
                self.db_embeddings[person] = np.mean(embeddings, axis=0)
        return self.db_embeddings

    def recognize(self, face_img_bgr, database=None, threshold=0.5):
        """Compare face with database using cosine similarity."""
        if database is not None:
            self.db_embeddings = database
        emb = self.get_embedding(face_img_bgr)
        if emb is None:
            return "unknown", 1.0

        best_name = "unknown"
        best_dist = 1.0
        for name, db_emb in self.db_embeddings.items():
            # cosine distance: 1 - cosine_similarity
            cos_sim = np.dot(emb, db_emb) / (np.linalg.norm(emb) * np.linalg.norm(db_emb))
            dist = 1 - cos_sim
            if dist < best_dist and dist < threshold:
                best_dist = dist
                best_name = name
        return best_name, best_dist