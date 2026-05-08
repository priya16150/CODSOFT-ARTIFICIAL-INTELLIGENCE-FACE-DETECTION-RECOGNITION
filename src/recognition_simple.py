import face_recognition
import numpy as np
import os

class SimpleRecognizer:
    @staticmethod
    def get_embedding(face_img_rgb):
        """face_img_rgb is RGB numpy array."""
        encodings = face_recognition.face_encodings(face_img_rgb)
        if len(encodings) == 0:
            return None
        return encodings[0]

    @staticmethod
    def build_database(dataset_path):
        db = {}
        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person)
            if not os.path.isdir(person_path):
                continue
            embeddings = []
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = face_recognition.load_image_file(os.path.join(person_path, img_file))
                    emb = SimpleRecognizer.get_embedding(img)
                    if emb is not None:
                        embeddings.append(emb)
            if embeddings:
                db[person] = np.mean(embeddings, axis=0)
        return db

    @staticmethod
    def recognize(face_img_rgb, database, threshold=0.5):
        emb = SimpleRecognizer.get_embedding(face_img_rgb)
        if emb is None:
            return ("unknown", 1.0)
        best_name = "unknown"
        best_dist = 1.0
        for name, db_emb in database.items():
            dist = np.linalg.norm(emb - db_emb)
            if dist < best_dist and dist < threshold:
                best_dist = dist
                best_name = name
        return best_name, best_dist
