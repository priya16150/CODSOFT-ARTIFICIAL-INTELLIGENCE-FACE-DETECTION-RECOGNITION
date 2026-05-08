import cv2
import argparse
import os
import sys

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection import FaceDetector
from src.recognition_arcface import ArcFaceRecognizer
from src.recognition_simple import SimpleRecognizer
from src.utils import draw_boxes

def process_image(image_path, detector, recognizer, database, threshold=0.6):
    """Process a single image file."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    names = []
    distances = []
    for (x1, y1, x2, y2) in faces:
        face_roi = img_rgb[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue
        # Recognizer expects BGR for ArcFace (DeepFace), but simple recognizer expects RGB.
        # We'll convert inside each recognizer.
        face_bgr = cv2.cvtColor(face_roi, cv2.COLOR_RGB2BGR)
        name, dist = recognizer.recognize(face_bgr, database, threshold)
        names.append(name)
        distances.append(dist)

    output = draw_boxes(img_rgb, faces, names, distances)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imshow("Face Recognition", output_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(source, detector, recognizer, database, threshold=0.6):
    """Process webcam (source=0) or video file."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return

    print("Press 'q' to quit, 's' to save current frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)
        names = []
        distances = []
        for (x1, y1, x2, y2) in faces:
            face_roi = img_rgb[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
            face_bgr = cv2.cvtColor(face_roi, cv2.COLOR_RGB2BGR)
            name, dist = recognizer.recognize(face_bgr, database, threshold)
            names.append(name)
            distances.append(dist)
        output = draw_boxes(img_rgb, faces, names, distances)
        cv2.imshow("Live Recognition", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("saved_frame.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            print("Frame saved as saved_frame.jpg")
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Face Detection & Recognition")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to image/video file, or '0' for webcam")
    parser.add_argument("--dataset", type=str, default="dataset",
                        help="Folder containing known faces (subfolders per person)")
    parser.add_argument("--recognizer", choices=["arcface",  "simple"], default="arcface",
                        help="Recognition backend: arcface (DeepFace) or simple (dlib)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Recognition threshold (lower = stricter)")
    args = parser.parse_args()

    # Initialize face detector (OpenCV DNN)
    detector = FaceDetector()

    # Initialize recognizer
    if args.recognizer == "arcface":
        recognizer = ArcFaceRecognizer()
    else:
        recognizer = SimpleRecognizer()

    # Build database if dataset folder exists
    database = {}
    if os.path.exists(args.dataset) and os.path.isdir(args.dataset):
        print(f"Building database from {args.dataset} ...")
        database = recognizer.build_database(args.dataset)
        print(f"Loaded {len(database)} persons.")
    else:
        print(f"Dataset folder '{args.dataset}' not found. Recognition will return 'unknown'.")

    # Determine input type
    if args.input.isdigit():
        source = int(args.input)   # webcam
        process_video(source, detector, recognizer, database, args.threshold)
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(args.input, detector, recognizer, database, args.threshold)
    else:
        process_image(args.input, detector, recognizer, database, args.threshold)

if __name__ == "__main__":
    main()