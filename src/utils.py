import cv2

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def draw_boxes(image, faces, names=None, distances=None):
    """Draw rectangles and labels on image (image is RGB)."""
    img_copy = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(faces):
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if names and i < len(names):
            label = f"{names[i]} ({distances[i]:.2f})" if distances else names[i]
            cv2.putText(img_copy, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return img_copy