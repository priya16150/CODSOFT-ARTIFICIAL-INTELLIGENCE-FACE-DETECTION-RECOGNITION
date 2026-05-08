import urllib.request
import os

os.makedirs("models", exist_ok=True)

urls = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/3.4.0/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
}

for filename, url in urls.items():
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, f"models/{filename}")
    print("Done")