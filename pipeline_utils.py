import os
import pickle

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path to your fine-tuned face detector, downloaded from the Hub.
# https://huggingface.co/uralman/yolo26l-widerface
FACE_DETECTOR_WEIGHTS = "best.pt"

# Where enrolled face embeddings are stored.
EMBEDDINGS_DB_PATH = "known_faces.pkl"

# Recognition threshold: smaller = stricter match. Distance is Euclidean
# on L2-normalized embeddings, so it roughly ranges 0 (identical) to 2 (opposite).
# 0.9-1.1 is a reasonable starting point for VGGFace2-based embeddings; tune
# this against your own data.
MATCH_THRESHOLD = 1.5


class FaceDetector:
    """Wraps the fine-tuned YOLO26l face model for detection only."""

    def __init__(self, weights_path: str = FACE_DETECTOR_WEIGHTS, conf: float = 0.5):
        self.model = YOLO(weights_path)
        self.conf = conf

    def detect(self, image):
        """
        Accepts either a file path (str) or an in-memory image (PIL.Image or
        numpy array, e.g. a webcam frame). Returns a list of
        (x1, y1, x2, y2, confidence) boxes for detected faces, in pixel coordinates.
        """
        results = self.model.predict(source=image, conf=self.conf, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        return boxes

    def crop_faces(self, image):
        """
        Accepts a file path (str), PIL.Image, or numpy array (BGR, as OpenCV
        gives you from a webcam). Returns a list of (PIL.Image crop, box)
        tuples for every detected face.
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Assume BGR (OpenCV convention) and convert to RGB for PIL.
            img = Image.fromarray(image[:, :, ::-1].copy())
        else:
            img = image.convert("RGB")

        crops = []
        for (x1, y1, x2, y2, conf) in self.detect(image):
            # Small margin around the box tends to help the embedding model.
            w, h = x2 - x1, y2 - y1
            mx, my = int(0.15 * w), int(0.15 * h)
            x1m, y1m = max(0, x1 - mx), max(0, y1 - my)
            x2m, y2m = min(img.width, x2 + mx), min(img.height, y2 + my)
            crop = img.crop((x1m, y1m, x2m, y2m))
            crops.append((crop, (x1, y1, x2, y2, conf)))
        return crops


class FaceEmbedder:
    """Wraps a pretrained face-embedding model (FaceNet / InceptionResnetV1)."""

    def __init__(self):
        # 'vggface2' pretrained weights are downloaded automatically on first use.
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    def embed(self, face_img: Image.Image) -> np.ndarray:
        """
        Takes a cropped face (PIL Image, any size) and returns a
        512-d L2-normalized embedding vector as a numpy array.
        """
        face_img = face_img.resize((160, 160))  # expected input size
        arr = np.asarray(face_img).astype(np.float32)
        arr = (arr - 127.5) / 128.0  # normalize to roughly [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = self.model(tensor)[0].cpu().numpy()

        emb = emb / np.linalg.norm(emb)
        return emb


def load_known_faces(db_path: str = EMBEDDINGS_DB_PATH) -> dict:
    """Loads {name: [embedding, ...]} from disk. Returns {} if no DB yet."""
    if not os.path.exists(db_path):
        return {}
    with open(db_path, "rb") as f:
        return pickle.load(f)


def save_known_faces(db: dict, db_path: str = EMBEDDINGS_DB_PATH):
    with open(db_path, "wb") as f:
        pickle.dump(db, f)


def identify(embedding: np.ndarray, known_faces: dict, threshold: float = MATCH_THRESHOLD):
    """
    Compares one embedding against all enrolled identities.
    Returns (best_name, best_distance) or (None, best_distance) if no match
    clears the threshold.
    """
    best_name, best_dist = None, float("inf")
    for name, ref_embeddings in known_faces.items():
        for ref in ref_embeddings:
            dist = np.linalg.norm(embedding - ref)
            if dist < best_dist:
                best_dist = dist
                best_name = name

    if best_dist <= threshold:
        return best_name, best_dist
    return None, best_dist
