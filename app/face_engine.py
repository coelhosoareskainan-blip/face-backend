import numpy as np
from PIL import Image
from io import BytesIO
from deepface import DeepFace

MODEL = "ArcFace"
DETECTOR = "retinaface"
THRESHOLD = 0.35

# banco temporário em memória
known_faces = {}

def load_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(img)

def get_embedding(img):
    reps = DeepFace.represent(
        img_path=img,
        model_name=MODEL,
        detector_backend=DETECTOR,
        enforce_detection=False,
        align=True
    )
    return reps[0]["embedding"] if reps else None

def cosine_dist(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(image_bytes):
    img = load_image(image_bytes)
    emb = get_embedding(img)

    if emb is None:
        return {"status": "error", "message": "no face detected"}

    best_name = "unknown"
    best_dist = 1.0

    for name, embs in known_faces.items():
        for e in embs:
            d = cosine_dist(emb, e)
            if d < best_dist:
                best_dist = d
                best_name = name

    if best_dist <= THRESHOLD:
        return {"status": "recognized", "name": best_name, "distance": best_dist}

    return {"status": "unknown"}
