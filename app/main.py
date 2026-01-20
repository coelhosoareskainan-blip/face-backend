from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.face_engine import recognize_face

app = FastAPI(title="Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "online"}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    result = recognize_face(await file.read())
    return result
