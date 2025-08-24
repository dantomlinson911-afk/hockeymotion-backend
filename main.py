
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from pathlib import Path
import shutil, uuid, json, os
import cv2, numpy as np
import jwt
from datetime import datetime, timedelta
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI(title="HockeyShot Backend (Cloud Minimal)")

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
PROC_DIR = BASE_DIR / "storage" / "processed"
CALIB_PATH = BASE_DIR / "storage" / "calibration.json"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ---- Auth (API key or JWT) ----
JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", "120"))
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin123")
API_KEY = os.getenv("API_KEY", "demo_api_key")

bearer = HTTPBearer(auto_error=False)

def create_jwt(sub: str):
    now = datetime.utcnow()
    payload = {"sub": sub, "iat": int(now.timestamp()), "exp": int((now + timedelta(minutes=JWT_EXPIRE_MIN)).timestamp())}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_jwt(token: str) -> bool:
    try:
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return True
    except Exception:
        return False

def require_auth(authorization: str | None = Header(None), x_api_key: str | None = Header(None, alias="x-api-key"), credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if x_api_key and x_api_key == API_KEY:
        return True
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    elif credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    if token and verify_jwt(token):
        return True
    raise HTTPException(status_code=401, detail="Unauthorized")

class AuthIn(BaseModel):
    username: str
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRE_MIN * 60

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/auth/token", response_model=TokenOut)
def auth_token(body: AuthIn):
    if body.username == ADMIN_USER and body.password == ADMIN_PASS:
        tok = create_jwt(sub=body.username)
        return TokenOut(access_token=tok)
    raise HTTPException(status_code=401, detail="Invalid credentials")

# ---- Calibration ----
class CalibrationIn(BaseModel):
    net_width_meters: float
    net_width_pixels_hint: float | None = None
    roi: List[float] = [0.2, 0.2, 0.8, 0.8]

@app.get("/v1/calibration", dependencies=[Depends(require_auth)])
def get_calibration():
    if CALIB_PATH.exists():
        try:
            return JSONResponse(json.loads(CALIB_PATH.read_text()))
        except Exception:
            pass
    return {"net_width_meters": 1.83, "net_width_pixels_hint": None, "roi": [0.2, 0.2, 0.8, 0.8]}

@app.post("/v1/calibration", dependencies=[Depends(require_auth)])
def set_calibration(body: CalibrationIn):
    if len(body.roi) != 4 or not all(0.0 <= v <= 1.0 for v in body.roi):
        raise HTTPException(status_code=400, detail="roi must be [x1,y1,x2,y2] within 0..1")
    payload = {"net_width_meters": float(body.net_width_meters), "net_width_pixels_hint": float(body.net_width_pixels_hint) if body.net_width_pixels_hint else None, "roi": [float(v) for v in body.roi]}
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(json.dumps(payload, indent=2))
    return {"ok": True, "calibration": payload}

# ---- Upload / process (dummy but functional overlay) ----
class UploadResponse(BaseModel):
    job_id: str

JOBS = {}  # in-memory status

@app.post("/v1/upload", response_model=UploadResponse, dependencies=[Depends(require_auth)])
async def upload(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    out_path = UPLOAD_DIR / f"{job_id}.mp4"
    with out_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    JOBS[job_id] = {"status": "queued"}
    # Simple processing: generate overlay with dummy text and copy result json
    try:
        result, overlay = process_clip(str(out_path), str(PROC_DIR / f"{job_id}.mp4"))
        (PROC_DIR / f"{job_id}.json").write_text(json.dumps(result, indent=2))
        JOBS[job_id] = {"status": "done"}
    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e)}
    return UploadResponse(job_id=job_id)

@app.get("/v1/status/{job_id}", dependencies=[Depends(require_auth)])
def status(job_id: str):
    st = JOBS.get(job_id, {"status": "unknown"})
    return st

@app.get("/v1/result/{job_id}", dependencies=[Depends(require_auth)])
def result(job_id: str):
    p = PROC_DIR / f"{job_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not ready")
    return JSONResponse(json.loads(p.read_text()))

@app.get("/v1/overlay/{job_id}.mp4", dependencies=[Depends(require_auth)])
def overlay(job_id: str):
    p = PROC_DIR / f"{job_id}.mp4"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not ready")
    return FileResponse(p, media_type="video/mp4")

# ---- Minimal processing (no ONNX in this minimal build, but keeps same API) ----
def process_clip(video_path: str, overlay_out: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    max_frames = 300
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Fake shot type and speed to prove flow works (replace later with real pipeline)
    shot_type = "wrist"
    speed_kmh = 90.0

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(overlay_out, fourcc, fps, (w, h))
    for f in frames:
        cv2.rectangle(f, (15, 15), (600, 120), (0,0,0), -1)
        cv2.putText(f, f"Shot: {shot_type}", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(f, f"Speed: {speed_kmh:.1f} km/h", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA)
        vw.write(f)
    vw.release()

    return {
        "shot_type": shot_type,
        "shot_meta": {"model": "demo"},
        "speed_kmh": speed_kmh,
        "speed_meta": {"calibration": "demo"},
        "frames_analyzed": len(frames),
        "fps": fps,
    }, overlay_out

# --- Optional: Self-test endpoint ---
from fastapi import Depends
import requests
import os

API_BASE = os.getenv("API_BASE", "https://hockeymotion-api.onrender.com")

@app.get("/admin/selftest")
def selftest():
    """
    Calls GET /v1/calibration using admin credentials.
    Useful for checking that tokens and auth flow work automatically.
    """
    try:
        # Login to get a token
        r = requests.post(
            f"{API_BASE}/v1/auth/token",
            json={"username": os.getenv("ADMIN_USER"), "password": os.getenv("ADMIN_PASS")}
        )
        r.raise_for_status()
        token = r.json()["access_token"]

        # Call a protected route
        r2 = requests.get(
            f"{API_BASE}/v1/calibration",
            headers={"Authorization": f"Bearer {token}"}
        )
        r2.raise_for_status()

        return {"ok": True, "calibration": r2.json()}
    except Exception as e:
        return {"ok": False, "error": str(e)}
