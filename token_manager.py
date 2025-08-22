# token_manager.py
import os, time, threading, requests

API_BASE = os.environ["API_BASE"]            # e.g. https://hockeymotion-api.onrender.com
USERNAME = os.environ["ADMIN_USERNAME"]
PASSWORD = os.environ["ADMIN_PASSWORD"]

class TokenManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._token = None
        self._exp = 0  # epoch seconds

    def _fetch(self):
        r = requests.post(f"{API_BASE}/v1/auth/token", json={
            "username": USERNAME,
            "password": PASSWORD
        }, timeout=15)
        r.raise_for_status()
        data = r.json()
        self._token = data["access_token"]
        # refresh 5 minutes early
        self._exp = time.time() + int(data.get("expires_in", 7200)) - 300

    def get(self):
        with self._lock:
            if not self._token or time.time() >= self._exp:
                self._fetch()
            return self._token

TOKEN_MANAGER = TokenManager()

def authed(method: str, path: str, **kwargs):
    """
    Example: authed("get", "/v1/calibration")
             authed("post", "/v1/upload", files={"file": ("video.mp4", f, "video/mp4")})
    """
    import requests as rq
    token = TOKEN_MANAGER.get()
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {token}"
    url = f"{API_BASE}{path}"
    resp = getattr(rq, method.lower())(url, headers=headers, timeout=60, **kwargs)
    if resp.status_code == 401:  # rare: token expired mid-call → refresh once
        TOKEN_MANAGER._fetch()
        headers["Authorization"] = f"Bearer {TOKEN_MANAGER.get()}"
        resp = getattr(rq, method.lower())(url, headers=headers, timeout=60, **kwargs)
    resp.raise_for_status()
    return resp
