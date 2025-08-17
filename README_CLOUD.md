# Cloud Deploy Starter (Phone-friendly)

## Deploy to Render
1) Create a GitHub repo and upload **contents of `backend/`**.
2) In Render → New → Web Service → connect repo.
3) Environment: Docker, Health check: `/health`.
4) Set env vars: `JWT_SECRET`, `JWT_EXPIRE_MIN=120`, `ADMIN_USER`, `ADMIN_PASS`, `API_KEY`.
5) Add Disk: name `storage`, mount `/app/storage`, size 2GB.
6) Deploy → visit `https://YOUR_URL/docs`.

## Use Web Uploader
Host `web_uploader/index.html` (Netlify Drop works from a phone). Enter your backend URL + API key, upload a video, view results.
