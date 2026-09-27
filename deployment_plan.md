# Deployment Plan: Canary (Vercel + Render)

## Architecture After Deployment

```
User Browser
     │
     ├──► Vercel (Frontend — React/Vite)       → Free, global CDN, instant
     │         hosts: dist/ of frontend/
     │
     └──► Render (Backend — Flask)             → Free tier (512MB RAM, 0.1 CPU)
              hosts: backend/ + sample_data/
```

---

## Critical Blockers to Fix First

> [!CAUTION]
> **You have two hard blockers that will crash any free deployment if you don't fix them before deploying.**

### Blocker 1: `torch`, `torchaudio`, `torchvision` in `requirements.txt`
- These three packages are **1.2+ GB combined**.
- Render's free tier has a **512 MB RAM limit** — it will OOM-kill during pip install.
- **Fix:** PyTorch is not actually used in `app.py`. The backend uses only NumPy/librosa/scipy. Remove the torch packages from a `backend/requirements.txt` (separate from the root one).

### Blocker 2: `sample_data/` is 444 MB
- This cannot be committed to GitHub (Git has a 100 MB file size limit, and GitHub has a 2 GB repo limit).
- It cannot sit in the Render repo either unless you use Render Disk (not on free tier).
- **Fix:** The data files need to be uploaded to a free object store (Cloudflare R2 or Backblaze B2) and downloaded at runtime on the first request. This is detailed in Step 4 below.

---

## Step-by-Step Deployment

### Step 1 — Create a Backend-Specific `requirements.txt`

**Create the file** `backend/requirements.txt` with only what `app.py` actually imports:

```txt
flask
flask-cors
numpy
scipy
pandas
librosa
soundfile
scikit-learn
matplotlib
gunicorn
```

> [!NOTE]
> `gunicorn` is required by Render to serve Flask in production. Without it, the deploy will fail.

### Step 2 — Create `backend/render.yaml` (Render config)

Create the file `backend/render.yaml`:

```yaml
services:
  - type: web
    name: canary-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn --bind 0.0.0.0:10000 app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

### Step 3 — Fix the Flask `app.py` for Production

Make two changes to `backend/app.py`:

**Change 1:** Make the port read from an environment variable (Render injects `PORT`):

```python
# Change the last 3 lines of app.py from:
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)

# To:
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
```

**Change 2:** Make `SAMPLE_DATA` path resolve from an environment variable for flexibility:

```python
# Change line 47 from:
SAMPLE_DATA = REPO_ROOT / "sample_data"

# To:
SAMPLE_DATA = Path(os.environ.get("SAMPLE_DATA_PATH", str(REPO_ROOT / "sample_data")))
```

### Step 4 — Handle the `sample_data/` Files

You have three options. **Option A is the simplest for a hackathon.**

#### Option A (Recommended for Hackathon): Commit Only Showcase Files

The `sample_data/` folder is 444 MB uncompressed. If your **showcase files** (the specific NPZ/CSV/WAV files referenced in the registry) are small individually, you can keep ONLY those files committed and delete everything else.

Run this to check sizes of just the showcase files:
```powershell
# In your project root
Get-Item "sample_data\cwru\*\*.npz" -Recurse | Select-Object FullName, Length
Get-Item "sample_data\subf\Dataset\*\*.csv" | Select-Object FullName, Length
```

If the showcase files total under 100 MB (likely — they're single samples), do this:
1. Add `sample_data/` to `.gitignore` EXCEPT for the showcase paths.
2. Update `.gitignore`:
   ```gitignore
   # Ignore all sample data except showcase files
   sample_data/**
   !sample_data/cwru/
   !sample_data/cwru/1750 RPM/
   !sample_data/cwru/1750 RPM/1750_Normal.npz
   !sample_data/cwru/1730 RPM/
   !sample_data/cwru/1730 RPM/1730_IR_14_DE48.npz
   # ... repeat for all other 10 showcase files
   ```

> [!IMPORTANT]
> With Git, to allow a specific nested file, you must un-ignore EVERY parent directory too, not just the file.

#### Option B: Use Git LFS (Git Large File Storage)
GitHub and Render support Git LFS for files up to 2 GB. Enable with:
```bash
git lfs install
git lfs track "sample_data/**/*.npz"
git lfs track "sample_data/**/*.csv"
git lfs track "sample_data/**/*.wav"
git lfs track "sample_data/**/*.mat"
git add .gitattributes
```

> [!WARNING]
> Git LFS has a 1 GB free bandwidth limit on GitHub. 444 MB × build attempts = can hit limit fast.

#### Option C: Cloudflare R2 Object Storage (Most Robust)
Upload the showcase files to Cloudflare R2 (free 10 GB storage), then add startup code to `app.py` to download them if not present. Skip this unless Options A and B fail.

### Step 5 — Fix CORS for the Production Frontend URL

In `backend/app.py`, replace the broad `CORS(app)` with a specific origin:

```python
from flask_cors import CORS

# Replace:
CORS(app)

# With:
CORS(app, origins=[
    "http://localhost:5174",        # local dev
    "https://your-app.vercel.app",  # production (fill in after Step 8)
])
```

> [!NOTE]
> You'll fill in the actual Vercel URL after Step 8. You can deploy with `CORS(app)` first, then tighten it.

### Step 6 — Fix the Frontend API URL for Production

In `frontend/src/App.tsx`, the API base is currently hardcoded:
```tsx
const API_BASE = 'http://127.0.0.1:8000';
```

Change it to read from a Vite environment variable:
```tsx
const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
```

Then create `frontend/.env.production`:
```env
VITE_API_URL=https://canary-backend.onrender.com
```

(You'll fill in the real Render URL after Step 7.)

### Step 7 — Deploy the Backend to Render

1. Go to [render.com](https://render.com) and create a free account.
2. Click **"New +"** → **"Web Service"**.
3. Connect your GitHub account and select the `Granica-Submission` repo.
4. Set these fields:
   - **Name:** `canary-backend`
   - **Region:** `Oregon (US West)` (cheapest free tier)
   - **Branch:** `main`
   - **Root Directory:** `backend`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --bind 0.0.0.0:$PORT app:app`
5. Click **"Create Web Service"**.
6. Wait ~3 minutes for the first build.
7. Copy the URL Render assigns: `https://canary-backend.onrender.com`.

> [!IMPORTANT]
> On the free tier, Render **spins down the service after 15 minutes of inactivity**. The first request after spin-down takes 30–60 seconds. For a hackathon demo, open the backend URL in a browser tab before presenting to pre-warm it.

### Step 8 — Deploy the Frontend to Vercel

1. Update `frontend/.env.production` with the Render URL from Step 7.
2. Go to [vercel.com](https://vercel.com) and create a free account.
3. Click **"Add New"** → **"Project"**.
4. Import the `Granica-Submission` GitHub repo.
5. Set these fields:
   - **Framework Preset:** `Vite`
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
6. Under **"Environment Variables"**, add:
   - Key: `VITE_API_URL`
   - Value: `https://canary-backend.onrender.com` (your Render URL)
7. Click **"Deploy"**.
8. Vercel builds and gives you a URL like `https://canary-submission.vercel.app`.

### Step 9 — Tighten CORS (After Both URLs Are Known)

Go back to `backend/app.py` and replace the open CORS with your specific Vercel URL:
```python
CORS(app, origins=["https://canary-submission.vercel.app", "http://localhost:5174"])
```

Commit and push — Render will auto-redeploy.

### Step 10 — Verify the Full Flow

Open `https://canary-submission.vercel.app` and check:
- [ ] Cards load (calls `/api/datasets`)
- [ ] Hovering cards shows Normal/Anomaly buttons
- [ ] Clicking a button streams logs in the terminal
- [ ] Results panel slides in with spectrogram images

---

## Deployment Checklist Summary

| # | Task | File |
|---|------|------|
| 1 | Create `backend/requirements.txt` (no torch) | `backend/requirements.txt` |
| 2 | Create `backend/render.yaml` | `backend/render.yaml` |
| 3 | Fix Flask port + host for production | `backend/app.py` |
| 4 | Commit only showcase `sample_data/` files | `.gitignore` |
| 5 | Fix CORS origins | `backend/app.py` |
| 6 | Use `VITE_API_URL` env var in frontend | `frontend/src/App.tsx` |
| 7 | Create `frontend/.env.production` | `frontend/.env.production` |
| 8 | Deploy backend to Render | render.com |
| 9 | Deploy frontend to Vercel | vercel.com |
| 10 | Tighten CORS with final Vercel URL | `backend/app.py` |

*Click **Proceed** to have me make all the code changes automatically (Steps 1–7).*
