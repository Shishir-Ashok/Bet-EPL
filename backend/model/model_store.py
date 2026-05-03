"""
backend/model/model_store.py
------------------------------
Handles model file persistence via Supabase Storage.

Render's filesystem is ephemeral — files in /checkpoints are wiped on
every restart. This module uploads model files to Supabase Storage after
training and downloads them on demand when load_model() can't find the
local file.

Bucket: set SUPABASE_STORAGE_BUCKET in your environment (e.g. "models").
Create it in Supabase dashboard → Storage → New bucket → name it, set to private.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase

BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "models")
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))


def upload(local_path: str) -> str:
    """
    Uploads a model file to Supabase Storage.
    Returns the storage path (used as storage_path in model_versions).
    Overwrites if the file already exists in the bucket.
    """
    filename     = os.path.basename(local_path)
    storage_path = f"checkpoints/{filename}"

    with open(local_path, "rb") as f:
        data = f.read()

    try:
        # Try update first (file exists), fall back to upload (new file)
        supabase.storage.from_(BUCKET).update(storage_path, data)
    except Exception:
        supabase.storage.from_(BUCKET).upload(storage_path, data)

    print(f"  ✓ Uploaded to Storage: {storage_path}")
    return storage_path


def download(storage_path: str) -> str:
    """
    Downloads a model file from Supabase Storage to local checkpoints dir.
    Returns the local path. Safe to call even if file already exists locally.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    filename   = storage_path.split("/")[-1]
    local_path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(local_path):
        return local_path   # already present — no download needed

    print(f"  Downloading model from Storage: {storage_path}...")
    data = supabase.storage.from_(BUCKET).download(storage_path)

    with open(local_path, "wb") as f:
        f.write(data)

    print(f"  ✓ Downloaded to: {local_path}")
    return local_path


def ensure_local(storage_path: str, local_path: str) -> str:
    """
    Returns a valid local path for a model file, downloading from Storage
    if the file isn't present on disk. Called by load_model() and load_active().
    """
    if os.path.exists(local_path):
        return local_path
    return download(storage_path)