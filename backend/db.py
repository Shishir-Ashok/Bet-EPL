"""
backend/db.py
-------------
Single shared Supabase client for all backend scripts.

Every pipeline script imports `supabase` from here — one place to
manage the connection, one place to update if Supabase's SDK changes.

The SECRET key is used throughout the backend because pipeline jobs
need to write to the DB. The publishable key is only for the frontend.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

_url  = os.environ.get("SUPABASE_URL")
_key  = os.environ.get("SUPABASE_SECRET_KEY")

if not _url or not _key:
    raise EnvironmentError(
        "SUPABASE_URL and SUPABASE_SECRET_KEY must be set. "
        "Copy .env.example to .env and fill in your values."
    )

supabase: Client = create_client(_url, _key)