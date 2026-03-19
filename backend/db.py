"""
backend/db.py
-------------
Single shared Supabase client for all backend scripts.

supabase-py v2 quirk:
  create_client(url, service_role_key) passes the key to the auth client
  but does NOT automatically forward it as the Authorization Bearer header
  on the PostgREST client. Without the explicit .auth() call below, every
  DB write goes out as an anonymous request and RLS blocks it with 42501.

  supabase.postgrest.auth(_key) sets the Authorization header directly on
  the PostgREST client so Postgres sees service_role and bypasses RLS.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

_url = os.environ.get("SUPABASE_URL")
_key = os.environ.get("SUPABASE_SECRET_KEY")

if not _url or not _key:
    raise EnvironmentError(
        "SUPABASE_URL and SUPABASE_SECRET_KEY must be set. "
        "Copy .env.example to .env and fill in your values."
    )

supabase: Client = create_client(_url, _key)

# Explicitly set the service role key as the Bearer token on the PostgREST
# client. This is what makes Postgres recognise the request as service_role
# and bypass RLS for all write operations.
supabase.postgrest.auth(_key)