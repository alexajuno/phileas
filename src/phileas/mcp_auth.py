"""OAuth 2.1 layer for serving the Phileas MCP server over HTTP.

Why this exists
---------------
The consumer Claude app (phone) can add a self-hosted MCP server as a custom
Connector, but ONLY over OAuth 2.1 — there is no static bearer/API-key option.
So to reach Phileas from the phone the MCP server must itself be a full OAuth
Authorization Server. The official ``mcp`` SDK serves the entire OAuth surface
(DCR ``/register``, ``/authorize``, ``/token``, the two ``.well-known`` docs, and
PKCE verification) once you hand ``FastMCP`` an ``auth_server_provider``; we only
supply the provider's store/issue methods plus a real login gate.

This was de-risked end-to-end against the real phone app first (see
``~/notes/vps/03-spike-result.md``). This module is the production version of
that spike, with the four carry-over learnings applied:

1. **Persistent store** — clients/codes/tokens live in sqlite, so a daemon
   restart does not silently invalidate the connector and force a re-auth.
2. **Real single-user login** — ``/authorize`` no longer auto-approves; it
   redirects to a password page gated by ``PHILEAS_AUTH_PASSWORD``. Without this
   anyone who discovers the URL would be granted a token to all your memory.
3. **DNS-rebinding host checks disabled** — behind a tunnel the ``Host`` header
   is the public domain, not localhost; FastMCP would otherwise answer HTTP 421.
   The OAuth bearer + the tunnel are the real access control.
4. **Issuer = the public domain** — ``issuer_url``/``resource_server_url`` point
   at ``PHILEAS_PUBLIC_URL`` (the stable Tailscale Funnel domain in production).

stdio mode (local Claude Code) is unaffected: ``build_auth_components`` returns
no auth unless ``PHILEAS_MCP_TRANSPORT=http``.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from pathlib import Path

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

from phileas.config import resolve_home

# Lifetimes (seconds).
_CODE_TTL = 300  # authorization code: short-lived, single use
_PENDING_TTL = 600  # pending login: how long the password page stays valid
_ACCESS_TTL = 3600  # access token
_LOGIN_PATH = "/phileas-login"


# ----------------------------------------------------------------------------
# Persistent provider
# ----------------------------------------------------------------------------


class SqliteOAuthProvider(OAuthAuthorizationServerProvider):
    """OAuth AS/RS provider backed by sqlite.

    Pydantic models are stored as their JSON dump and rebuilt on read, so the
    schema survives SDK field changes. PKCE is verified by the SDK token handler
    before ``exchange_authorization_code`` is called — we only store and issue.
    """

    def __init__(self, db_path: Path, public_url: str, password: str) -> None:
        self._public_url = public_url.rstrip("/")
        self._password = password
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: uvicorn serves requests from a worker thread
        # pool; a single short-lived connection guarded by sqlite's own locking
        # is sufficient for one user's connector traffic.
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS oauth_clients (client_id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS oauth_codes (
                code TEXT PRIMARY KEY, data TEXT NOT NULL, expires_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS oauth_access (
                token TEXT PRIMARY KEY, data TEXT NOT NULL, expires_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS oauth_refresh (token TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS oauth_pending (
                lt TEXT PRIMARY KEY, data TEXT NOT NULL, expires_at REAL NOT NULL);
            """
        )
        self._db.commit()

    # --- tiny sqlite helpers -------------------------------------------------

    def _get(self, table: str, key_col: str, key: str) -> str | None:
        row = self._db.execute(f"SELECT data FROM {table} WHERE {key_col} = ?", (key,)).fetchone()
        return row[0] if row else None

    def _put(self, table: str, cols: tuple[str, ...], values: tuple) -> None:
        placeholders = ", ".join("?" for _ in cols)
        self._db.execute(f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})", values)
        self._db.commit()

    def _delete(self, table: str, key_col: str, key: str) -> None:
        self._db.execute(f"DELETE FROM {table} WHERE {key_col} = ?", (key,))
        self._db.commit()

    # --- clients (Dynamic Client Registration) -------------------------------

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        raw = self._get("oauth_clients", "client_id", client_id)
        return OAuthClientInformationFull.model_validate_json(raw) if raw else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._put("oauth_clients", ("client_id", "data"), (client_info.client_id, client_info.model_dump_json()))

    # --- authorize: store a pending login, redirect to the password page ------

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        lt = secrets.token_urlsafe(32)
        pending = {
            "client_id": client.client_id,
            "redirect_uri": str(params.redirect_uri),
            "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
            "state": params.state,
            "scopes": params.scopes or ["mcp"],
            "code_challenge": params.code_challenge,
            "resource": params.resource,
        }
        self._put("oauth_pending", ("lt", "data", "expires_at"), (lt, json.dumps(pending), time.time() + _PENDING_TTL))
        return f"{self._public_url}{_LOGIN_PATH}?lt={lt}"

    # --- login page completes the authorization (called from the routes) ------

    def check_password(self, candidate: str) -> bool:
        return secrets.compare_digest(candidate, self._password)

    def load_pending(self, lt: str) -> dict | None:
        raw = self._get("oauth_pending", "lt", lt)
        if not raw:
            return None
        pending = json.loads(raw)
        row = self._db.execute("SELECT expires_at FROM oauth_pending WHERE lt = ?", (lt,)).fetchone()
        if not row or row[0] < time.time():
            self._delete("oauth_pending", "lt", lt)
            return None
        return pending

    def complete_login(self, lt: str) -> str | None:
        """Consume a pending login, mint a code, return the client redirect URL."""
        pending = self.load_pending(lt)
        if pending is None:
            return None
        self._delete("oauth_pending", "lt", lt)
        code = secrets.token_urlsafe(32)
        ac = AuthorizationCode(
            code=code,
            scopes=pending["scopes"],
            expires_at=time.time() + _CODE_TTL,
            client_id=pending["client_id"],
            code_challenge=pending["code_challenge"],
            redirect_uri=AnyHttpUrl(pending["redirect_uri"]),
            redirect_uri_provided_explicitly=pending["redirect_uri_provided_explicitly"],
            resource=pending["resource"],
        )
        self._put("oauth_codes", ("code", "data", "expires_at"), (code, ac.model_dump_json(), ac.expires_at))
        return construct_redirect_uri(pending["redirect_uri"], code=code, state=pending["state"])

    # --- token endpoint ------------------------------------------------------

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        raw = self._get("oauth_codes", "code", authorization_code)
        if not raw:
            return None
        ac = AuthorizationCode.model_validate_json(raw)
        if ac.expires_at < time.time():
            self._delete("oauth_codes", "code", authorization_code)
            return None
        return ac

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # SDK already verified PKCE (code_verifier vs stored code_challenge).
        self._delete("oauth_codes", "code", authorization_code.code)
        return self._issue_tokens(client.client_id, authorization_code.scopes, authorization_code.resource)

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        raw = self._get("oauth_refresh", "token", refresh_token)
        return RefreshToken.model_validate_json(raw) if raw else None

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]
    ) -> OAuthToken:
        self._delete("oauth_refresh", "token", refresh_token.token)
        return self._issue_tokens(client.client_id, scopes or refresh_token.scopes, None)

    def _issue_tokens(self, client_id: str, scopes: list[str], resource: str | None) -> OAuthToken:
        at, rt = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        expires_at = int(time.time()) + _ACCESS_TTL
        access = AccessToken(token=at, client_id=client_id, scopes=scopes, expires_at=expires_at, resource=resource)
        refresh = RefreshToken(token=rt, client_id=client_id, scopes=scopes, expires_at=None)
        self._put("oauth_access", ("token", "data", "expires_at"), (at, access.model_dump_json(), expires_at))
        self._put("oauth_refresh", ("token", "data"), (rt, refresh.model_dump_json()))
        return OAuthToken(
            access_token=at, token_type="Bearer", expires_in=_ACCESS_TTL, refresh_token=rt, scope=" ".join(scopes)
        )

    # --- resource server: verify a presented bearer --------------------------

    async def load_access_token(self, token: str) -> AccessToken | None:
        raw = self._get("oauth_access", "token", token)
        if not raw:
            return None
        access = AccessToken.model_validate_json(raw)
        if access.expires_at is not None and access.expires_at < time.time():
            self._delete("oauth_access", "token", token)
            return None
        return access

    async def revoke_token(self, token) -> None:
        tok = getattr(token, "token", None)
        if tok:
            self._delete("oauth_access", "token", tok)
            self._delete("oauth_refresh", "token", tok)


# ----------------------------------------------------------------------------
# Login page (the single-user gate that replaces auto-approve)
# ----------------------------------------------------------------------------


def _login_page(lt: str, error: str = "") -> str:
    err_html = f'<p class="err">{error}</p>' if error else ""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phileas — sign in</title>
<style>
  body {{ background:#0f1115; color:#e6e6e6; font-family:system-ui,sans-serif;
         display:flex; min-height:100vh; align-items:center; justify-content:center; margin:0 }}
  form {{ background:#1a1d24; padding:2rem; border-radius:12px; width:min(90vw,340px);
          box-shadow:0 8px 30px rgba(0,0,0,.4) }}
  h1 {{ font-size:1.2rem; margin:0 0 .25rem }} p.sub {{ color:#8a8f98; margin:.25rem 0 1.25rem; font-size:.85rem }}
  input {{ width:100%; box-sizing:border-box; padding:.7rem; border-radius:8px; border:1px solid #2c313c;
           background:#0f1115; color:#e6e6e6; font-size:1rem }}
  button {{ width:100%; margin-top:1rem; padding:.7rem; border:0; border-radius:8px; background:#5b8cff;
            color:#fff; font-size:1rem; cursor:pointer }}
  p.err {{ color:#ff6b6b; font-size:.85rem; margin:0 0 .75rem }}
</style></head>
<body>
  <form method="post" action="{_LOGIN_PATH}">
    <h1>Phileas</h1>
    <p class="sub">Sign in to connect your memory.</p>
    {err_html}
    <input type="hidden" name="lt" value="{lt}">
    <input type="password" name="password" placeholder="Password" autofocus autocomplete="current-password">
    <button type="submit">Sign in</button>
  </form>
</body></html>"""


def register_login_routes(mcp, provider: SqliteOAuthProvider) -> None:
    """Attach the GET/POST login routes to the FastMCP app."""

    @mcp.custom_route(_LOGIN_PATH, methods=["GET"])
    async def login_form(request: Request) -> Response:
        lt = request.query_params.get("lt", "")
        if not lt or provider.load_pending(lt) is None:
            return HTMLResponse("<h1>Link expired</h1><p>Restart the connection from Claude.</p>", status_code=400)
        return HTMLResponse(_login_page(lt))

    @mcp.custom_route(_LOGIN_PATH, methods=["POST"])
    async def login_submit(request: Request) -> Response:
        form = await request.form()
        lt = str(form.get("lt", ""))
        password = str(form.get("password", ""))
        if not lt or provider.load_pending(lt) is None:
            return HTMLResponse("<h1>Link expired</h1><p>Restart the connection from Claude.</p>", status_code=400)
        if not provider.check_password(password):
            return HTMLResponse(_login_page(lt, "Incorrect password."), status_code=401)
        redirect = provider.complete_login(lt)
        if redirect is None:  # raced with expiry between the two checks
            return HTMLResponse("<h1>Link expired</h1><p>Restart the connection from Claude.</p>", status_code=400)
        return RedirectResponse(url=redirect, status_code=302)


# ----------------------------------------------------------------------------
# Wiring entry point (called by mcp_server.py)
# ----------------------------------------------------------------------------


def build_auth_components() -> tuple[dict, SqliteOAuthProvider | None]:
    """Return (FastMCP kwargs, provider) for HTTP mode, or ({}, None) for stdio.

    HTTP mode is opt-in via ``PHILEAS_MCP_TRANSPORT=http``. Env knobs:
      PHILEAS_PUBLIC_URL    (required) public https base, e.g. the Funnel domain
      PHILEAS_AUTH_PASSWORD (required) single-user login password
      PHILEAS_MCP_HOST      bind host (default 127.0.0.1 — Funnel fronts it)
      PHILEAS_MCP_PORT      bind port (default 8848)
      PHILEAS_OAUTH_DB      sqlite path (default <profile home>/oauth.db)
    """
    if os.environ.get("PHILEAS_MCP_TRANSPORT", "stdio").lower() != "http":
        return {}, None

    public_url = os.environ.get("PHILEAS_PUBLIC_URL", "").rstrip("/")
    if not public_url:
        raise RuntimeError(
            "PHILEAS_MCP_TRANSPORT=http requires PHILEAS_PUBLIC_URL (e.g. https://phileas.<tailnet>.ts.net)"
        )
    password = os.environ.get("PHILEAS_AUTH_PASSWORD")
    if not password:
        raise RuntimeError("PHILEAS_MCP_TRANSPORT=http requires PHILEAS_AUTH_PASSWORD (the single-user login password)")

    host = os.environ.get("PHILEAS_MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("PHILEAS_MCP_PORT", "8848"))
    db_path = Path(os.environ.get("PHILEAS_OAUTH_DB", str(resolve_home() / "oauth.db")))

    provider = SqliteOAuthProvider(db_path, public_url, password)
    kwargs = {
        "host": host,
        "port": port,
        # Behind a tunnel the Host header is the public domain, not localhost;
        # the OAuth bearer + the tunnel are the real access control.
        "transport_security": TransportSecuritySettings(enable_dns_rebinding_protection=False),
        "auth": AuthSettings(
            issuer_url=AnyHttpUrl(public_url),
            resource_server_url=AnyHttpUrl(public_url),
            required_scopes=None,  # single user: any valid token passes
            client_registration_options=ClientRegistrationOptions(
                enabled=True, valid_scopes=["mcp"], default_scopes=["mcp"]
            ),
        ),
        "auth_server_provider": provider,
    }
    return kwargs, provider
