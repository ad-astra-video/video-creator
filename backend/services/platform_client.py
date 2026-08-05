"""Remote "platform" credits API client service.

The desktop talks to a remote platform worker that owns all secrets and the
credit ledger. This module is the narrow HTTP boundary for that API:

- ``POST {base}/provision``  body {externalUserId} -> 200 {apiKey, externalUserId}
- ``GET  {base}/balance``    header ``Authorization: Bearer *** -> {hasAccess, ...}
- ``POST {base}/checkout``   header ``Authorization: Bearer *** body {tier} -> {url}
- ``POST {base}/link-email`` header ``Authorization: Bearer *** body {email} -> sets recovery email
- ``POST {base}/recover/request`` body {email} -> emails a one-time code
- ``POST {base}/recover/confirm`` body {email, code} -> 200 {apiKey} (rotates a lost key)

``base_url`` is a per-user setting, so it is passed per-call rather than baked into
the client (the client stays a stateless HTTP boundary, mirroring how ``HTTPClient``
is injected once and URLs come from call sites).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, cast

from services.http_client.http_client import HTTPClient, HttpResponseLike

logger = logging.getLogger(__name__)

_PROVISION_TIMEOUT_S = 10


class PlatformError(RuntimeError):
    """Raised by the platform client on a non-2xx response or a malformed body."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class PlatformClient(Protocol):
    """Protocol for the platform credits API (real + fake implementations)."""

    def provision(self, *, base_url: str, external_user_id: str) -> str:
        """Provision a per-user API key. Returns the key. Idempotent on the server."""
        ...

    def get_balance(self, *, base_url: str, api_key: str) -> dict[str, Any]:
        """Return the current balance/access payload for ``api_key``."""
        ...

    def create_checkout(self, *, base_url: str, api_key: str, tier_credits_cents: int) -> str:
        """Create a checkout session for the given top-up tier. Returns the hosted URL."""
        ...

    def link_email(self, *, base_url: str, api_key: str, email: str) -> None:
        """Associate ``email`` with ``api_key`` as the recovery email."""
        ...

    def recover_request(self, *, base_url: str, email: str) -> None:
        """Ask the platform to email a one-time recovery code to ``email``."""
        ...

    def recover_confirm(self, *, base_url: str, email: str, code: str) -> str:
        """Confirm a recovery code, rotating the lost key. Returns the NEW key."""
        ...


class HttpPlatformClient:
    """Real HTTP implementation using the shared ``HTTPClient`` service."""

    def __init__(self, http: HTTPClient, *, timeout: int = _PROVISION_TIMEOUT_S) -> None:
        self._http = http
        self._timeout = timeout

    def _url(self, base_url: str, path: str) -> str:
        return base_url.rstrip("/") + path

    def _require_json(self, resp: HttpResponseLike, endpoint: str) -> dict[str, Any]:
        if resp.status_code < 200 or resp.status_code >= 300:
            text = resp.text[:200] if resp.text else "unknown error"
            raise PlatformError(
                resp.status_code,
                f"platform {endpoint} request failed (HTTP {resp.status_code}): {text}",
            )
        try:
            payload = resp.json()
        except json.JSONDecodeError as exc:
            raise PlatformError(resp.status_code, f"platform {endpoint} returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise PlatformError(resp.status_code, f"platform {endpoint} returned an unexpected shape")
        return cast(dict[str, Any], payload)

    def provision(self, *, base_url: str, external_user_id: str) -> str:
        resp = self._http.post(
            self._url(base_url, "/provision"),
            json_payload={"externalUserId": external_user_id},
            timeout=self._timeout,
        )
        payload = self._require_json(resp, "provision")
        key = payload.get("apiKey")
        if not isinstance(key, str) or not key:
            raise PlatformError(resp.status_code, "platform provision response missing apiKey")
        return key

    def get_balance(self, *, base_url: str, api_key: str) -> dict[str, Any]:
        resp = self._http.get(
            self._url(base_url, "/balance"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self._timeout,
        )
        return self._require_json(resp, "balance")

    def create_checkout(self, *, base_url: str, api_key: str, tier_credits_cents: int) -> str:
        resp = self._http.post(
            self._url(base_url, "/checkout"),
            headers={"Authorization": f"Bearer {api_key}"},
            json_payload={"tier": tier_credits_cents},
            timeout=self._timeout,
        )
        payload = self._require_json(resp, "checkout")
        url = payload.get("url")
        if not isinstance(url, str) or not url:
            raise PlatformError(resp.status_code, "platform checkout response missing url")
        return url

    def link_email(self, *, base_url: str, api_key: str, email: str) -> None:
        resp = self._http.post(
            self._url(base_url, "/link-email"),
            headers={"Authorization": f"Bearer {api_key}"},
            json_payload={"email": email},
            timeout=self._timeout,
        )
        self._require_json(resp, "link-email")

    def recover_request(self, *, base_url: str, email: str) -> None:
        resp = self._http.post(
            self._url(base_url, "/recover/request"),
            json_payload={"email": email},
            timeout=self._timeout,
        )
        self._require_json(resp, "recover/request")

    def recover_confirm(self, *, base_url: str, email: str, code: str) -> str:
        resp = self._http.post(
            self._url(base_url, "/recover/confirm"),
            json_payload={"email": email, "code": code},
            timeout=self._timeout,
        )
        payload = self._require_json(resp, "recover/confirm")
        key = payload.get("apiKey")
        if not isinstance(key, str) or not key:
            raise PlatformError(resp.status_code, "platform recover/confirm response missing apiKey")
        return key
