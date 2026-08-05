# Video Creator — Platform API Integration Checklist

Tracks wiring the `video-creator-platform` serverless backend (Stripe checkout + platform fee,
PymtHouse credits, per-user keys) into the **Video Creator** desktop app (this repo).

Backend API surface we integrate with: `POST /provision`, `GET /balance`, `POST /checkout`,
`POST /link-email`, `POST /link-email/verify`, `POST /recover/request`, `POST /recover/confirm`.
User routes authenticate with `Authorization: Bearer <per-user key>` (user derived from key).

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## 0. Preflight
- [x] Baseline clean commit before any changes (`465af83`: runner/ + docker/ baseline).
- [x] Populated checklist file (this) at start of implementation.

## 1. Module A — Backend foundation: platform client + settings + provisioning
**Goal:** desktop can spend + recover credits and gate remote generation.
**Commit:** `bf2c2d9`. All touched files pyright-clean; 587/588 tests pass (only `test_pyright` red on 16 pre-existing out-of-scope errors in ic_lora_handler/providers/app_factory).
- [x] Add platform settings fields: `platform_base_url`, `platform_user_id`, `platform_api_key`
      (secret), `platform_recovery_email` — in `backend/state/app_settings.py`.
- [x] Mask `platform_api_key` in `SettingsResponse` (`has_platform_api_key`) via existing pattern.
- [x] Add `PlatformClient` service (Protocol + `HttpPlatformClient` + `FakePlatformClient`).
- [x] Wire real client into `AppHandler`; fake into `tests/fakes/` + conftest wiring.
- [x] `PlatformHandler.ensure_ready()`: ensure `platform_user_id` (uuid4 once) + provision key once.
- [x] Thin route `GET /platform/status` (no secrets) — camelCase JSON.
- [x] Integration test (`tests/test_platform_client.py`) — 5/5 green.
- [x] Verify: Module A adds **zero** pyright errors (remaining 71 are pre-existing in
      `livepeer_client.py`/`ic_lora_handler.py`/`providers.py`/`video_generation_handler.py` —
      untouched by Module A; `livepeer_client.py` gets fixed in Module B). New tests green.
- [x] **Clean commit — Module A** (`8ec6f40`).

## 2. Module B — Backend credits: balance/checkout + recovery + dispatch gate (DONE)
- [x] Credits domain: handler + routes for `GET /balance`, `POST /checkout` (tier -> Stripe URL).
- [x] Recovery endpoints: `POST /link-email`, `POST /recover/request`, `POST /recover/confirm` (key rotation).
- [x] Balance gate on remote Livepeer dispatch (handler-level `ensure_generation_allowed`, wired in `app_handler.py`;
      fail-open, no-op when unconfigured).
- [x] Pydantic request/response models in `backend/api_types.py` (camelCase aliases).
- [x] Integration tests (`tests/test_platform_credits.py`).
- [x] Verify: all Module B files pyright-clean; 587/588 pass; `test_logging_policy` now green.
- [x] **Clean commit — Module B** (`bf2c2d9`).

## 3. Module C — Frontend: Credits panel + settings (IN PROGRESS)
- [~] Platform base URL field in settings UI (frontend).
- [ ] Credits panel: show live balance, Refresh, Deposit button (opens Stripe checkout URL).
- [ ] Wire all calls through `backendFetch` (never raw `fetch`).
- [ ] Handle insufficient-credits state in the generation UI.
- [ ] Verify: `pnpm typecheck:ts` + `pnpm build:frontend` + `pnpm typecheck:py` + `pnpm backend:test`.
- [ ] **Clean commit — Module C.**

## 4. Final verification
- [ ] `pnpm typecheck` (TS + Python) green.
- [ ] `pnpm backend:test` green.
- [ ] Manual flow: fresh install provisions -> deposit opens Stripe -> balance reflects credits ->
      remote generation gates correctly.
- [ ] Update `ONBOARDING_AND_EXECUTION_PLAN.md` desktop-side section to reflect implemented state.
