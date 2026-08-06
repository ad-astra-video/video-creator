# Video-Creator 3-service runner compose

Builds and runs `live-runner` + `ltx-worker` + `idv2v-worker` on **one shared
GPU** (RTX 5090 / RTX PRO 6000, 32 GB).

```
live-runner ──(registers/heartbeats)──> Livepeer Orchestrator   app=video-creator
    │  owns swap policy
    ├─(routing/proxy)──> ltx-worker    :8991   generate/retake/extend/ic-lora
    └─(routing/proxy)──> idv2v-worker  :8992   /v1/restyle
```

- Only ONE worker model is resident on the GPU at a time. The live-runner evicts
  the current worker (`/evict`) before loading the requested one (`/load`).
- Workers do NOT register with the Orchestrator — they only serve the internal
  `/health /load /evict /v1/*` surface over the Docker network.
- Health-gating: workers start only after `live-runner` is healthy
  (`depends_on: condition: service_healthy`).
- Worker auth: all three share `WORKER_TOKEN` (sent as `X-Worker-Token`).
  Requests without it are rejected with 403 (covered by
  `runner/tests/test_worker_auth.py`).

## Build & run

From the video-creator repo root:

```bash
export WORKER_TOKEN="$(openssl rand -hex 16)"     # or set a fixed one
export MODELS_DIR=/home/brad/models                # host models bind-mount
docker compose -f docker/docker-compose.video-creator.yml up --build
```

To also bring up a **local** go-livepeer orchestrator (offchain, for testing
discovery/heartbeats):

```bash
docker compose -f docker/docker-compose.video-creator.yml \
  --profile orchestrator up --build
```

For production point `ORCHESTRATOR_URL` / `ORCHESTRATOR_SECRET` at your real
orchestrator (the `orchestrator` profile service is then unused).

## Images

| Service        | Dockerfile                        | Entry                    |
|----------------|-----------------------------------|--------------------------|
| live-runner    | `docker/live-runner.Dockerfile`   | `python -m runner.live_runner` |
| ltx-worker     | `docker/ltx-worker.Dockerfile`    | `python -m runner.ltx.server`  |
| idv2v-worker   | `docker/idv2v-worker.Dockerfile`  | `python -m runner.idv2v`       |

The idv2v-worker installs the Wan/VACE `diffsynth` fork + SAM3 from the
Eyeline-Labs/ID-V2V reference repo (cloned at build), then int8-quantizes
DiT+VACE with CPU offload for the 32 GB card (`IDV2V_QUANT=int8`,
`IDV2V_OFFLOAD=true`).

## Model files (bind-mount at /models)

- `ltx-worker`: `checkpoint/`, `gemma/`, optional `loras/`
- `idv2v-worker`: `idv2v.pth`, `wan/`, `sam3/` (or set `IDV2V_SKIP_SAM3=1`)
