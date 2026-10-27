# Conductor v0.3

A sparse MoE Conductor (128 experts, top-8) routes every query across a pool of specialist models. Four of the specialists are MoE models themselves, and the fifth is a dense fast path. Each MoE specialist runs its own token-level expert gating. The Conductor then merges their outputs, weighted by the routing probabilities it assigned. That gives three stacked softmaxes per request: cluster, expert, token.

```
query ──┬──────────────────────────────► FAST  (dense 4B, always on, streams into a buffer)
        └► L0 Conductor (MoE 128/top-8) ─ softmax over {code, reasoning, vision, long_ctx}
                │  constraints → capacity factor (k≤2) → breaker-aware top-k
                ├─ trivial ───────────────► return FAST
                ├─ k=1 ───► specialist ───► return directly (streamed)
                └─ k=2 ───► specialists in parallel (each: own L2 token gating)
                                 └► L0 re-invoked as synthesiser, outputs annotated with p(cluster)
```

## Pool

| Tier | Role | Default checkpoint | Arch | Experts / active |
|---|---|---|---|---|
| L0 | Conductor + synthesiser | `Qwen/Qwen3-30B-A3B-FP8` | MoE 30.5B / 3.3B active, 32K | 128 / 8 |
| L1 | code | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | MoE 30.5B / 3.3B, 256K | 128 / 8 |
| L1 | reasoning | `Qwen/Qwen3-Next-80B-A3B-Thinking-FP8` | Hybrid Gated DeltaNet + attention MoE, 80B / 3B | 512 + 1 shared / 10 |
| L1 | vision | `Qwen/Qwen3-VL-30B-A3B-Instruct` (BF16) | VL-MoE 31B / 3B | 128 / 8 |
| L1 | long_ctx | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | MoE 30.5B / 3.3B, 256K native, 1M via YaRN | 128 / 8 |
| — | fast (shadow) | `Qwen/Qwen3-4B-Instruct-2507-FP8` | dense 4B | none |

All checkpoints are Apache-2.0. Swapping one means setting `*_MODEL` / `*_URL` env vars or editing `config/pool.yaml`. The roles stay fixed.

## The six techniques and where they live

| Technique | MoE analogue | Implementation |
|---|---|---|
| Soft routing with weights | top-k gating | `router.plan` normalises the L0 softmax and keeps clusters with p ≥ `min_weight` |
| Capacity factor | expert capacity | `routing.capacity_factor: 2`. Overflow is counted in `conductor_capacity_drops_total` |
| Load-balance surveillance | aux load-balancing loss | `monitor.Monitor` raises an alert when a cluster's 24h share is <5% or >60% (shares of L1 invocations, summing to 1) |
| Shadow path as latency floor | shared expert | `orchestrator.FastShadow` starts at t=0 and is replayed live (trivial path / fallback) or attached to synthesis |
| Probability-weighted synthesis | gated combination | `Orchestrator._synthesis_messages` orders outputs by weight and annotates each with p |
| Routing-collapse detection | dead-expert detection | 1h share >80% or <2% sets a shared uniform-mix flag for `uniform_hold_s` and raises an alert |

Production behaviour beyond the spec:

- **Guided decoding.** L0 output is constrained by a JSON schema sent through `response_format`, which vLLM serves with xgrammar or outlines. A malformed reply is retried once. After that the request uses the fast path (`conductor_router_fallbacks_total`).
- **Hard constraints.** Image attachments force `vision`. Context above `long_ctx_force_tokens` forces `long_ctx`. A forced cluster keeps at least `min_weight` of the merge mass.
- **Opaque attachments.** The Conductor sees `images_attached: N`, a token estimate and a head+tail clip of the query. It never sees image bytes or full long contexts, so its 32K window is never exceeded.
- **Circuit breakers.** Each specialist has its own breaker. When a breaker is open, that cluster's capacity slot passes to the next-ranked cluster. A cancelled half-open probe gives its slot back.
- **Graceful degradation.** If one specialist fails, synthesis is skipped. If all fail, the fast answer is returned. If synthesis fails, the highest-weighted output is returned. If a streamed specialist fails before its first token, the fast-path stream is replayed.
- **Reasoning privacy.** Thinking traces (`reasoning_content` or inline `<think>`) are dropped unless the request sets `show_reasoning: true`.
- **Shared state.** With `MONITOR_STORE=redis`, load counters, the uniform flag and alert de-duplication are shared across replicas.

## API

OpenAI-compatible:

```bash
curl localhost:8080/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Prove sqrt(2) is irrational, then write a Python check"}],"stream":true,"debug":true}'
```

Extensions: `show_reasoning`, `debug` (adds an `x_conductor` block with probs, selection, merge weights and reasons), `session_id`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | full pipeline, streaming or not |
| `POST /v1/route` | L0 only, no specialists fire (for router evals) |
| `GET /v1/pool` | tiers, checkpoints, breaker states |
| `GET /v1/monitor` | current balance / collapse windows |
| `GET /metrics` | Prometheus |
| `GET /healthz`, `/readyz` | liveness / readiness (requires Conductor and fast path up) |

## Run

The default config uses `mock://` backends, so the service runs without GPUs:

```bash
pip install -e ".[dev]"
pytest                 # 49 tests
conductor              # :8080
```

Production on one node with 6×80GB GPUs. Card layout and vLLM flags are in `deploy/docker-compose.yml`:

```bash
cd deploy
export VLLM_API_KEY=$(openssl rand -hex 24) HF_TOKEN=... CONDUCTOR_API_KEYS=client-key-1
printf %s "$VLLM_API_KEY" > prometheus/vllm_api_key
docker compose up -d
```

Kubernetes (GPU operator required):

```bash
docker build -t ghcr.io/dmhall3204/conductor:0.3.0 . && docker push ghcr.io/dmhall3204/conductor:0.3.0
kubectl apply -f deploy/k8s/namespace.yaml
kubectl -n conductor create secret generic conductor-secrets \
  --from-literal=vllm-api-key=$(openssl rand -hex 24) --from-literal=conductor-api-keys=client-key-1
kubectl apply -k deploy/k8s
```

`deploy/k8s/vllm-models.yaml` is generated from `deploy/k8s/generate.py`. Edit the generator, not the YAML.

Tracing: `pip install ".[otel]"` and set `OTEL_EXPORTER_OTLP_ENDPOINT`. You get spans for `l0.route`, each `l1.<cluster>` and `l0.synthesise`.

## Known limits

- The routing quality depends on the L0 prompt and model. Use `/v1/route` with a labelled set to measure it before going live. The load-balance alerts tell you when to revisit it, but they don't tune anything automatically.
- 1M-token long context needs YaRN plus far more KV memory than one GPU has. The compose file ships native 256K and shows the flag in a comment.
- vLLM flags target `vllm/vllm-openai:v0.11.0`. Newer images rename some options (for example structured-output backends), so re-check them when you upgrade.
