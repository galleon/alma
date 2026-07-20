# Academic Timetable Copilot

A self-contained proof-of-concept for university academic scheduling automation.
It exists to make one point concrete to a prospective client: **a real constraint
solver produces the timetable; an LLM only explains and queries what the solver
already decided.** The LLM never guesses at a schedule.

- **Solver (the core, non-negotiable component):** Google OR-Tools CP-SAT
- **Graph store:** Neo4j (Students / Faculty / Courses / Sections / Rooms / Policies)
- **Explainability layer:** a small function-calling agent over the graph + solver,
  backed by a local Ollama model by default (swappable for any OpenAI-compatible API)
- **Registration eligibility RAG (stretch goal):** local sentence-transformers
  embeddings over 6 plain-text academic policies
- **UI:** Gradio, single page, three tabs

Everything runs locally or on free tiers. All data is synthetic -- no real
institutional data is used or required.

## Architecture

```
data/generate_synthetic_data.py --> data/*.json (courses, faculty, rooms, students, policies)
                                          |
                                          v
                          solver/timetable_solver.py  (OR-Tools CP-SAT)
                                          |
                                          v
                                data/schedule.json
                                          |
                                          v
                          graph/load_graph.py  -->  Neo4j (Docker)
                                          |
                     graph/queries.py (canned Cypher) <--- explain/agent.py (Ollama, function-calling)
                                          |                           |
                                          |                 rag/eligibility_bot.py (sentence-transformers)
                                          v                           |
                                       app.py (Gradio) -------------- +
```

## Prerequisites

- Docker + Docker Compose
- [Ollama](https://ollama.com) running **natively on the host** (not in Docker,
  so it can use your GPU), with a small tool-calling-capable model pulled
  (default: `qwen3.5:9b`; any tool-calling-capable Ollama model works -- adjust
  `LLM_MODEL` if you use a different one, e.g. `nemotron-3-nano` once available
  locally, or `llama3.1`, `qwen2.5`, etc.)
- Python 3.11+ and [uv](https://docs.astral.sh/uv/) only needed for the
  non-Docker/manual setup path below

No paid API keys are required. If you'd rather use a real OpenAI-compatible API,
set `OPENAI_API_KEY` in `.env` -- this is optional and off by default.

## Setup (under 10 minutes) -- Docker Compose (recommended)

Everything -- Neo4j and the app itself -- runs via Compose. Ollama stays on
the host so the container can reach it at `host.docker.internal:11434`.

```bash
# 1. Make sure Ollama has a model pulled (runs on the host, not in Docker)
ollama pull qwen3.5:9b   # or any other tool-calling-capable model

# 2. Build and start everything (Neo4j + the app)
docker compose up -d --build
```

The first run generates the synthetic dataset, solves the initial timetable,
and loads Neo4j automatically (see `docker-entrypoint.sh`) -- watch it happen
with `docker compose logs -f app`. Once it says `Starting app...` and is
listening, open **http://localhost:7860**.

Data persists in `./data` (bind-mounted) and Neo4j's volume, so `docker compose
up -d` on subsequent runs reuses the existing dataset/schedule instead of
regenerating it. To start over with a fresh dataset, delete `./data/*.json`
(or just `./data/schedule.json` to keep the data but force a fresh solve) and
restart the `app` service.

```bash
docker compose down          # stop everything
docker compose logs -f app   # tail the app's logs
```

### Note on Ollama connectivity

The app container reaches your host's Ollama via `host.docker.internal`,
which is supported by Docker Desktop on Mac/Windows out of the box; on Linux
`docker-compose.yml` adds the `extra_hosts: host-gateway` entry needed to make
that resolve too. Ollama must be listening on its default port (11434); no
extra Ollama configuration is required, it does not need to bind to `0.0.0.0`.

## Setup -- manual / without Docker for the app

Useful if you want faster iteration on the code, or don't want to rebuild the
image on every change. Neo4j still runs via Docker; everything else runs
directly with `uv`.

```bash
# 1. Start Neo4j only
docker compose up -d neo4j

# 2. Install Python dependencies
uv sync

# 3. Make sure Ollama has a model pulled
ollama pull qwen3.5:9b

# 4. Generate the synthetic dataset (seeded, reproducible)
uv run python data/generate_synthetic_data.py --seed 42

# 5. Run the solver once to produce the initial schedule (sanity check + timing)
uv run python solver/timetable_solver.py

# 6. Load everything into Neo4j
uv run python graph/load_graph.py

# 7. (Optional) sanity-check the canned Cypher queries against the live graph
uv run python graph/queries.py

# 8. Launch the app
uv run python app.py
```

Then open the URL Gradio prints (typically http://127.0.0.1:7860).

Copy `.env.example` to `.env` if you want to override any defaults (Neo4j
credentials, LLM model/endpoint, data seed) -- the app runs fine with no `.env`
at all, using the built-in defaults. Note: `.env` is for this manual path only
-- `docker-compose.yml` intentionally hardcodes the in-network Neo4j URL and
the `host.docker.internal` Ollama address for the containerized app so a
host-oriented `.env` can't accidentally break it (see the comment in
`docker-compose.yml`).

## What's in the synthetic dataset

One small fictional college, generated by `data/generate_synthetic_data.py --seed 42`:

- **20 courses** across 6 subjects (CS, MATH, PHYS, BIO, CHEM, HUM), mixing
  mandatory and elective, lecture and lab, with prerequisite chains
- **10 faculty** with subject specializations, weekly teaching-hour caps,
  per-day availability windows, and pre-existing leave slots
- **5 rooms**: 2 lecture halls, 1 seminar room, 2 labs, each with a capacity
- **300 students** with a program, year, and prerequisite-consistent completed-course history
- **6 plain-text academic policies**, each of which maps directly to one of
  the solver's hard constraints (and is what the RAG eligibility bot cites)

Re-run the generator with `--seed <n>` for a different (still reproducible) dataset.

## Key design decisions / simplifications (documented so nothing looks accidental)

- **One section per course.** This is a single-term PoC; each course has exactly
  one weekly meeting block (matching the "section -> faculty, room, day/time
  block" schedule shape from the spec), not multiple sessions per week. This was
  chosen deliberately to keep the CP-SAT model small enough to resolve in well
  under 5 seconds for a live demo, while still exercising every real hard
  constraint (faculty qualification/availability, room type/capacity, no
  double-booking, no same-program mandatory-course clashes, max weekly load).
- **30-slot week grid.** 5 days x 6 periods (90 minutes each, 08:00-17:00).
- **Minimal-disruption re-solves.** When you toggle a constraint (e.g. mark a
  faculty member unavailable), the solver is re-run with the *original* schedule
  as an anchor and a heavy penalty for deviating from it. Without this, CP-SAT
  would happily return an equally-optimal but totally different schedule on
  every re-solve, which would make the "what changed and why" story
  incoherent. With it, a single-faculty change produces a small, explainable
  diff (typically 1-3 sections), which is what you want to be able to say to a
  client: "look how contained the impact of this change is."
- **Deterministic solves.** `num_search_workers=1` and a fixed `random_seed`
  so re-running the exact same model on the same machine/environment always
  gives the exact same schedule -- important for demo repeatability and for
  diffing to mean anything. (This determinism is per-environment, not
  cross-platform: CP-SAT's tie-breaking among equally-optimal solutions can
  differ between e.g. a native macOS run and a Linux container, since the
  guarantee is about the search being reproducible given identical
  binaries/hardware, not bit-identical across different platforms. The
  anchored what-if re-solve -- the one actually used in the demo -- sidesteps
  this entirely, since it always minimizes disruption from whatever the
  current schedule already is.)
- **Enrollment estimates are capped to fit an available room by construction**
  in the data generator, so the dataset is guaranteed solvable (the sanity
  check in `generate_synthetic_data.py` fails loudly instead of silently
  producing an infeasible instance).
- **Single-presenter app state.** `app.py` keeps schedule/override state in
  module-level globals rather than per-browser-session state, since this is a
  demo one person drives at a time, not a multi-tenant product.

## Project structure

```
data/generate_synthetic_data.py   synthetic data generator (seeded)
solver/timetable_solver.py        CP-SAT solver + independent schedule verifier
graph/load_graph.py                loads data + solved schedule into Neo4j
graph/queries.py                   canned Cypher queries used by the agent
explain/agent.py                   function-calling explainability agent (Ollama)
rag/eligibility_bot.py             RAG eligibility bot over the 6 policies
app.py                             Gradio UI (3 tabs)
Dockerfile                         image for the app service
docker-entrypoint.sh               generates data / solves / loads graph / starts app
docker-compose.yml                 Neo4j + the app, wired together
config.py                          shared env config + time-slot grid
```

## Troubleshooting

- **"Missing synthetic data files"** -- in the manual setup, run step 4 above
  first; in Docker, this shouldn't happen (the entrypoint generates it), but
  if it does check `docker compose logs app`.
- **Neo4j connection errors** -- confirm `docker compose ps` shows the
  `timetable-copilot-neo4j` container healthy; the Neo4j browser is at
  http://localhost:7474 (user `neo4j`, password `password123` by default).
- **Chat tool calls fail / empty replies** -- confirm `ollama list` shows your
  configured `LLM_MODEL` and that Ollama is running on the host (it normally
  runs as a background service after install). From inside the container you
  can check reachability with:
  `docker exec timetable-copilot-app python -c "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:11434/v1/models').status)"`
- **Solver takes longer than expected** -- the default time budget is 2s
  (`time_limit_s` in `solve_timetable`); on this dataset it reliably reaches a
  near-optimal, always constraint-satisfying schedule well within that budget.
- **A "fresh regenerate" right after switching between Docker and manual runs
  shows a big diff** -- expected; see the determinism note above. It's still a
  fully valid, constraint-satisfying schedule, just a different equally-optimal
  one. Click "Reset to original schedule" or just re-run "Regenerate" again
  within the same environment to see it stabilize.

## Non-functional notes

- No secrets are hardcoded; all configuration is via `.env` (see `.env.example`).
- Solver runtime is bounded to 2s by default (well under the 5s requirement),
  and what-if re-solves anchored to an existing schedule typically finish in
  well under 0.5s.
- See `DEMO_SCRIPT.md` for a 5-minute walkthrough script for presenting this
  to a prospective client.
