#!/bin/sh
set -e

cd /app

if [ ! -f data/courses.json ]; then
    echo "[entrypoint] No synthetic data found -- generating (seed=${DATA_SEED:-42})..."
    uv run python data/generate_synthetic_data.py --seed "${DATA_SEED:-42}"
else
    echo "[entrypoint] Synthetic data already present, reusing it."
fi

echo "[entrypoint] Waiting for Neo4j at ${NEO4J_URI:-bolt://neo4j:7687}..."
uv run python - <<'EOF'
import os
import sys
import time

from neo4j import GraphDatabase

uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
user = os.environ.get("NEO4J_USER", "neo4j")
password = os.environ.get("NEO4J_PASSWORD", "password123")

for attempt in range(30):
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        print("[entrypoint] Neo4j is reachable.")
        sys.exit(0)
    except Exception as e:
        print(f"[entrypoint] Neo4j not ready yet ({e}); retrying...")
        time.sleep(2)

print("[entrypoint] Gave up waiting for Neo4j.", file=sys.stderr)
sys.exit(1)
EOF

if [ ! -f data/schedule.json ]; then
    echo "[entrypoint] No schedule found -- running the solver once..."
    uv run python solver/timetable_solver.py
fi

echo "[entrypoint] Loading data + schedule into Neo4j..."
uv run python graph/load_graph.py

echo "[entrypoint] Starting app..."
exec uv run python app.py
