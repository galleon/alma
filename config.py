"""Shared configuration for Academic Timetable Copilot.

Centralizes environment settings and the time-slot grid so the solver,
graph loader, agent, and UI all agree on the same representation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# --- LLM (Ollama by default, OpenAI-compatible if OPENAI_API_KEY / LLM_* set) ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b")
LLM_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY", "ollama")

# --- Data generation ---
DEFAULT_SEED = int(os.getenv("DATA_SEED", "42"))

# --- Time-slot grid ---
# 5 days x 6 periods/day = 30 discrete slots. Each period is 90 minutes,
# 08:00-17:00. Sections occupy exactly one slot (one weekly meeting block),
# matching the "section -> faculty, room, day/time block" schedule shape.
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
PERIODS = [
    ("08:00", "09:30"),
    ("09:30", "11:00"),
    ("11:00", "12:30"),
    ("12:30", "14:00"),
    ("14:00", "15:30"),
    ("15:30", "17:00"),
]
PERIODS_PER_DAY = len(PERIODS)
NUM_SLOTS = len(DAYS) * PERIODS_PER_DAY
MORNING_PERIODS = {0, 1, 2}  # periods before 12:30 count as "morning"


def slot_index(day_idx: int, period_idx: int) -> int:
    return day_idx * PERIODS_PER_DAY + period_idx


def slot_to_day_period(slot: int) -> tuple[int, int]:
    return divmod(slot, PERIODS_PER_DAY)


def slot_label(slot: int) -> str:
    day_idx, period_idx = slot_to_day_period(slot)
    start, end = PERIODS[period_idx]
    return f"{DAYS[day_idx]} {start}-{end}"


def all_slot_labels() -> list[str]:
    return [slot_label(s) for s in range(NUM_SLOTS)]


def parse_slot_label(label: str) -> int | None:
    """Inverse of slot_label(). Returns None if the label doesn't match."""
    label = label.strip()
    for s in range(NUM_SLOTS):
        if slot_label(s) == label:
            return s
    return None


@dataclass(frozen=True)
class Paths:
    courses: Path = DATA_DIR / "courses.json"
    faculty: Path = DATA_DIR / "faculty.json"
    rooms: Path = DATA_DIR / "rooms.json"
    students: Path = DATA_DIR / "students.json"
    policies: Path = DATA_DIR / "policies.json"
    schedule: Path = DATA_DIR / "schedule.json"


PATHS = Paths()
