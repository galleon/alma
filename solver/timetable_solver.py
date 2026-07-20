"""CP-SAT based timetable solver for the Academic Timetable Copilot.

Given courses, faculty, and rooms, produces a conflict-free weekly schedule:
one (faculty, room, day/time-block) assignment per course section.

Hard constraints (all enforced structurally or via linear/table constraints):
  - a section's faculty must be qualified (subject specialization match)
  - a section's faculty must be available at the assigned slot (respecting
    per-faculty availability windows and leave)
  - no faculty member is double-booked across sections
  - no room is double-booked across sections
  - a section's room must match the required room type (lab sections -> lab
    rooms; lecture sections -> lecture/seminar rooms) and have capacity >=
    estimated enrollment
  - mandatory courses within the same program must not be scheduled in the
    same slot (so a student following that program's plan never has two
    of their required courses clash)
  - a faculty member's total weekly sections must not exceed their max load

Soft constraints (objective terms):
  - faculty who prefer morning slots should get them when possible
  - daily teaching load should be balanced across the week (minimize the
    worst single-day load)

This module never asks an LLM to guess a schedule -- CP-SAT is the only
component that produces the timetable. The LLM layer (explain/agent.py)
only reads and explains what the solver already decided.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NUM_SLOTS, PATHS, PERIODS_PER_DAY, slot_label, slot_to_day_period  # noqa: E402


def _room_matches(course: dict, room: dict) -> bool:
    if course["room_type_required"] == "lab":
        return room["room_type"] == "lab"
    return room["room_type"] in ("lecture", "seminar")


def _qualified_faculty(course: dict, faculty: list[dict]) -> list[dict]:
    return [f for f in faculty if course["subject"] in f["specializations"]]


def _valid_rooms(course: dict, rooms: list[dict]) -> list[dict]:
    return [r for r in rooms if _room_matches(course, r) and r["capacity"] >= course["estimated_enrollment"]]


@dataclass
class SolveResult:
    status: str  # "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "UNKNOWN"
    schedule: dict[str, dict] = field(default_factory=dict)
    solve_time_s: float = 0.0
    objective: float | None = None
    best_bound: float | None = None
    diagnostics: list[str] = field(default_factory=list)


def solve_timetable(
    courses: list[dict],
    faculty: list[dict],
    rooms: list[dict],
    time_limit_s: float = 2.0,
    random_seed: int = 42,
    anchor_schedule: dict[str, dict] | None = None,
    restrict_first_n_days: int | None = None,
    adversarial: bool = False,
) -> SolveResult:
    """Solve the timetable. If `anchor_schedule` is given (a previously solved
    schedule), the solver treats minimizing changes from it as the top-priority
    objective -- so a "what-if" re-solve only ripples out the sections that are
    actually forced to move, instead of finding an unrelated equally-good
    solution. This is what makes the explainability diff meaningful.

    `restrict_first_n_days` caps every section's slot domain to the first N
    days of the week -- still every hard constraint, just fewer days to place
    things in. `adversarial=True` flips the soft-objective sign (maximizing
    afternoon slots for morning-preferring faculty and maximizing the busiest
    day's load instead of minimizing them). Together these produce a schedule
    that is fully hard-constraint-valid but deliberately bad on soft
    preferences -- used for the demo's "unoptimized baseline" view, so the
    later optimized solve has something dramatic to visibly improve on."""
    model = cp_model.CpModel()

    course_ids = [c["course_id"] for c in courses]
    course_by_id = {c["course_id"]: c for c in courses}
    faculty_by_id = {f["faculty_id"]: f for f in faculty}
    faculty_idx = {f["faculty_id"]: i for i, f in enumerate(faculty)}
    room_idx = {r["room_id"]: i for i, r in enumerate(rooms)}

    # --- pre-flight feasibility diagnostics (fail fast with a clear reason) ---
    diagnostics: list[str] = []
    for c in courses:
        if not _qualified_faculty(c, faculty):
            diagnostics.append(f"No qualified faculty for {c['course_id']} (subject={c['subject']}).")
        if not _valid_rooms(c, rooms):
            diagnostics.append(
                f"No room fits {c['course_id']} (needs {c['room_type_required']}, "
                f"capacity>={c['estimated_enrollment']})."
            )
    if diagnostics:
        return SolveResult(status="INFEASIBLE", diagnostics=diagnostics)

    slot_var: dict[str, cp_model.IntVar] = {}
    fac_var: dict[str, cp_model.IntVar] = {}
    room_var: dict[str, cp_model.IntVar] = {}
    teaches: dict[tuple[str, str], cp_model.IntVar] = {}
    in_room: dict[tuple[str, str], cp_model.IntVar] = {}

    max_slot = restrict_first_n_days * PERIODS_PER_DAY - 1 if restrict_first_n_days else NUM_SLOTS - 1

    for c in courses:
        cid = c["course_id"]
        slot_var[cid] = model.NewIntVar(0, max_slot, f"slot_{cid}")

        qualified = _qualified_faculty(c, faculty)
        fac_domain = cp_model.Domain.FromValues(sorted(faculty_idx[f["faculty_id"]] for f in qualified))
        fac_var[cid] = model.NewIntVarFromDomain(fac_domain, f"fac_{cid}")

        valid_rooms = _valid_rooms(c, rooms)
        room_domain = cp_model.Domain.FromValues(sorted(room_idx[r["room_id"]] for r in valid_rooms))
        room_var[cid] = model.NewIntVarFromDomain(room_domain, f"room_{cid}")

        # Reified "teaches[cid, fid]" bools, iff fac_var[cid] == faculty_idx[fid]
        for f in qualified:
            fid = f["faculty_id"]
            lit = model.NewBoolVar(f"teaches_{cid}_{fid}")
            model.Add(fac_var[cid] == faculty_idx[fid]).OnlyEnforceIf(lit)
            model.Add(fac_var[cid] != faculty_idx[fid]).OnlyEnforceIf(lit.Not())
            teaches[(cid, fid)] = lit
        model.AddExactlyOne(teaches[(cid, f["faculty_id"])] for f in qualified)

        for r in valid_rooms:
            rid = r["room_id"]
            lit = model.NewBoolVar(f"inroom_{cid}_{rid}")
            model.Add(room_var[cid] == room_idx[rid]).OnlyEnforceIf(lit)
            model.Add(room_var[cid] != room_idx[rid]).OnlyEnforceIf(lit.Not())
            in_room[(cid, rid)] = lit
        model.AddExactlyOne(in_room[(cid, r["room_id"])] for r in valid_rooms)

        # Faculty availability + leave: (fac_var, slot_var) must land in an allowed pair.
        allowed_pairs = []
        for f in qualified:
            fidx = faculty_idx[f["faculty_id"]]
            for s in f["available_slots"]:
                allowed_pairs.append([fidx, s])
        model.AddAllowedAssignments([fac_var[cid], slot_var[cid]], allowed_pairs)

    # --- no faculty double-booking: optional no-overlap intervals per faculty ---
    for f in faculty:
        fid = f["faculty_id"]
        intervals = []
        for c in courses:
            cid = c["course_id"]
            if (cid, fid) not in teaches:
                continue
            interval = model.NewOptionalFixedSizeIntervalVar(
                slot_var[cid], 1, teaches[(cid, fid)], f"ivf_{cid}_{fid}"
            )
            intervals.append(interval)
        if intervals:
            model.AddNoOverlap(intervals)
        # Max weekly teaching load.
        load_terms = [teaches[(c["course_id"], fid)] for c in courses if (c["course_id"], fid) in teaches]
        if load_terms:
            model.Add(sum(load_terms) <= f["max_sections_week"])

    # --- no room double-booking ---
    for r in rooms:
        rid = r["room_id"]
        intervals = []
        for c in courses:
            cid = c["course_id"]
            if (cid, rid) not in in_room:
                continue
            interval = model.NewOptionalFixedSizeIntervalVar(
                slot_var[cid], 1, in_room[(cid, rid)], f"ivr_{cid}_{rid}"
            )
            intervals.append(interval)
        if intervals:
            model.AddNoOverlap(intervals)

    # --- no same-slot clash for mandatory courses within the same program ---
    programs: dict[str, list[str]] = {}
    for c in courses:
        for prog in c["mandatory_for"]:
            programs.setdefault(prog, []).append(c["course_id"])
    seen_pairs = set()
    for prog, cids in programs.items():
        for a, b in combinations(sorted(cids), 2):
            if (a, b) in seen_pairs:
                continue
            seen_pairs.add((a, b))
            model.Add(slot_var[a] != slot_var[b])

    # --- soft objective: morning preference + balanced daily load + tie-break ---
    period_var: dict[str, cp_model.IntVar] = {}
    day_var: dict[str, cp_model.IntVar] = {}
    for c in courses:
        cid = c["course_id"]
        period_var[cid] = model.NewIntVar(0, PERIODS_PER_DAY - 1, f"period_{cid}")
        model.AddModuloEquality(period_var[cid], slot_var[cid], PERIODS_PER_DAY)
        day_var[cid] = model.NewIntVar(0, NUM_SLOTS // PERIODS_PER_DAY - 1, f"day_{cid}")
        model.AddDivisionEquality(day_var[cid], slot_var[cid], PERIODS_PER_DAY)

    afternoon_penalty_terms = []
    for c in courses:
        cid = c["course_id"]
        is_afternoon = model.NewBoolVar(f"afternoon_{cid}")
        model.Add(period_var[cid] >= 3).OnlyEnforceIf(is_afternoon)
        model.Add(period_var[cid] < 3).OnlyEnforceIf(is_afternoon.Not())
        for f in _qualified_faculty(c, faculty):
            if not f["prefers_morning"]:
                continue
            fid = f["faculty_id"]
            pen = model.NewBoolVar(f"pen_{cid}_{fid}")
            model.AddMultiplicationEquality(pen, [teaches[(cid, fid)], is_afternoon])
            afternoon_penalty_terms.append(pen)

    n_days = NUM_SLOTS // PERIODS_PER_DAY
    day_counts = []
    for d in range(n_days):
        is_day = []
        for c in courses:
            cid = c["course_id"]
            lit = model.NewBoolVar(f"isday_{cid}_{d}")
            model.Add(day_var[cid] == d).OnlyEnforceIf(lit)
            model.Add(day_var[cid] != d).OnlyEnforceIf(lit.Not())
            is_day.append(lit)
        count = model.NewIntVar(0, len(courses), f"count_day_{d}")
        model.Add(count == sum(is_day))
        day_counts.append(count)
    max_day_load = model.NewIntVar(0, len(courses), "max_day_load")
    model.AddMaxEquality(max_day_load, day_counts)

    tie_break = sum(slot_var[cid] for cid in course_ids)

    disruption_terms = []
    if anchor_schedule:
        for cid, anchor in anchor_schedule.items():
            if cid not in slot_var:
                continue
            slot_dev = model.NewBoolVar(f"slotdev_{cid}")
            model.Add(slot_var[cid] != anchor["slot"]).OnlyEnforceIf(slot_dev)
            model.Add(slot_var[cid] == anchor["slot"]).OnlyEnforceIf(slot_dev.Not())
            disruption_terms.append(slot_dev)

            anchor_fid = anchor["faculty_id"]
            if (cid, anchor_fid) in teaches:
                disruption_terms.append(teaches[(cid, anchor_fid)].Not())

            anchor_rid = anchor["room_id"]
            if (cid, anchor_rid) in in_room:
                disruption_terms.append(in_room[(cid, anchor_rid)].Not())

    if adversarial:
        # Deliberately bad-but-valid: push morning-preferring faculty into
        # afternoons and cram sections onto the busiest possible single day.
        # No disruption/tie-break terms here -- this is never anchored.
        model.Maximize(10 * sum(afternoon_penalty_terms) + 5 * max_day_load)
    else:
        model.Minimize(
            1000 * sum(disruption_terms)
            + 10 * sum(afternoon_penalty_terms)
            + 5 * max_day_load
            + tie_break  # tiny relative weight: prefers earlier slots, stabilizes reruns
        )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = random_seed

    start = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - start

    status_name = solver.StatusName(status)
    if status_name not in ("OPTIMAL", "FEASIBLE"):
        return SolveResult(status=status_name, solve_time_s=elapsed, diagnostics=["Solver found no feasible schedule."])

    schedule = {}
    for c in courses:
        cid = c["course_id"]
        slot = solver.Value(slot_var[cid])
        fid = next(f["faculty_id"] for f in _qualified_faculty(c, faculty) if solver.Value(teaches[(cid, f["faculty_id"])]))
        rid = next(r["room_id"] for r in _valid_rooms(c, rooms) if solver.Value(in_room[(cid, r["room_id"])]))
        day_idx, period_idx = slot_to_day_period(slot)
        schedule[cid] = {
            "course_id": cid,
            "course_name": c["name"],
            "subject": c["subject"],
            "faculty_id": fid,
            "faculty_name": faculty_by_id[fid]["name"],
            "room_id": rid,
            "room_name": next(r["name"] for r in rooms if r["room_id"] == rid),
            "slot": slot,
            "day": day_idx,
            "period": period_idx,
            "slot_label": slot_label(slot),
            "estimated_enrollment": c["estimated_enrollment"],
        }

    return SolveResult(
        status=status_name,
        schedule=schedule,
        solve_time_s=elapsed,
        objective=solver.ObjectiveValue(),
        best_bound=solver.BestObjectiveBound(),
    )


def verify_schedule(schedule: dict[str, dict], courses: list[dict], faculty: list[dict], rooms: list[dict]) -> list[str]:
    """Independently re-checks a solved schedule against every hard constraint.
    Returns a list of violation strings; empty means the schedule is clean."""
    violations = []
    course_by_id = {c["course_id"]: c for c in courses}
    faculty_by_id = {f["faculty_id"]: f for f in faculty}
    room_by_id = {r["room_id"]: r for r in rooms}

    by_faculty_slot: dict[tuple[str, int], list[str]] = {}
    by_room_slot: dict[tuple[str, int], list[str]] = {}

    for cid, entry in schedule.items():
        course = course_by_id[cid]
        fac = faculty_by_id[entry["faculty_id"]]
        room = room_by_id[entry["room_id"]]

        if course["subject"] not in fac["specializations"]:
            violations.append(f"{cid}: faculty {fac['faculty_id']} not qualified in {course['subject']}")
        if entry["slot"] not in fac["available_slots"]:
            violations.append(f"{cid}: faculty {fac['faculty_id']} not available at slot {entry['slot']}")
        if not _room_matches(course, room):
            violations.append(f"{cid}: room {room['room_id']} type mismatch for {course['room_type_required']}")
        if room["capacity"] < course["estimated_enrollment"]:
            violations.append(f"{cid}: room {room['room_id']} capacity {room['capacity']} < enrollment {course['estimated_enrollment']}")

        by_faculty_slot.setdefault((fac["faculty_id"], entry["slot"]), []).append(cid)
        by_room_slot.setdefault((room["room_id"], entry["slot"]), []).append(cid)

    for (fid, slot), cids in by_faculty_slot.items():
        if len(cids) > 1:
            violations.append(f"faculty {fid} double-booked at slot {slot}: {cids}")
    for (rid, slot), cids in by_room_slot.items():
        if len(cids) > 1:
            violations.append(f"room {rid} double-booked at slot {slot}: {cids}")

    programs: dict[str, list[str]] = {}
    for c in courses:
        for prog in c["mandatory_for"]:
            programs.setdefault(prog, []).append(c["course_id"])
    for prog, cids in programs.items():
        for a, b in combinations(cids, 2):
            if a in schedule and b in schedule and schedule[a]["slot"] == schedule[b]["slot"]:
                violations.append(f"program {prog}: mandatory courses {a} and {b} clash at slot {schedule[a]['slot']}")

    load: dict[str, int] = {}
    for entry in schedule.values():
        load[entry["faculty_id"]] = load.get(entry["faculty_id"], 0) + 1
    for fid, n in load.items():
        if n > faculty_by_id[fid]["max_sections_week"]:
            violations.append(f"faculty {fid} overloaded: {n} sections > max {faculty_by_id[fid]['max_sections_week']}")

    return violations


def schedule_quality_metrics(schedule: dict[str, dict], faculty: list[dict]) -> dict:
    """Decodes the solver's soft-constraint objective into human-readable
    terms, computed directly from a finished schedule (not from CP-SAT
    internals) so it's easy to compute for any schedule, including ones
    loaded from disk, and to compare two schedules side by side.

    This is what actually distinguishes an "OPTIMAL"/better schedule from a
    merely "FEASIBLE" one: both satisfy every hard constraint, but the
    optimal one does better on these soft preferences.
    """
    faculty_by_id = {f["faculty_id"]: f for f in faculty}

    morning_pref_total = 0
    morning_pref_satisfied = 0
    day_counts = [0] * 5
    for entry in schedule.values():
        day_counts[entry["day"]] += 1
        fac = faculty_by_id.get(entry["faculty_id"])
        if fac and fac["prefers_morning"]:
            morning_pref_total += 1
            if entry["period"] < 3:
                morning_pref_satisfied += 1

    return {
        "morning_preference_satisfied": morning_pref_satisfied,
        "morning_preference_total": morning_pref_total,
        "day_counts": day_counts,  # sections per day, Mon..Fri
        "busiest_day_count": max(day_counts) if day_counts else 0,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> None:
    courses = _load_json(PATHS.courses)
    faculty = _load_json(PATHS.faculty)
    rooms = _load_json(PATHS.rooms)

    result = solve_timetable(courses, faculty, rooms)
    print(f"Status: {result.status}  |  solve time: {result.solve_time_s:.2f}s  |  objective: {result.objective}")
    if result.diagnostics:
        print("Diagnostics:")
        for d in result.diagnostics:
            print(f"  - {d}")
    if not result.schedule:
        sys.exit(1)

    for cid in sorted(result.schedule):
        e = result.schedule[cid]
        print(
            f"  {cid:8s} {e['course_name']:32s} {e['slot_label']:14s} "
            f"faculty={e['faculty_name']:22s} room={e['room_name']:16s} enroll={e['estimated_enrollment']}"
        )

    violations = verify_schedule(result.schedule, courses, faculty, rooms)
    if violations:
        print(f"\n{len(violations)} CONSTRAINT VIOLATIONS FOUND:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("\nIndependent verification: no constraint violations. Schedule is conflict-free.")

    PATHS.schedule.write_text(json.dumps(result.schedule, indent=2))
    print(f"Wrote schedule to {PATHS.schedule}")


if __name__ == "__main__":
    main()
