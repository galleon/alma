"""Academic Timetable Copilot -- single-page Gradio demo.

Three panels:
  1. Generated Timetable -- grid view of the CP-SAT solved schedule, with a
     "regenerate" button and faculty-availability toggles that re-solve live.
  2. Ask the Copilot -- chat with the graph+solver-grounded explainability agent.
  3. Registration Eligibility Check -- RAG over the plain-text academic policies.

This is a single-presenter local demo tool, so state is held in module-level
globals (like a Streamlit cached singleton) rather than per-session Gradio
State -- simpler, and fine for a PoC that one person drives at a time.

Run with: uv run python app.py
"""
from __future__ import annotations

import copy
import json

import gradio as gr
import pandas as pd

from config import DAYS, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, PATHS
from graph.load_graph import load_all
from solver.timetable_solver import schedule_quality_metrics, solve_timetable, verify_schedule

DAY_NAME_TO_IDX = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}


# --- bootstrap: data, agent, RAG bot, neo4j driver (lazy singletons) ------------

def _require_synthetic_data():
    missing = [p.name for p in [PATHS.courses, PATHS.faculty, PATHS.rooms, PATHS.students, PATHS.policies] if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Missing synthetic data files: {', '.join(missing)}. "
            "Run `uv run python data/generate_synthetic_data.py` first."
        )


_require_synthetic_data()

STATIC = {
    "courses": json.loads(PATHS.courses.read_text()),
    "rooms": json.loads(PATHS.rooms.read_text()),
    "base_faculty": json.loads(PATHS.faculty.read_text()),
    "policies": json.loads(PATHS.policies.read_text()),
    "students": json.loads(PATHS.students.read_text()),
}

_agent = None
_eligibility_bot = None
_neo4j_driver = None


def get_agent():
    global _agent
    if _agent is None:
        from explain.agent import TimetableAgent

        _agent = TimetableAgent()
    return _agent


def get_eligibility_bot():
    global _eligibility_bot
    if _eligibility_bot is None:
        from rag.eligibility_bot import EligibilityBot

        _eligibility_bot = EligibilityBot()
    return _eligibility_bot


def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import GraphDatabase

        _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _neo4j_driver


def _sync_graph_and_agent(faculty: list[dict], schedule: dict) -> str | None:
    """Persist faculty/schedule to disk, reload Neo4j, keep the agent in sync.
    Returns a warning string on failure, else None."""
    PATHS.faculty.write_text(json.dumps(faculty, indent=2))
    PATHS.schedule.write_text(json.dumps(schedule, indent=2))
    warning = None
    try:
        load_all(get_neo4j_driver(), wipe_first=True)
    except Exception as e:
        warning = f"Neo4j sync failed ({e}). Explainability chat may show stale data. Is `docker compose up -d` running?"
    try:
        get_agent().reload_baseline()
    except Exception:
        pass
    return warning


# --- app state (single-presenter demo -> module-level, not per-session) --------
#
# Demo flow this tab is built around:
#   1. Stage "baseline": an unoptimized-but-valid schedule is shown (every hard
#      constraint holds; soft preferences are deliberately not optimized).
#   2. Stage "optimized": user clicks Optimize -> real CP-SAT objective solve,
#      diffed against the baseline so the improvement is visually obvious.
#   3. Stage "constrained": user marks a faculty member unavailable and
#      re-solves, anchored to the last optimized schedule (minimal disruption),
#      diffed against that previous schedule so only the ripple is highlighted.
#
# STATE["compare_schedule"] is "whatever was shown right before this solve" --
# it drives the "Changed" column, the "What changed" text, and the timeline
# highlighting. STATE["anchor_schedule"] is "the last *optimized* schedule" --
# it's what constrained re-solves anchor to, so toggling a constraint never
# accidentally re-anchors to the deliberately-bad baseline.

BASELINE_DAYS = 2  # matches the number of days the optimized solve naturally uses


def _solve_baseline(faculty: list[dict]) -> dict:
    result = solve_timetable(STATIC["courses"], faculty, STATIC["rooms"], restrict_first_n_days=BASELINE_DAYS, adversarial=True)
    return result.schedule


STATE = {
    "original_faculty": copy.deepcopy(STATIC["base_faculty"]),
    "faculty_overrides": {},  # faculty_id -> {"unavailable_days": [...]}
}
STATE["current_faculty"] = copy.deepcopy(STATE["original_faculty"])
STATE["schedule"] = _solve_baseline(STATE["current_faculty"])
STATE["compare_schedule"] = None  # nothing to diff against yet
STATE["anchor_schedule"] = None  # no optimized schedule to anchor to yet
_startup_warning = _sync_graph_and_agent(STATE["current_faculty"], STATE["schedule"])
if _startup_warning:
    print(f"[app] {_startup_warning}")


# --- solving helpers -------------------------------------------------------------

def _apply_overrides(base_faculty: list[dict], overrides: dict) -> list[dict]:
    modified = copy.deepcopy(base_faculty)
    for f in modified:
        ov = overrides.get(f["faculty_id"])
        if not ov:
            continue
        day_idxs = {DAY_NAME_TO_IDX[d] for d in ov.get("unavailable_days", [])}
        if day_idxs:
            removed = {s for s in f["available_slots"] if s // 6 in day_idxs}
            f["available_slots"] = sorted(set(f["available_slots"]) - removed)
            f["leave_slots"] = sorted(set(f["leave_slots"]) | removed)
    return modified


def _num_changed(schedule: dict, original_schedule: dict) -> int:
    return sum(
        1
        for cid, e in schedule.items()
        if original_schedule.get(cid, {}).get("slot_label") != e["slot_label"]
        or original_schedule.get(cid, {}).get("faculty_id") != e["faculty_id"]
        or original_schedule.get(cid, {}).get("room_id") != e["room_id"]
    )


def _entry_changed(e: dict, orig: dict | None) -> bool:
    if orig is None:
        return True
    return (
        orig.get("slot_label") != e["slot_label"]
        or orig.get("faculty_id") != e["faculty_id"]
        or orig.get("room_id") != e["room_id"]
    )


def _diff_description(e: dict, orig: dict | None) -> str:
    if orig is None:
        return "new"
    parts = []
    if orig.get("slot_label") != e["slot_label"]:
        parts.append(f"time {orig['slot_label']} → {e['slot_label']}")
    if orig.get("faculty_name") != e["faculty_name"]:
        parts.append(f"faculty {orig['faculty_name']} → {e['faculty_name']}")
    if orig.get("room_name") != e["room_name"]:
        parts.append(f"room {orig['room_name']} → {e['room_name']}")
    return "; ".join(parts)


def _schedule_dataframe(schedule: dict, compare_schedule: dict | None) -> pd.DataFrame:
    """`compare_schedule` is whatever was shown right before this solve -- None
    means "nothing to diff against yet" (the very first baseline view)."""
    rows = []
    for cid, e in schedule.items():
        orig = (compare_schedule or {}).get(cid)
        changed = _entry_changed(e, orig) if compare_schedule is not None else False
        rows.append(
            {
                "Changed": "●" if changed else "",
                "Course": cid,
                "Name": e["course_name"],
                "Day": DAYS[e["day"]],
                "Time": e["slot_label"].split(" ", 1)[1],
                "Faculty": e["faculty_name"],
                "Room": e["room_name"],
                "Enrollment": e["estimated_enrollment"],
                "What changed": _diff_description(e, orig) if compare_schedule is not None else "",
            }
        )
    df = pd.DataFrame(rows).sort_values(["Day", "Time", "Course"])
    return df.reset_index(drop=True)


def _style_dataframe(df: pd.DataFrame):
    """Highlights changed rows so the diff is obvious at a glance, not just
    the bullet in the Changed column. Columns stay independently sortable by
    clicking their headers -- this only paints cell backgrounds."""

    def highlight(row):
        style = "background-color:#ffe08a; color:#3a2a00" if row["Changed"] == "●" else ""
        return [style] * len(row)

    return df.style.apply(highlight, axis=1)


def _quality_markdown(schedule: dict, faculty: list[dict], label: str) -> str:
    m = schedule_quality_metrics(schedule, faculty)
    total, sat = m["morning_preference_total"], m["morning_preference_satisfied"]
    morning_line = f"{sat}/{total} respected" if total else "n/a (no morning-preferring faculty assigned)"
    day_line = " | ".join(
        f"**{DAYS[i]}: {c}**" if c == m["busiest_day_count"] and c > 0 else f"{DAYS[i]}: {c}"
        for i, c in enumerate(m["day_counts"])
    )
    is_good = total > 0 and sat / total >= 0.7 and m["busiest_day_count"] <= 15
    verdict = "close to optimal" if is_good else "far from optimal"
    return (
        f"**{label} -- soft-preference scorecard ({verdict}):**\n"
        f"- Morning-slot preference: {morning_line}\n"
        f"- Sections per day (bold = busiest day): {day_line}"
    )


def _timeline_html(schedule: dict, compare_schedule: dict | None) -> str:
    """Weekly calendar view: days as columns, periods as rows. Changed
    sections (vs. `compare_schedule`) get a distinct highlighted card so the
    same diff that's in the table is also visible spatially."""
    from config import PERIODS

    grid: dict[tuple[int, int], list[tuple[str, dict]]] = {}
    for cid, e in schedule.items():
        grid.setdefault((e["day"], e["period"]), []).append((cid, e))

    style = (
        "<style>"
        ".tt-grid{display:grid;grid-template-columns:92px repeat(5,1fr);gap:4px;font-size:12.5px;}"
        ".tt-head{font-weight:700;text-align:center;padding:6px 4px;}"
        ".tt-time{font-weight:600;padding:6px 4px;opacity:0.75;}"
        ".tt-cell{min-height:44px;border:1px solid rgba(128,128,128,0.3);border-radius:6px;padding:3px;}"
        ".tt-card{border-radius:4px;padding:3px 5px;margin-bottom:3px;line-height:1.35;}"
        ".tt-unchanged{background:rgba(100,150,255,0.18);}"
        ".tt-changed{background:#ffe08a;color:#3a2a00;border:1px solid #d99a00;font-weight:600;}"
        ".tt-empty{opacity:0.4;font-style:italic;}"
        "</style>"
    )
    cells = ['<div class="tt-head"></div>'] + [f'<div class="tt-head">{d}</div>' for d in DAYS]
    for p, (start, end) in enumerate(PERIODS):
        cells.append(f'<div class="tt-time">{start}-{end}</div>')
        for d in range(len(DAYS)):
            entries = grid.get((d, p), [])
            if not entries:
                cells.append('<div class="tt-cell"><span class="tt-empty">—</span></div>')
                continue
            cards = []
            for cid, e in entries:
                orig = (compare_schedule or {}).get(cid)
                changed = _entry_changed(e, orig) if compare_schedule is not None else False
                cls = "tt-changed" if changed else "tt-unchanged"
                tag = " ⬚ changed" if changed else ""
                cards.append(
                    f'<div class="tt-card {cls}"><b>{cid}</b>{tag}<br>{e["faculty_name"]}<br>{e["room_name"]}</div>'
                )
            cells.append(f'<div class="tt-cell">{"".join(cards)}</div>')
    return style + f'<div class="tt-grid">{"".join(cells)}</div>'


CONSTRAINTS_EXPLAINER = """
### How this schedule is produced
Every schedule on this tab is computed by **Google OR-Tools CP-SAT**, a real constraint solver -- never by an LLM guessing.
An LLM is only ever used to *explain* a schedule the solver already produced (see "Ask the Copilot"); it never invents one.

**Hard constraints (must always hold -- non-negotiable):**
- Faculty must be qualified in the course's subject.
- Faculty must be available at the assigned day/period (respecting leave).
- No faculty member teaches two sections at the same time.
- No room hosts two sections at the same time.
- A section's room must match the required room type (lab vs. lecture/seminar) and have enough seats.
- Two mandatory courses in the same program are never scheduled in the same slot (so a student on that program's plan never has a clash).
- No faculty member exceeds their weekly teaching-load cap.

Every schedule shown here -- optimized or not -- is independently re-verified against this exact list right after solving;
any violation would be reported in red immediately below the table.

**Soft objectives (preferences the solver tries to honor, but can trade off against each other):**
- Faculty who prefer morning slots should get them.
- Daily teaching load should be balanced across the week, avoiding one over-stuffed day.
- When re-solving after a change, disrupt as few already-scheduled sections as possible.

A schedule can satisfy *every hard constraint* and still score badly on these soft objectives -- that is exactly what the
unoptimized baseline below is built to demonstrate, so the effect of clicking **Optimize** is visible rather than assumed.
"""


# --- Gradio callbacks: Timetable tab --------------------------------------------

def _current_views(label: str):
    df = _style_dataframe(_schedule_dataframe(STATE["schedule"], STATE["compare_schedule"]))
    timeline = _timeline_html(STATE["schedule"], STATE["compare_schedule"])
    quality = _quality_markdown(STATE["schedule"], STATE["current_faculty"], label)
    return df, timeline, quality


def cb_show_baseline():
    """Stage 1: (re)generate the deliberately unoptimized-but-valid baseline.
    Used both for the initial demo state and the 'reset the demo' button."""
    STATE["faculty_overrides"] = {}
    STATE["current_faculty"] = copy.deepcopy(STATE["original_faculty"])
    STATE["schedule"] = _solve_baseline(STATE["current_faculty"])
    STATE["compare_schedule"] = None
    STATE["anchor_schedule"] = None
    warning = _sync_graph_and_agent(STATE["current_faculty"], STATE["schedule"])

    violations = verify_schedule(STATE["schedule"], STATIC["courses"], STATE["current_faculty"], STATIC["rooms"])
    status = (
        "**Stage 1 -- unoptimized baseline.** Every hard constraint below is independently verified to hold, "
        "but soft preferences (morning slots, balanced load) were deliberately *not* optimized -- click "
        "**Optimize Schedule** to see the contrast."
    )
    status += "\n\n**CONSTRAINT VIOLATIONS FOUND:** " + str(violations) if violations else "\n\nIndependently verified: 0 hard-constraint violations."
    if warning:
        status += f"\n\n:warning: {warning}"

    df, timeline, quality = _current_views("Baseline")
    return status, df, timeline, quality, ""


def cb_optimize():
    """Stage 2: real CP-SAT objective solve, diffed against whatever was shown
    before (the baseline, on a first click)."""
    prev_schedule = STATE["schedule"]
    prev_faculty = STATE["current_faculty"]
    result = solve_timetable(STATIC["courses"], STATE["current_faculty"], STATIC["rooms"])

    if not result.schedule:
        status = f"**No feasible schedule found ({result.status}):** {'; '.join(result.diagnostics)}"
        df, timeline, quality = _current_views("Current")
        return status, df, timeline, quality

    violations = verify_schedule(result.schedule, STATIC["courses"], STATE["current_faculty"], STATIC["rooms"])
    STATE["compare_schedule"] = prev_schedule
    STATE["schedule"] = result.schedule
    STATE["anchor_schedule"] = result.schedule
    warning = _sync_graph_and_agent(STATE["current_faculty"], result.schedule)

    n_changed = _num_changed(result.schedule, prev_schedule)
    prev_metrics = schedule_quality_metrics(prev_schedule, prev_faculty)
    new_metrics = schedule_quality_metrics(result.schedule, STATE["current_faculty"])

    status = (
        f"**Stage 2 -- optimized.** Status: {result.status} | solved in {result.solve_time_s:.2f}s | "
        f"{n_changed}/{len(result.schedule)} section(s) changed from the unoptimized baseline (highlighted below).\n\n"
        f"**What improved:** morning-preference satisfied {prev_metrics['morning_preference_satisfied']}/{prev_metrics['morning_preference_total']} "
        f"→ {new_metrics['morning_preference_satisfied']}/{new_metrics['morning_preference_total']} | "
        f"busiest day {prev_metrics['busiest_day_count']} → {new_metrics['busiest_day_count']} sections"
    )
    if result.status == "FEASIBLE" and result.best_bound is not None:
        status += (
            f"\n\n_FEASIBLE (not OPTIMAL) means CP-SAT hit its time budget before it could `prove` no "
            f"better arrangement of the soft preferences exists -- every hard constraint is still fully satisfied. "
            f"Objective={result.objective:.0f} vs. best possible bound={result.best_bound:.0f} "
            f"(lower is better; 0 gap = provably optimal)._"
        )
    status += "\n\n**CONSTRAINT VIOLATIONS FOUND:** " + str(violations) if violations else "\n\nIndependently verified: 0 hard-constraint violations."
    if warning:
        status += f"\n\n:warning: {warning}"

    df, timeline, quality = _current_views("Current (optimized)")
    return status, df, timeline, quality


def cb_apply_toggle(faculty_label: str, days_off: list[str]):
    """Stage 3: apply/remove a faculty-unavailability override and re-solve,
    anchored to the last *optimized* schedule (never the bad baseline) so
    only the forced ripple is disrupted."""
    fid = faculty_label.split(" - ")[0].strip()
    if days_off:
        STATE["faculty_overrides"][fid] = {"unavailable_days": days_off}
    else:
        STATE["faculty_overrides"].pop(fid, None)

    prev_schedule = STATE["schedule"]
    prev_faculty = STATE["current_faculty"]
    modified_faculty = _apply_overrides(STATE["original_faculty"], STATE["faculty_overrides"])
    anchor = STATE["anchor_schedule"] or prev_schedule
    result = solve_timetable(STATIC["courses"], modified_faculty, STATIC["rooms"], anchor_schedule=anchor)

    if not result.schedule:
        status = f"**No feasible schedule found ({result.status}):** {'; '.join(result.diagnostics)}"
        df, timeline, quality = _current_views("Current")
        return status, df, timeline, quality, ""

    violations = verify_schedule(result.schedule, STATIC["courses"], modified_faculty, STATIC["rooms"])
    STATE["current_faculty"] = modified_faculty
    STATE["compare_schedule"] = prev_schedule
    STATE["schedule"] = result.schedule
    STATE["anchor_schedule"] = result.schedule
    warning = _sync_graph_and_agent(modified_faculty, result.schedule)

    n_changed = _num_changed(result.schedule, prev_schedule)
    prev_metrics = schedule_quality_metrics(prev_schedule, prev_faculty)
    new_metrics = schedule_quality_metrics(result.schedule, modified_faculty)

    status = (
        f"**Stage 3 -- constraint applied & re-optimized.** Status: {result.status} | solved in {result.solve_time_s:.2f}s | "
        f"only {n_changed}/{len(result.schedule)} section(s) had to move because of this change "
        "(anchored to the last optimized schedule -- only what's forced gets disrupted, highlighted below).\n\n"
        f"**Soft-preference quality:** morning-preference satisfied {prev_metrics['morning_preference_satisfied']}/{prev_metrics['morning_preference_total']} "
        f"→ {new_metrics['morning_preference_satisfied']}/{new_metrics['morning_preference_total']} | "
        f"busiest day {prev_metrics['busiest_day_count']} → {new_metrics['busiest_day_count']} sections"
    )
    if result.status == "FEASIBLE" and result.best_bound is not None:
        status += (
            f"\n\n_FEASIBLE (not OPTIMAL): Objective={result.objective:.0f} vs. best possible bound="
            f"{result.best_bound:.0f} (0 gap = provably optimal)._"
        )
    status += "\n\n**CONSTRAINT VIOLATIONS FOUND:** " + str(violations) if violations else "\n\nIndependently verified: 0 hard-constraint violations."
    if warning:
        status += f"\n\n:warning: {warning}"

    faculty_names = {f["faculty_id"]: f["name"] for f in STATE["original_faculty"]}
    if STATE["faculty_overrides"]:
        active = " | ".join(
            f"{faculty_names[k]} off {'/'.join(ov['unavailable_days'])}" for k, ov in STATE["faculty_overrides"].items()
        )
        overrides_text = f"**Active overrides:** {active}"
    else:
        overrides_text = ""

    df, timeline, quality = _current_views("Current")
    return status, df, timeline, quality, overrides_text


# --- Gradio callbacks: Chat tab --------------------------------------------------

def cb_chat(message: str, history: list[dict]):
    agent = get_agent()
    # `history` from gr.Chatbot(type="messages") is already [{"role":..,"content":..}, ...]
    past = [h for h in history if h["role"] in ("user", "assistant")]
    result = agent.chat(message, history=past)
    reply = result["reply"]
    if result["trace"]:
        calls = "\n".join(f"- `{t['tool']}({t['args']})`" for t in result["trace"])
        reply += f"\n\n<details><summary>{len(result['trace'])} tool call(s)</summary>\n\n{calls}\n\n</details>"
    return reply


# --- Gradio callbacks: Eligibility tab -------------------------------------------

def cb_eligibility(question: str, student_id: str):
    bot = get_eligibility_bot()
    sid = None if not student_id or student_id == "(none)" else student_id
    result = bot.answer(question, student_id=sid)
    candidates = "\n".join(f"- **[{p['id']}]** (score={p['score']:.3f}) {p['text']}" for p in result["cited_policies"])
    return f"{result['answer']}\n\n---\n**Retrieved policy candidates:**\n{candidates}"


# --- build UI ---------------------------------------------------------------------

def build_app() -> gr.Blocks:
    faculty_choices = [f"{f['faculty_id']} - {f['name']}" for f in STATE["original_faculty"]]
    student_choices = ["(none)"] + [s["student_id"] for s in STATIC["students"]]
    initial_df = _style_dataframe(_schedule_dataframe(STATE["schedule"], STATE["compare_schedule"]))
    initial_timeline = _timeline_html(STATE["schedule"], STATE["compare_schedule"])
    initial_quality = _quality_markdown(STATE["schedule"], STATE["current_faculty"], "Baseline")

    with gr.Blocks(title="Academic Timetable Copilot") as demo:
        gr.Markdown(
            "# Academic Timetable Copilot\n"
            "Real constraint solving (OR-Tools CP-SAT) for scheduling, with a graph-grounded "
            "explainability layer on top -- not an LLM guessing at a timetable."
        )

        with gr.Tab("Generated Timetable"):
            with gr.Accordion("Hard constraints vs. soft objectives -- how this schedule is produced", open=False):
                gr.Markdown(CONSTRAINTS_EXPLAINER)

            gr.Markdown(
                "**Demo flow:** ① an unoptimized-but-valid baseline is shown below → ② click **Optimize Schedule** "
                "and see exactly what improved → ③ mark a faculty member unavailable and re-solve to see a minimal, "
                "explainable ripple."
            )
            with gr.Row():
                baseline_btn = gr.Button("① Reset / Generate Unoptimized Baseline")
                optimize_btn = gr.Button("② Optimize Schedule", variant="primary")

            gr.Markdown("**③ Toggle a constraint:** mark a faculty member unavailable on given days, then re-solve.")
            with gr.Row():
                faculty_dd = gr.Dropdown(choices=faculty_choices, value=faculty_choices[0], label="Faculty member")
                days_cbg = gr.CheckboxGroup(choices=DAYS, label="Mark unavailable on")
                apply_btn = gr.Button("Apply Constraint & Re-solve")

            overrides_md = gr.Markdown("")
            status_md = gr.Markdown("")
            quality_md = gr.Markdown(value=initial_quality)

            gr.Markdown("Rows highlighted in amber changed vs. the previous step. Click any column header to sort.")
            with gr.Tabs():
                with gr.Tab("Table"):
                    schedule_table = gr.Dataframe(value=initial_df, wrap=True, label="Weekly schedule", interactive=False)
                with gr.Tab("Timeline"):
                    timeline_html = gr.HTML(value=initial_timeline)

            baseline_btn.click(
                cb_show_baseline, outputs=[status_md, schedule_table, timeline_html, quality_md, overrides_md]
            )
            optimize_btn.click(cb_optimize, outputs=[status_md, schedule_table, timeline_html, quality_md])
            apply_btn.click(
                cb_apply_toggle,
                inputs=[faculty_dd, days_cbg],
                outputs=[status_md, schedule_table, timeline_html, quality_md, overrides_md],
            )

        with gr.Tab("Ask the Copilot"):
            gr.Markdown(
                "Answers are grounded in the graph + solver -- the LLM explains and re-solves, "
                "it never invents a schedule."
            )
            gr.ChatInterface(
                fn=cb_chat,
                examples=[
                    "What else is Dr. Youssef Ahmed teaching this week?",
                    "Why is CS301 scheduled when it is?",
                    "What if Dr. Youssef Ahmed is unavailable on Tuesdays?",
                ],
            )

        with gr.Tab("Registration Eligibility Check"):
            gr.Markdown("RAG over the college's plain-text academic policies (local sentence-transformers embeddings).")
            with gr.Accordion("View the policy snippets this bot cites from", open=False):
                gr.Markdown("\n\n".join(f"**[{p['id']}]** {p['text']}" for p in STATIC["policies"]))

            with gr.Row():
                question_tb = gr.Textbox(
                    label="Ask an eligibility question",
                    value="Can I register for CS301 without having completed CS201?",
                    scale=3,
                )
                student_dd = gr.Dropdown(choices=student_choices, value="(none)", label="As student (optional)", scale=1)
            check_btn = gr.Button("Check eligibility")
            eligibility_out = gr.Markdown("")
            check_btn.click(cb_eligibility, inputs=[question_tb, student_dd], outputs=[eligibility_out])

    return demo


if __name__ == "__main__":
    import os

    app = build_app()
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
