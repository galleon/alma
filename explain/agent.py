"""Explainability agent for the Academic Timetable Copilot.

This agent never invents a schedule. It only:
  (a) answers "why" questions by querying the Neo4j graph that mirrors the
      solver's decisions, citing the specific hard/soft constraint or policy
      snippet that drove the outcome, and
  (b) on request, re-runs the CP-SAT solver with a modified constraint
      (e.g. a faculty member going on leave) and diffs the new schedule
      against the current one, explaining what changed and why.

It uses plain OpenAI-style function calling (works against local Ollama by
default, or a real OpenAI-compatible API if OPENAI_API_KEY is set) -- the LLM
is only ever a natural-language front end over tools backed by the graph and
the solver. It never guesses a schedule itself.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    PATHS,
    all_slot_labels,
    parse_slot_label,
)
from graph import queries as gq  # noqa: E402
from solver.timetable_solver import solve_timetable, verify_schedule  # noqa: E402

DAY_NAME_TO_IDX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4}

SYSTEM_PROMPT = f"""You are the Academic Timetable Copilot's explainability assistant.

CRITICAL: The timetable was produced by a constraint solver (Google OR-Tools CP-SAT),
not by you. You never invent, guess, or alter schedule facts. Every factual claim
about the schedule (who teaches what, which room, which slot) must come from a tool
call result -- never from memory or assumption. If a tool hasn't given you a fact,
call a tool to get it, or say you don't know.

When explaining *why* something is scheduled the way it is, ground your answer in:
  1. the concrete data returned by tools (faculty availability, room capacity, etc.), and
  2. the specific policy below that the constraint enforces (cite it by ID and quote it).

Academic policies in force (these map directly to the solver's hard constraints):
{chr(10).join(f"  [{p['id']}] {p['text']}" for p in json.loads(PATHS.policies.read_text()))}

When a user asks a hypothetical / "what if" question (e.g. "what if Professor X is
unavailable on Tuesdays?"), use the resolve_what_if tool to actually re-run the solver
-- do not guess the outcome yourself. Then explain the diff it returns: which sections
moved, to where, and why (which constraint forced the change). Sections not mentioned
in the diff did not change.

If the scenario involves more than one faculty member at once (e.g. "what if X and Y
are both unavailable Tuesday"), you MUST put every affected faculty member in the
`changes` list of a single resolve_what_if call. Calling the tool once per faculty
member does NOT combine the hypotheticals -- each call only ever sees the real current
schedule, so two separate single-faculty calls give you two unrelated single-person
scenarios, never the combined one.

Be concise and concrete. Prefer citing exact course IDs, faculty names, room names,
and slot labels over vague language.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_faculty",
            "description": "Look up a faculty member by id (e.g. 'F3') or by (partial) name.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_course",
            "description": "Look up a course by id (e.g. 'CS301') or by (partial) name.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_section_details",
            "description": "Get the full scheduled assignment for a course section: faculty, "
            "room, slot, capacity, and the faculty's availability/leave (useful for explaining "
            "why it was scheduled where it was).",
            "parameters": {
                "type": "object",
                "properties": {"course_id": {"type": "string"}},
                "required": ["course_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "what_else_is_faculty_teaching",
            "description": "List all other sections a faculty member teaches this week.",
            "parameters": {
                "type": "object",
                "properties": {"faculty_id": {"type": "string"}},
                "required": ["faculty_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sections_in_room",
            "description": "List all sections held in a given room this week.",
            "parameters": {
                "type": "object",
                "properties": {"room_id": {"type": "string"}},
                "required": ["room_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "qualified_faculty_for_course",
            "description": "List faculty who are subject-qualified to teach a given course "
            "(whether or not they were actually assigned), with their availability.",
            "parameters": {
                "type": "object",
                "properties": {"course_id": {"type": "string"}},
                "required": ["course_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "would_moving_create_conflict",
            "description": "Check whether moving a course's section to a different slot would "
            "violate any hard constraint (faculty availability/double-booking, room double-booking, "
            "or a same-program mandatory-course clash). Returns the list of concrete conflicts, if any.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_id": {"type": "string"},
                    "new_slot_label": {
                        "type": "string",
                        "description": "One of: " + ", ".join(all_slot_labels()),
                    },
                },
                "required": ["course_id", "new_slot_label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_what_if",
            "description": "Re-run the CP-SAT solver with one or more modified faculty constraints "
            "and diff the result against the current schedule. Use this for any 'what if' question "
            "about faculty availability or teaching load. IMPORTANT: if the scenario involves more "
            "than one faculty member (e.g. 'what if X and Y are both unavailable Tuesday'), put ALL "
            "of them in the `changes` list of a SINGLE call -- calling this tool once per faculty "
            "member does NOT combine the hypotheticals, since each call only ever sees the real "
            "current schedule and has no memory of other calls. Does not permanently change the live "
            "schedule -- it's a hypothetical exploration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "changes": {
                        "type": "array",
                        "description": "One entry per faculty member affected by this hypothetical "
                        "(a single-faculty scenario is just a one-element list).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "faculty_id": {"type": "string", "description": "e.g. 'F3'"},
                                "unavailable_days": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": ["Mon", "Tue", "Wed", "Thu", "Fri"]},
                                    "description": "Days this faculty member should become fully unavailable.",
                                },
                                "max_hours_week": {
                                    "type": "number",
                                    "description": "If given, overrides this faculty member's max weekly teaching hours.",
                                },
                            },
                            "required": ["faculty_id"],
                        },
                    },
                },
                "required": ["changes"],
            },
        },
    },
]


class TimetableAgent:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self.courses = json.loads(PATHS.courses.read_text())
        self.faculty = json.loads(PATHS.faculty.read_text())
        self.rooms = json.loads(PATHS.rooms.read_text())
        self.baseline_schedule = json.loads(PATHS.schedule.read_text()) if PATHS.schedule.exists() else {}

    def close(self):
        self.driver.close()

    def reload_baseline(self):
        """Call after the UI regenerates/edits the live schedule so the agent's
        'current schedule' stays in sync."""
        self.faculty = json.loads(PATHS.faculty.read_text())
        self.baseline_schedule = json.loads(PATHS.schedule.read_text()) if PATHS.schedule.exists() else {}

    # --- tool implementations ---

    def _tool_find_faculty(self, query: str) -> list[dict]:
        return gq.find_faculty(self.driver, query)

    def _tool_find_course(self, query: str) -> list[dict]:
        return gq.find_course(self.driver, query)

    def _tool_get_section_details(self, course_id: str) -> list[dict]:
        return gq.section_details(self.driver, course_id)

    def _tool_what_else_is_faculty_teaching(self, faculty_id: str) -> list[dict]:
        return gq.what_else_is_faculty_teaching(self.driver, faculty_id)

    def _tool_sections_in_room(self, room_id: str) -> list[dict]:
        return gq.sections_in_room(self.driver, room_id)

    def _tool_qualified_faculty_for_course(self, course_id: str) -> list[dict]:
        return gq.qualified_faculty_for_course(self.driver, course_id)

    def _tool_would_moving_create_conflict(self, course_id: str, new_slot_label: str) -> dict:
        slot = parse_slot_label(new_slot_label)
        if slot is None:
            return {"error": f"'{new_slot_label}' is not a recognized slot label."}
        return gq.would_moving_create_conflict(self.driver, course_id, slot)

    def _tool_resolve_what_if(self, changes: list[dict]) -> dict:
        if not changes:
            return {"error": "No changes given."}

        modified_faculty = copy.deepcopy(self.faculty)
        faculty_by_id = {f["faculty_id"]: f for f in modified_faculty}
        applied = []

        for change in changes:
            faculty_id = change.get("faculty_id")
            if faculty_id not in faculty_by_id:
                matches = gq.find_faculty(self.driver, faculty_id)
                if not matches:
                    return {"error": f"No faculty found matching '{faculty_id}'."}
                faculty_id = matches[0]["faculty_id"]

            target = faculty_by_id[faculty_id]
            original_available = set(target["available_slots"])

            unavailable_days = change.get("unavailable_days") or []
            day_idxs = {DAY_NAME_TO_IDX[d.strip().lower()[:3]] for d in unavailable_days}
            if day_idxs:
                removed = {s for s in target["available_slots"] if s // 6 in day_idxs}
                target["available_slots"] = sorted(set(target["available_slots"]) - removed)
                target["leave_slots"] = sorted(set(target["leave_slots"]) | removed)

            max_hours_week = change.get("max_hours_week")
            if max_hours_week is not None:
                target["max_hours_week"] = max_hours_week
                target["max_sections_week"] = int(max_hours_week // 1.5)

            if set(target["available_slots"]) == original_available and max_hours_week is None:
                return {"error": f"No change was actually applied for {faculty_id} -- check the day names or hours given."}

            applied.append(
                {"faculty_id": faculty_id, "faculty_name": target["name"], "unavailable_days": unavailable_days, "max_hours_week": max_hours_week}
            )

        result = solve_timetable(
            self.courses, modified_faculty, self.rooms, anchor_schedule=self.baseline_schedule
        )
        if not result.schedule:
            return {
                "applied_changes": applied,
                "status": result.status,
                "feasible": False,
                "diagnostics": result.diagnostics,
                "message": "No feasible schedule exists under this hypothetical constraint.",
            }

        violations = verify_schedule(result.schedule, self.courses, modified_faculty, self.rooms)

        diffs = []
        for cid, new_entry in result.schedule.items():
            old_entry = self.baseline_schedule.get(cid)
            if old_entry is None:
                continue
            changed_fields = {}
            for field_name in ("slot_label", "faculty_id", "faculty_name", "room_id", "room_name"):
                if old_entry.get(field_name) != new_entry.get(field_name):
                    changed_fields[field_name] = {"before": old_entry.get(field_name), "after": new_entry.get(field_name)}
            if changed_fields:
                diffs.append({"course_id": cid, "changes": changed_fields})

        # Deliberately not including the full hypothetical schedule here: it's
        # ~20 sections of noise the model has to wade through for no benefit
        # over `diffs` (which already says exactly what changed), and bloating
        # the tool result increases the odds of the small local model losing
        # the thread and returning an empty final answer.
        return {
            "applied_changes": applied,
            "status": result.status,
            "feasible": True,
            "solve_time_s": round(result.solve_time_s, 3),
            "constraint_violations": violations,
            "num_sections_changed": len(diffs),
            "diffs": diffs,
        }

    def _dispatch(self, name: str, args: dict) -> Any:
        fn = getattr(self, f"_tool_{name}", None)
        if fn is None:
            return {"error": f"Unknown tool {name}"}
        try:
            return fn(**args)
        except Exception as e:  # keep the agent loop alive on bad tool args
            return {"error": str(e)}

    def chat(self, user_message: str, history: list[dict] | None = None, max_tool_rounds: int = 6) -> dict:
        """Runs the function-calling loop. Returns {"reply": str, "trace": [...]}
        where trace records every tool call made, for UI transparency."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += history or []
        messages.append({"role": "user", "content": user_message})

        trace = []
        for _ in range(max_tool_rounds):
            msg = self._complete_with_retry(messages)
            if not msg.tool_calls:
                reply = (msg.content or "").strip()
                if not reply:
                    reply = self._fallback_reply(trace)
                return {"reply": reply, "trace": trace}

            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = self._dispatch(tc.function.name, args)
                trace.append({"tool": tc.function.name, "args": args, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str)[:8000],
                    }
                )

        return {"reply": "I ran out of tool-call budget answering this -- try a more specific question.", "trace": trace}

    def _complete_with_retry(self, messages: list[dict], max_attempts: int = 3):
        """Small local models occasionally return an empty final message
        (finish_reason='stop', no content, no tool_calls) after a round of
        tool calls -- a sampling quirk, not a real "no answer". Resampling
        almost always produces a real answer on the next try."""
        last_msg = None
        for _ in range(max_attempts):
            resp = self.client.chat.completions.create(model=LLM_MODEL, messages=messages, tools=TOOLS)
            last_msg = resp.choices[0].message
            if last_msg.tool_calls or (last_msg.content or "").strip():
                return last_msg
        return last_msg

    def _fallback_reply(self, trace: list[dict]) -> str:
        """Used only if every retry still came back empty -- summarize the
        last tool result directly rather than showing the user nothing."""
        if not trace:
            return "I wasn't able to generate an answer -- please try rephrasing the question."
        last = trace[-1]
        return (
            "The model didn't produce a written answer this time, but here's the raw result of its "
            f"last lookup (`{last['tool']}`), which should answer the question:\n\n"
            f"```json\n{json.dumps(last['result'], default=str, indent=2)[:2000]}\n```"
        )


def main() -> None:
    """Quick CLI smoke test with a handful of sample questions."""
    agent = TimetableAgent()
    sample_questions = [
        "What section does Dr. Youssef Ahmed teach, and what else is he teaching?",
        "Why is CS301 scheduled when it is? What room and faculty member is it assigned to?",
        "What if Dr. Youssef Ahmed is unavailable on Tuesdays -- what would change?",
    ]
    for q in sample_questions:
        print(f"\n{'=' * 70}\nQ: {q}")
        result = agent.chat(q)
        print(f"\n[{len(result['trace'])} tool call(s)]")
        for t in result["trace"]:
            print(f"  -> {t['tool']}({t['args']})")
        print(f"\nA: {result['reply']}")
    agent.close()


if __name__ == "__main__":
    main()
