"""Canned Cypher queries against the timetable graph, used by explain/agent.py
and available here for manual sanity-checking.

Each function takes a neo4j.Driver and returns plain Python data (list[dict]),
so the explainability agent can format them into natural language without
needing to know Cypher.
"""
from __future__ import annotations

import sys
from pathlib import Path

from neo4j import Driver

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER  # noqa: E402


def _run(driver: Driver, query: str, **params) -> list[dict]:
    with driver.session() as session:
        result = session.run(query, **params)
        return [record.data() for record in result]


def find_faculty(driver: Driver, name_or_id: str) -> list[dict]:
    """Resolve a faculty member by id or fuzzy name match."""
    return _run(
        driver,
        """
        MATCH (f:Faculty)
        WHERE f.faculty_id = $q OR toLower(f.name) CONTAINS toLower($q)
        RETURN f.faculty_id AS faculty_id, f.name AS name, f.max_hours_week AS max_hours_week,
               f.max_sections_week AS max_sections_week, f.prefers_morning AS prefers_morning,
               f.available_slots AS available_slots, f.leave_slots AS leave_slots
        """,
        q=name_or_id,
    )


def find_course(driver: Driver, course_id_or_name: str) -> list[dict]:
    return _run(
        driver,
        """
        MATCH (c:Course)
        WHERE c.course_id = $q OR toLower(c.name) CONTAINS toLower($q)
        RETURN c.course_id AS course_id, c.name AS name, c.subject AS subject,
               c.room_type_required AS room_type_required, c.is_mandatory AS is_mandatory,
               c.estimated_enrollment AS estimated_enrollment
        """,
        q=course_id_or_name,
    )


def what_else_is_faculty_teaching(driver: Driver, faculty_id: str) -> list[dict]:
    """'What else is this faculty member teaching?'"""
    return _run(
        driver,
        """
        MATCH (f:Faculty {faculty_id: $faculty_id})<-[:ASSIGNED_TO]-(sec:Section)-[:PART_OF]->(c:Course)
        MATCH (sec)-[:HELD_IN]->(r:Room)
        RETURN c.course_id AS course_id, c.name AS course_name, sec.slot_label AS slot_label,
               r.name AS room_name
        ORDER BY sec.day, sec.period
        """,
        faculty_id=faculty_id,
    )


def sections_in_room(driver: Driver, room_id: str) -> list[dict]:
    """'What sections are in this room this week?'"""
    return _run(
        driver,
        """
        MATCH (r:Room {room_id: $room_id})<-[:HELD_IN]-(sec:Section)-[:PART_OF]->(c:Course)
        MATCH (sec)-[:ASSIGNED_TO]->(f:Faculty)
        RETURN c.course_id AS course_id, c.name AS course_name, sec.slot_label AS slot_label,
               f.name AS faculty_name
        ORDER BY sec.day, sec.period
        """,
        room_id=room_id,
    )


def section_details(driver: Driver, course_id: str) -> list[dict]:
    """Full assignment detail for one scheduled section."""
    return _run(
        driver,
        """
        MATCH (sec:Section {course_id: $course_id})-[:PART_OF]->(c:Course)
        MATCH (sec)-[:ASSIGNED_TO]->(f:Faculty)
        MATCH (sec)-[:HELD_IN]->(r:Room)
        RETURN c.course_id AS course_id, c.name AS course_name, c.subject AS subject,
               c.room_type_required AS room_type_required, c.estimated_enrollment AS estimated_enrollment,
               sec.slot_label AS slot_label, sec.slot AS slot, sec.day AS day, sec.period AS period,
               f.faculty_id AS faculty_id, f.name AS faculty_name,
               f.available_slots AS faculty_available_slots, f.leave_slots AS faculty_leave_slots,
               r.room_id AS room_id, r.name AS room_name, r.capacity AS room_capacity, r.room_type AS room_type
        """,
        course_id=course_id,
    )


def qualified_faculty_for_course(driver: Driver, course_id: str) -> list[dict]:
    """Which faculty *could* have taught this course (same subject specialization)?"""
    return _run(
        driver,
        """
        MATCH (c:Course {course_id: $course_id})-[:IN_SUBJECT]->(s:Subject)<-[:QUALIFIED_IN]-(f:Faculty)
        RETURN f.faculty_id AS faculty_id, f.name AS name, f.available_slots AS available_slots,
               f.leave_slots AS leave_slots, f.max_sections_week AS max_sections_week
        """,
        course_id=course_id,
    )


def rooms_fitting_course(driver: Driver, course_id: str) -> list[dict]:
    """Which rooms *could* have hosted this course (type + capacity match)?"""
    return _run(
        driver,
        """
        MATCH (c:Course {course_id: $course_id})
        MATCH (r:Room)
        WHERE r.capacity >= c.estimated_enrollment
          AND ((c.room_type_required = 'lab' AND r.room_type = 'lab')
               OR (c.room_type_required = 'lecture' AND r.room_type IN ['lecture', 'seminar']))
        RETURN r.room_id AS room_id, r.name AS name, r.room_type AS room_type, r.capacity AS capacity
        """,
        course_id=course_id,
    )


def mandatory_courses_same_program(driver: Driver, course_id: str) -> list[dict]:
    """Other mandatory courses sharing a program with this one (i.e. courses this one
    must not clash with, per POL-3)."""
    return _run(
        driver,
        """
        MATCH (c:Course {course_id: $course_id})-[:MANDATORY_FOR]->(p:Program)<-[:MANDATORY_FOR]-(other:Course)
        WHERE other.course_id <> c.course_id
        OPTIONAL MATCH (sec:Section {course_id: other.course_id})
        RETURN DISTINCT other.course_id AS course_id, other.name AS name, p.name AS program,
               sec.slot_label AS slot_label
        """,
        course_id=course_id,
    )


def would_moving_create_conflict(driver: Driver, course_id: str, new_slot: int) -> dict:
    """'Why would moving section X to slot K create a conflict?' Checks every hard
    constraint against the hypothetical new slot and returns the reasons, if any."""
    details = section_details(driver, course_id)
    if not details:
        return {"course_id": course_id, "conflicts": [f"No such section: {course_id}"]}
    d = details[0]
    conflicts = []

    if new_slot not in (d["faculty_available_slots"] or []):
        conflicts.append(
            f"Faculty {d['faculty_name']} is not available at that slot "
            f"(on leave or outside their availability window)."
        )

    other_faculty_sections = _run(
        driver,
        """
        MATCH (f:Faculty {faculty_id: $faculty_id})<-[:ASSIGNED_TO]-(sec:Section)-[:PART_OF]->(c:Course)
        WHERE sec.slot = $slot AND c.course_id <> $course_id
        RETURN c.course_id AS course_id, c.name AS name
        """,
        faculty_id=d["faculty_id"],
        slot=new_slot,
        course_id=course_id,
    )
    if other_faculty_sections:
        conflicts.append(
            f"Faculty {d['faculty_name']} already teaches "
            f"{other_faculty_sections[0]['course_id']} at that slot."
        )

    other_room_sections = _run(
        driver,
        """
        MATCH (r:Room {room_id: $room_id})<-[:HELD_IN]-(sec:Section)-[:PART_OF]->(c:Course)
        WHERE sec.slot = $slot AND c.course_id <> $course_id
        RETURN c.course_id AS course_id, c.name AS name
        """,
        room_id=d["room_id"],
        slot=new_slot,
        course_id=course_id,
    )
    if other_room_sections:
        conflicts.append(
            f"Room {d['room_name']} is already booked for "
            f"{other_room_sections[0]['course_id']} at that slot."
        )

    program_peers = mandatory_courses_same_program(driver, course_id)
    for peer in program_peers:
        if peer.get("slot_label") is not None:
            peer_detail = section_details(driver, peer["course_id"])
            if peer_detail and peer_detail[0]["slot"] == new_slot:
                conflicts.append(
                    f"{peer['course_id']} ({peer['program']} program) is mandatory alongside "
                    f"{course_id} and is already scheduled at that slot -- students in that "
                    f"program couldn't take both."
                )

    return {"course_id": course_id, "new_slot": new_slot, "conflicts": conflicts}


def prerequisites_for(driver: Driver, course_id: str) -> list[dict]:
    return _run(
        driver,
        """
        MATCH (c:Course {course_id: $course_id})-[:REQUIRES_PREREQUISITE]->(pre:Course)
        RETURN pre.course_id AS course_id, pre.name AS name
        """,
        course_id=course_id,
    )


def full_schedule(driver: Driver) -> list[dict]:
    return _run(
        driver,
        """
        MATCH (sec:Section)-[:PART_OF]->(c:Course)
        MATCH (sec)-[:ASSIGNED_TO]->(f:Faculty)
        MATCH (sec)-[:HELD_IN]->(r:Room)
        RETURN c.course_id AS course_id, c.name AS course_name, sec.slot_label AS slot_label,
               sec.day AS day, sec.period AS period, f.name AS faculty_name, r.name AS room_name,
               sec.estimated_enrollment AS estimated_enrollment
        ORDER BY sec.day, sec.period
        """,
    )


def _demo() -> None:
    """Manual sanity check: run each canned query once and print results."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    print("=== full_schedule (first 5) ===")
    for row in full_schedule(driver)[:5]:
        print(" ", row)

    sample_course = full_schedule(driver)[0]["course_id"]
    print(f"\n=== section_details({sample_course}) ===")
    details = section_details(driver, sample_course)
    print(" ", details[0] if details else "NONE")

    fid = details[0]["faculty_id"]
    print(f"\n=== what_else_is_faculty_teaching({fid}) ===")
    for row in what_else_is_faculty_teaching(driver, fid):
        print(" ", row)

    rid = details[0]["room_id"]
    print(f"\n=== sections_in_room({rid}) ===")
    for row in sections_in_room(driver, rid):
        print(" ", row)

    print(f"\n=== would_moving_create_conflict({sample_course}, slot=0) ===")
    print(" ", would_moving_create_conflict(driver, sample_course, 0))

    driver.close()


if __name__ == "__main__":
    _demo()
