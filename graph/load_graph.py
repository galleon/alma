"""Loads synthetic data + the solver's schedule output into Neo4j.

Models: Student, Program, Faculty, Course, Section, Room, Policy as nodes,
connected by relationships that mirror how a real registrar would reason
about scheduling:

  (Student)-[:ENROLLED_IN]->(Program)
  (Student)-[:COMPLETED]->(Course)
  (Course)-[:REQUIRES_PREREQUISITE]->(Course)
  (Course)-[:MANDATORY_FOR]->(Program)
  (Section)-[:PART_OF]->(Course)
  (Section)-[:ASSIGNED_TO]->(Faculty)
  (Section)-[:HELD_IN]->(Room)
  (Faculty)-[:QUALIFIED_IN]->(Subject)
  (Course)-[:IN_SUBJECT]->(Subject)

Usage:
    uv run python graph/load_graph.py            # wipes and reloads everything
    uv run python graph/load_graph.py --no-wipe   # loads without clearing first
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, PATHS, slot_label  # noqa: E402


def _load_json(path: Path):
    return json.loads(path.read_text())


def wipe(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def create_constraints(tx):
    for label, prop in [
        ("Course", "course_id"),
        ("Section", "course_id"),
        ("Faculty", "faculty_id"),
        ("Room", "room_id"),
        ("Student", "student_id"),
        ("Program", "name"),
        ("Subject", "name"),
        ("Policy", "id"),
    ]:
        tx.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")


def load_subjects_and_programs(tx, courses, programs):
    subjects = sorted({c["subject"] for c in courses})
    for s in subjects:
        tx.run("MERGE (:Subject {name: $name})", name=s)
    for prog_name in programs:
        tx.run("MERGE (:Program {name: $name})", name=prog_name)


def load_courses(tx, courses):
    for c in courses:
        tx.run(
            """
            MERGE (co:Course {course_id: $course_id})
            SET co.name = $name,
                co.subject = $subject,
                co.credit_hours = $credit_hours,
                co.room_type_required = $room_type_required,
                co.is_mandatory = $is_mandatory,
                co.estimated_enrollment = $estimated_enrollment
            WITH co
            MATCH (s:Subject {name: $subject})
            MERGE (co)-[:IN_SUBJECT]->(s)
            """,
            **{
                "course_id": c["course_id"],
                "name": c["name"],
                "subject": c["subject"],
                "credit_hours": c["credit_hours"],
                "room_type_required": c["room_type_required"],
                "is_mandatory": c["is_mandatory"],
                "estimated_enrollment": c["estimated_enrollment"],
            },
        )
        for prog in c["mandatory_for"]:
            tx.run(
                """
                MATCH (co:Course {course_id: $course_id}), (p:Program {name: $prog})
                MERGE (co)-[:MANDATORY_FOR]->(p)
                """,
                course_id=c["course_id"],
                prog=prog,
            )
        for prereq in c["prerequisites"]:
            tx.run(
                """
                MATCH (co:Course {course_id: $course_id}), (pre:Course {course_id: $prereq})
                MERGE (co)-[:REQUIRES_PREREQUISITE]->(pre)
                """,
                course_id=c["course_id"],
                prereq=prereq,
            )


def load_faculty(tx, faculty):
    for f in faculty:
        tx.run(
            """
            MERGE (fac:Faculty {faculty_id: $faculty_id})
            SET fac.name = $name,
                fac.max_hours_week = $max_hours_week,
                fac.max_sections_week = $max_sections_week,
                fac.prefers_morning = $prefers_morning,
                fac.available_slots = $available_slots,
                fac.leave_slots = $leave_slots
            """,
            **{
                "faculty_id": f["faculty_id"],
                "name": f["name"],
                "max_hours_week": f["max_hours_week"],
                "max_sections_week": f["max_sections_week"],
                "prefers_morning": f["prefers_morning"],
                "available_slots": f["available_slots"],
                "leave_slots": f["leave_slots"],
            },
        )
        for subj in f["specializations"]:
            tx.run(
                """
                MATCH (fac:Faculty {faculty_id: $faculty_id}), (s:Subject {name: $subj})
                MERGE (fac)-[:QUALIFIED_IN]->(s)
                """,
                faculty_id=f["faculty_id"],
                subj=subj,
            )


def load_rooms(tx, rooms):
    for r in rooms:
        tx.run(
            """
            MERGE (rm:Room {room_id: $room_id})
            SET rm.name = $name, rm.room_type = $room_type, rm.capacity = $capacity
            """,
            **r,
        )


def load_students(tx, students):
    for s in students:
        tx.run(
            """
            MERGE (st:Student {student_id: $student_id})
            SET st.name = $name, st.year = $year
            WITH st
            MATCH (p:Program {name: $program})
            MERGE (st)-[:ENROLLED_IN]->(p)
            """,
            student_id=s["student_id"],
            name=s["name"],
            year=s["year"],
            program=s["program"],
        )
        for cid in s["completed_courses"]:
            tx.run(
                """
                MATCH (st:Student {student_id: $student_id}), (co:Course {course_id: $course_id})
                MERGE (st)-[:COMPLETED]->(co)
                """,
                student_id=s["student_id"],
                course_id=cid,
            )


def load_policies(tx, policies):
    for p in policies:
        tx.run("MERGE (pol:Policy {id: $id}) SET pol.text = $text", id=p["id"], text=p["text"])


def load_schedule(tx, schedule):
    for cid, entry in schedule.items():
        tx.run(
            """
            MERGE (sec:Section {course_id: $course_id})
            SET sec.slot = $slot,
                sec.day = $day,
                sec.period = $period,
                sec.slot_label = $slot_label,
                sec.estimated_enrollment = $estimated_enrollment
            WITH sec
            MATCH (co:Course {course_id: $course_id})
            MERGE (sec)-[:PART_OF]->(co)
            WITH sec
            MATCH (fac:Faculty {faculty_id: $faculty_id})
            MERGE (sec)-[:ASSIGNED_TO]->(fac)
            WITH sec
            MATCH (rm:Room {room_id: $room_id})
            MERGE (sec)-[:HELD_IN]->(rm)
            """,
            **{
                "course_id": cid,
                "slot": entry["slot"],
                "day": entry["day"],
                "period": entry["period"],
                "slot_label": entry["slot_label"],
                "estimated_enrollment": entry["estimated_enrollment"],
                "faculty_id": entry["faculty_id"],
                "room_id": entry["room_id"],
            },
        )


def load_all(driver, wipe_first: bool = True) -> dict[str, int]:
    courses = _load_json(PATHS.courses)
    faculty = _load_json(PATHS.faculty)
    rooms = _load_json(PATHS.rooms)
    students = _load_json(PATHS.students)
    policies = _load_json(PATHS.policies)
    programs = _load_json(PATHS.students.parent / "programs.json")
    schedule = _load_json(PATHS.schedule) if PATHS.schedule.exists() else {}

    with driver.session() as session:
        if wipe_first:
            session.execute_write(wipe)
        session.execute_write(create_constraints)
        session.execute_write(load_subjects_and_programs, courses, programs)
        session.execute_write(load_courses, courses)
        session.execute_write(load_faculty, faculty)
        session.execute_write(load_rooms, rooms)
        session.execute_write(load_students, students)
        session.execute_write(load_policies, policies)
        if schedule:
            session.execute_write(load_schedule, schedule)

    return {
        "courses": len(courses),
        "faculty": len(faculty),
        "rooms": len(rooms),
        "students": len(students),
        "policies": len(policies),
        "sections": len(schedule),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-wipe", action="store_true", help="Don't clear the graph before loading")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"Could not connect to Neo4j at {NEO4J_URI}: {e}", file=sys.stderr)
        print("Is it running? Try: docker compose up -d", file=sys.stderr)
        sys.exit(1)

    counts = load_all(driver, wipe_first=not args.no_wipe)
    driver.close()

    print("Loaded into Neo4j:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    if counts["sections"] == 0:
        print("\nNote: no schedule found. Run solver/timetable_solver.py first to load Section nodes.")


if __name__ == "__main__":
    main()
