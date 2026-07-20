"""Synthetic data generator for the Academic Timetable Copilot PoC.

Generates one small, fictional college's worth of scheduling data:
20 courses, 10 faculty, 5 rooms, 300 students, and a handful of plain-text
academic policies. Everything is synthetic -- no real institutional data.

Usage:
    uv run python data/generate_synthetic_data.py --seed 42
    uv run python data/generate_synthetic_data.py --seed 7 --out-dir data
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_SEED, NUM_SLOTS, PERIODS_PER_DAY, slot_index  # noqa: E402

SUBJECTS = ["CS", "MATH", "PHYS", "BIO", "CHEM", "HUM"]

# (course_id, name, subject, credit_hours, room_type, prereqs, mandatory_for programs)
COURSE_DEFS = [
    ("CS101", "Intro to Computer Science", "CS", 3, "lecture", [], ["Computer Science"]),
    ("CS201", "Data Structures", "CS", 4, "lecture", ["CS101"], ["Computer Science"]),
    ("CS301", "Algorithms", "CS", 4, "lecture", ["CS201"], ["Computer Science"]),
    ("CS310", "Systems Programming Lab", "CS", 3, "lab", ["CS201"], []),
    ("MATH101", "Calculus I", "MATH", 4, "lecture", [], ["Computer Science", "Physics"]),
    ("MATH201", "Linear Algebra", "MATH", 3, "lecture", ["MATH101"], ["Computer Science", "Physics"]),
    ("MATH301", "Discrete Mathematics", "MATH", 3, "lecture", ["MATH101"], ["Computer Science"]),
    ("MATH210", "Statistics", "MATH", 3, "lecture", ["MATH101"], []),
    ("PHYS101", "Physics I: Mechanics", "PHYS", 4, "lecture", ["MATH101"], ["Physics"]),
    ("PHYS102", "Physics I Lab", "PHYS", 1, "lab", [], ["Physics"]),
    ("PHYS201", "Physics II: Electromagnetism", "PHYS", 4, "lecture", ["PHYS101"], ["Physics"]),
    ("BIO101", "Introduction to Biology", "BIO", 3, "lecture", [], ["Biology"]),
    ("BIO102", "Biology Lab", "BIO", 1, "lab", [], ["Biology"]),
    ("BIO201", "Genetics", "BIO", 3, "lecture", ["BIO101"], ["Biology"]),
    ("CHEM101", "General Chemistry", "CHEM", 4, "lecture", [], ["Biology"]),
    ("CHEM102", "Chemistry Lab", "CHEM", 1, "lab", [], []),
    ("CHEM201", "Organic Chemistry", "CHEM", 4, "lecture", ["CHEM101"], []),
    ("ENG101", "Academic Writing", "HUM", 3, "lecture", [], ["Computer Science", "Physics", "Biology"]),
    ("HIST101", "World History", "HUM", 3, "lecture", [], []),
    ("PHIL101", "Ethics", "HUM", 3, "lecture", [], []),
]

PROGRAMS = {
    "Computer Science": {
        "mandatory": ["CS101", "CS201", "CS301", "MATH101", "MATH201", "MATH301", "ENG101"],
        "share": 0.43,
    },
    "Physics": {
        "mandatory": ["PHYS101", "PHYS102", "PHYS201", "MATH101", "MATH201", "ENG101"],
        "share": 0.30,
    },
    "Biology": {
        "mandatory": ["BIO101", "BIO102", "BIO201", "CHEM101", "ENG101"],
        "share": 0.27,
    },
}

ELECTIVES = ["CS310", "MATH210", "CHEM102", "CHEM201", "HIST101", "PHIL101"]

FACULTY_NAMES = [
    "Dr. Amara Okafor", "Dr. Youssef Ahmed", "Dr. Lena Kowalski", "Dr. Raj Patel",
    "Dr. Maria Santos", "Dr. Wei Zhang", "Dr. Fatima Al-Sayed", "Dr. Tom O'Brien",
    "Dr. Priya Nair", "Dr. Carlos Mendes",
]

FIRST_NAMES = [
    "Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn",
    "Nina", "Omar", "Zara", "Leo", "Mia", "Ivan", "Sofia", "Noah", "Grace", "Kai",
]
LAST_NAMES = [
    "Johnson", "Williams", "Brown", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Chen", "Kim", "Silva", "Novak", "Haddad", "Ibrahim", "Petrov", "Andersen", "Costa", "Lund",
]

ROOM_DEFS = [
    ("R1", "Lecture Hall A", "lecture", 100),
    ("R2", "Lecture Hall B", "lecture", 60),
    ("R3", "Seminar Room", "seminar", 20),
    ("R4", "Lab 1", "lab", 30),
    ("R5", "Lab 2", "lab", 24),
]

POLICIES = [
    {
        "id": "POL-1",
        "text": "No faculty member may teach more than 12 hours per week of scheduled sections.",
    },
    {
        "id": "POL-2",
        "text": "Lab sections must be held in lab-type rooms; lecture sections may be held in "
                "lecture halls or seminar rooms, provided room capacity is sufficient.",
    },
    {
        "id": "POL-3",
        "text": "No student may be scheduled for two sections at the same day/time block. In "
                "particular, mandatory courses within the same program and year must never overlap.",
    },
    {
        "id": "POL-4",
        "text": "A student may not register for a course before completing all of its listed "
                "prerequisite courses.",
    },
    {
        "id": "POL-5",
        "text": "The room assigned to a section must have capacity greater than or equal to the "
                "section's estimated enrollment.",
    },
    {
        "id": "POL-6",
        "text": "Faculty members may only be assigned to teach courses within their declared "
                "subject specialization(s).",
    },
]


def build_courses() -> list[dict]:
    courses = []
    for course_id, name, subject, credits, room_type, prereqs, mandatory_for in COURSE_DEFS:
        courses.append(
            {
                "course_id": course_id,
                "name": name,
                "subject": subject,
                "credit_hours": credits,
                "room_type_required": room_type,
                "prerequisites": prereqs,
                "mandatory_for": mandatory_for,
                "is_mandatory": len(mandatory_for) > 0,
            }
        )
    return courses


def build_faculty(rng: random.Random) -> list[dict]:
    # Guarantee every subject has at least 2 distinct qualified faculty.
    assignments: dict[int, set[str]] = {i: set() for i in range(len(FACULTY_NAMES))}
    for subj in SUBJECTS:
        for i in rng.sample(range(len(FACULTY_NAMES)), 2):
            assignments[i].add(subj)

    faculty = []
    for i, name in enumerate(FACULTY_NAMES):
        specializations = sorted(assignments[i]) or [rng.choice(SUBJECTS)]
        max_hours = rng.choice([6.0, 7.5, 9.0, 10.5, 12.0])

        # Available on 4 of 5 days; the 5th day off. Within available days, 0-2 leave slots.
        days_off = rng.sample(range(5), 1)
        available_slots = [
            slot_index(d, p)
            for d in range(5)
            if d not in days_off
            for p in range(PERIODS_PER_DAY)
        ]
        num_leave = rng.randint(0, 2)
        leave_slots = sorted(rng.sample(available_slots, num_leave)) if num_leave else []
        available_slots = sorted(set(available_slots) - set(leave_slots))

        faculty.append(
            {
                "faculty_id": f"F{i + 1}",
                "name": name,
                "specializations": specializations,
                "max_hours_week": max_hours,
                "max_sections_week": int(max_hours // 1.5),
                "available_slots": available_slots,
                "leave_slots": leave_slots,
                "prefers_morning": rng.random() < 0.5,
            }
        )
    return faculty


def build_rooms() -> list[dict]:
    return [
        {"room_id": rid, "name": name, "room_type": rtype, "capacity": cap}
        for rid, name, rtype, cap in ROOM_DEFS
    ]


def build_students(rng: random.Random, n_students: int) -> list[dict]:
    course_by_id = {c[0]: c for c in COURSE_DEFS}
    program_names = list(PROGRAMS.keys())
    shares = [PROGRAMS[p]["share"] for p in program_names]

    students = []
    for i in range(n_students):
        program = rng.choices(program_names, weights=shares, k=1)[0]
        year = rng.choices([1, 2, 3, 4], weights=[0.30, 0.27, 0.24, 0.19], k=1)[0]

        mandatory_sequence = PROGRAMS[program]["mandatory"]
        # Roughly (year - 1) mandatory courses completed already, respecting prereq order
        # as authored in COURSE_DEFS (earlier entries are prerequisite-safe first).
        n_completed = min(len(mandatory_sequence), max(0, (year - 1) * 2 + rng.randint(-1, 1)))
        completed = mandatory_sequence[:n_completed]

        # A student may also have completed 0-2 electives with satisfied prereqs.
        possible_electives = [
            e for e in ELECTIVES
            if all(p in completed for p in course_by_id[e][5])
        ]
        n_elective = rng.randint(0, min(2, len(possible_electives)))
        completed = completed + rng.sample(possible_electives, n_elective) if possible_electives else completed

        students.append(
            {
                "student_id": f"S{i + 1:04d}",
                "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
                "program": program,
                "year": year,
                "completed_courses": completed,
            }
        )
    return students


def estimate_enrollments(courses: list[dict], students: list[dict], rooms: list[dict]) -> None:
    """Attach an `estimated_enrollment` field to each course, capped to fit an available
    room of the required type so the dataset is solvable by construction."""
    max_capacity_by_type = {"lecture": 0, "lab": 0}
    for r in rooms:
        t = "lecture" if r["room_type"] in ("lecture", "seminar") else "lab"
        max_capacity_by_type[t] = max(max_capacity_by_type[t], r["capacity"])

    for course in courses:
        cid = course["course_id"]
        if course["is_mandatory"]:
            pool = [s for s in students if course["mandatory_for"] and s["program"] in course["mandatory_for"]]
            # Only students who haven't already completed it are "currently enrolled".
            pool = [s for s in pool if cid not in s["completed_courses"]]
            estimate = max(5, len(pool) // 4 if course["subject"] != "HUM" else len(pool) // 3)
        else:
            eligible = [
                s for s in students
                if all(p in s["completed_courses"] for p in course["prerequisites"])
            ]
            estimate = max(8, int(len(eligible) * 0.12))

        cap = max_capacity_by_type["lab" if course["room_type_required"] == "lab" else "lecture"]
        course["estimated_enrollment"] = min(estimate, cap)


def sanity_check(courses, faculty, rooms, students, policies) -> list[str]:
    problems = []
    if len(courses) != 20:
        problems.append(f"expected 20 courses, got {len(courses)}")
    if len(faculty) != 10:
        problems.append(f"expected 10 faculty, got {len(faculty)}")
    if len(rooms) != 5:
        problems.append(f"expected 5 rooms, got {len(rooms)}")
    if len(students) != 300:
        problems.append(f"expected 300 students, got {len(students)}")
    if not (4 <= len(policies) <= 6):
        problems.append(f"expected 4-6 policies, got {len(policies)}")

    course_ids = {c["course_id"] for c in courses}
    for c in courses:
        for p in c["prerequisites"]:
            if p not in course_ids:
                problems.append(f"{c['course_id']} lists unknown prerequisite {p}")

    subjects_covered = {s for f in faculty for s in f["specializations"]}
    for subj in SUBJECTS:
        n_qualified = sum(1 for f in faculty if subj in f["specializations"])
        if n_qualified < 2:
            problems.append(f"subject {subj} has only {n_qualified} qualified faculty (need >=2)")

    room_types = {r["room_type"] for r in rooms}
    if "lab" not in room_types:
        problems.append("no lab-type room defined")

    for c in courses:
        need_type = "lab" if c["room_type_required"] == "lab" else ("lecture", "seminar")
        ok = any(
            r["room_type"] == need_type if isinstance(need_type, str) else r["room_type"] in need_type
            for r in rooms
            for _ in [0]
            if r["capacity"] >= c["estimated_enrollment"]
        )
        if not ok:
            problems.append(
                f"{c['course_id']} (needs {c['room_type_required']}, "
                f"est. enrollment {c['estimated_enrollment']}) has no room that fits"
            )

    for c in courses:
        n_qualified = sum(1 for f in faculty if c["subject"] in f["specializations"])
        if n_qualified == 0:
            problems.append(f"{c['course_id']} ({c['subject']}) has no qualified faculty at all")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--n-students", type=int, default=300)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    courses = build_courses()
    faculty = build_faculty(rng)
    rooms = build_rooms()
    students = build_students(rng, args.n_students)
    estimate_enrollments(courses, students, rooms)
    policies = POLICIES

    problems = sanity_check(courses, faculty, rooms, students, policies)
    if problems:
        print("SANITY CHECK FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "courses.json": courses,
        "faculty.json": faculty,
        "rooms.json": rooms,
        "students.json": students,
        "policies.json": policies,
        "programs.json": PROGRAMS,
    }
    for fname, payload in out.items():
        path = args.out_dir / fname
        path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {path} ({len(payload)} top-level items)")

    print(f"\nSanity check passed. Seed={args.seed}. "
          f"{len(courses)} courses, {len(faculty)} faculty, {len(rooms)} rooms, "
          f"{len(students)} students, {len(policies)} policies.")


if __name__ == "__main__":
    main()
