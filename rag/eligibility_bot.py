"""Minimal RAG over the college's plain-text academic policies, for answering
student registration-eligibility questions (e.g. "can I register for CS301
without completing CS201?"). Uses local sentence-transformers embeddings --
no paid embedding API required.

Prerequisite facts are checked deterministically against courses.json /
students.json; the RAG layer's job is to retrieve and cite the specific
policy text that backs the answer, not to guess the prerequisite chain.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS  # noqa: E402

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class EligibilityBot:
    def __init__(self):
        self.courses = json.loads(PATHS.courses.read_text())
        self.students = json.loads(PATHS.students.read_text())
        self.policies = json.loads(PATHS.policies.read_text())
        self.course_by_id = {c["course_id"]: c for c in self.courses}
        self.student_by_id = {s["student_id"]: s for s in self.students}

        self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        policy_texts = [p["text"] for p in self.policies]
        self._policy_embeddings = self._model.encode(policy_texts, normalize_embeddings=True)

        # Sort course ids longest-first so e.g. "CS310" isn't shadowed by "CS3".
        self._course_id_pattern = re.compile(
            r"\b(" + "|".join(sorted(self.course_by_id, key=len, reverse=True)) + r")\b",
            re.IGNORECASE,
        )

    def _retrieve_policies(self, question: str, top_k: int = 2) -> list[dict]:
        q_emb = self._model.encode([question], normalize_embeddings=True)[0]
        scores = self._policy_embeddings @ q_emb
        top_idx = np.argsort(-scores)[:top_k]
        return [{**self.policies[i], "score": float(scores[i])} for i in top_idx]

    def _mentioned_course_ids(self, question: str) -> list[str]:
        """Course ids in order of first appearance (the first mention is treated
        as the course the question is actually about)."""
        seen: list[str] = []
        for m in self._course_id_pattern.findall(question):
            cid = m.upper()
            if cid not in seen:
                seen.append(cid)
        return seen

    def answer(self, question: str, student_id: str | None = None) -> dict:
        cited_policies = self._retrieve_policies(question)
        mentioned = self._mentioned_course_ids(question)

        student = self.student_by_id.get(student_id) if student_id else None
        completed = set(student["completed_courses"]) if student else None

        # Deterministic prerequisite check when a known course is mentioned.
        if mentioned:
            target_id = mentioned[0]
            course = self.course_by_id[target_id]
            prereqs = course["prerequisites"]

            if not prereqs:
                verdict = f"Yes -- {target_id} ({course['name']}) has no prerequisites."
            elif student is not None:
                missing = [p for p in prereqs if p not in completed]
                if missing:
                    verdict = (
                        f"No -- {student_id} is missing prerequisite(s) "
                        f"{', '.join(missing)} for {target_id} ({course['name']})."
                    )
                else:
                    verdict = f"Yes -- {student_id} has completed all prerequisites for {target_id}: {', '.join(prereqs)}."
            else:
                other_mentioned = [c for c in mentioned[1:] if c != target_id]
                if other_mentioned and set(other_mentioned) >= set(prereqs) and len(prereqs) <= len(other_mentioned):
                    # Question explicitly says those prereqs are NOT completed.
                    verdict = (
                        f"No -- {target_id} ({course['name']}) requires completing "
                        f"{', '.join(prereqs)} first."
                    )
                else:
                    verdict = (
                        f"{target_id} ({course['name']}) requires prerequisite(s): {', '.join(prereqs)}. "
                        f"Registration is only allowed once those are completed."
                    )

            policy_cite = next((p for p in cited_policies if p["id"] == "POL-4"), cited_policies[0])
            answer_text = f"{verdict}\n\nPer [{policy_cite['id']}]: \"{policy_cite['text']}\""
            return {
                "answer": answer_text,
                "course_id": target_id,
                "prerequisites": prereqs,
                "cited_policies": cited_policies,
            }

        # General policy question -- pure retrieval, no specific course mentioned.
        top = cited_policies[0]
        answer_text = f"Per [{top['id']}]: \"{top['text']}\""
        return {"answer": answer_text, "course_id": None, "cited_policies": cited_policies}


def main() -> None:
    bot = EligibilityBot()
    sample_questions = [
        ("Can I register for CS301 without having completed CS201?", None),
        ("Can I register for CS201 if I haven't completed CS101?", None),
        ("Can S0001 register for BIO201?", "S0001"),
        ("How many hours per week can a faculty member teach?", None),
        ("Do labs need to be in special rooms?", None),
    ]
    for q, sid in sample_questions:
        print(f"\nQ: {q}" + (f"  [student={sid}]" if sid else ""))
        result = bot.answer(q, student_id=sid)
        print(f"A: {result['answer']}")


if __name__ == "__main__":
    main()
