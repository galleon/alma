# Demo Script (5 minutes)

**Audience:** a prospective client evaluating whether to buy AI-assisted
academic scheduling automation.

**One sentence to open with:**

> "This is real constraint solving -- Google OR-Tools, the same solver used
> for airline crew scheduling and vehicle routing -- not an LLM guessing at a
> timetable. The AI layer only explains and queries what the solver decided."

Keep this framing in mind throughout: every time you touch the LLM, remind
them it is reading facts from a graph the solver populated, not inventing them.

---

## 1. The Generated Timetable tab (~90 seconds)

- Open the app, land on **Generated Timetable**.
- Say: *"This is one college's worth of synthetic data -- 20 courses, 10
  faculty, 5 rooms, 300 students, all fake, none of your real data. The solver
  just produced this from scratch."*
- Point at a couple of rows: a lab course sitting in a lab room, a large lecture
  course sitting in the biggest lecture hall, two different courses taught by
  the same faculty member at different times.
- Click **Regenerate (fresh solve, no overrides)**.
  - Watch the status line: *"solved in Xs"* -- call out that it's a couple of
    seconds, not minutes, and that it's independently verified afterward
    (*"no constraint violations"*) -- the solver isn't trusted blindly, its
    output is checked.
- Say: *"Every one of these placements respects hard rules: no faculty member
  double-booked, no room double-booked, labs only in lab rooms, rooms big
  enough for the class, and no two required courses in the same program
  clashing for a student."*

## 2. Toggle a constraint live (~60 seconds)

- Pick a faculty member from the dropdown (ideally one you noticed teaching
  2+ courses in step 1).
- Check a day off, e.g. **Tue**.
- Click **Apply & Re-solve**.
- Call out:
  - The solve time (still a couple of seconds).
  - The **"N section(s) differ from the original schedule"** count -- should
    be small (1-3 sections), and the `●` markers in the *Changed* column show
    exactly which rows moved.
- Say: *"Notice it didn't reshuffle the whole college. It found the minimal
  set of changes needed to respect the new constraint -- that's a direct
  consequence of using a real solver with an explicit objective, not a
  language model regenerating a schedule from scratch."*

## 3. Ask the Copilot tab (~90 seconds)

- Switch to **Ask the Copilot**.
- Ask: *"Why is CS301 scheduled when it is?"*
  - The answer will cite the actual faculty, room, and capacity, plus the
    specific policy (e.g. POL-5 on room capacity) -- expand the tool-call
    trace and show that the answer came from a live graph query, not the
    model's imagination.
- Ask the same what-if you just did manually in step 2, in natural language:
  *"What if \[faculty name\] is unavailable on Tuesdays?"*
  - The agent re-runs the solver itself and explains the diff in prose.
- Say: *"This is the same solver under the hood -- the chat interface is just
  a more natural way to pose the question and get the reasoning back in
  words, with the receipts (the tool calls) visible if you want to check them."*

## 4. Registration Eligibility Check tab (~60 seconds)

- Switch to **Registration Eligibility Check**.
- Ask the pre-filled question: *"Can I register for CS301 without having
  completed CS201?"*
- Click **Check eligibility**. Show the answer citing the exact policy text
  (POL-4), and expand *"Retrieved policy candidates"* to show the similarity
  scores -- this is retrieval-augmented generation grounded in the college's
  actual written policy, not a guess.
- Optionally pick a specific student ID from the dropdown and ask about one
  of their courses to show it cross-references real enrollment history too.

## 5. Close (~30 seconds)

> "Three things to take away: one, the schedule itself always comes from a
> real solver that can prove it satisfies every hard rule -- so trust in the
> output doesn't depend on trusting a language model's judgment. Two, when
> something changes, you get a minimal, explainable diff instead of a
> black-box reshuffle. Three, every 'why' question is answered by querying
> the same graph the solver populated, with citations back to your actual
> written policies -- not by an LLM improvising."

---

### If something goes wrong live

- **Ollama not responding / chat tab errors:** fall back to the Generated
  Timetable tab and the manual toggle-and-resolve flow -- that's the core
  pitch (constraint solving) and doesn't depend on the LLM at all.
- **Neo4j down:** the timetable tab and solver still work standalone; mention
  the graph layer conceptually and show `graph/queries.py` output from an
  earlier terminal session if needed.
- **Solve takes longer than expected:** still well under 5 seconds on this
  dataset size; if it's visibly slow, it's almost certainly a cold-start
  (first import of OR-Tools/Torch), not the solve itself -- the status line's
  `solved in Xs` reports the actual solver time.
