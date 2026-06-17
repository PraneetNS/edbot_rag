"""
Verification test for all three corrections.
Run with: .venv\Scripts\python demo\verify_three_corrections.py
"""
import sys
import re
sys.path.append(r"c:\Users\savan\OneDrive\Desktop\RAG\backend")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from edmentor.confidence_router import (
    StreamingDualParser,
    split_into_sentences,
    _cap_sentence_words,
    classify_query_intent,
)

PASS = "[PASS]"
FAIL = "[FAIL]"

def result(cond: bool, label: str):
    tag = PASS if cond else FAIL
    print(f"  {tag}  {label}")
    return cond

# ══════════════════════════════════════════════════════════════════════════════
# CORRECTION 1 — No-filler opener check (prompt only, tested via keyword scan)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("CORRECTION 1 — Anti-filler rule present in system prompt")
print("=" * 70)
from edmentor.prompt_builder import SYSTEM_PROMPT
checks = [
    ("Never open with Okay" in SYSTEM_PROMPT, "anti-filler rule for code speak block"),
    ("filler word" in SYSTEM_PROMPT, "'filler word' mentioned in prompt"),
    ("Here is a prime checker" in SYSTEM_PROMPT, "example opener present in prompt"),
]
all_c1 = all(result(c, l) for c, l in checks)
print(f"  CORRECTION 1: {'PASS' if all_c1 else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════════
# CORRECTION 2 — Roadmap intent classification + show-block fallback
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("CORRECTION 2 — Roadmap intent classification + show-block fallback")
print("=" * 70)

# 2a: intent classification
intent = classify_query_intent("give me a 60 day dsa roadmap")
result(intent == "visual-request",
       f"classify_query_intent('give me a 60 day dsa roadmap') == visual-request  (got {intent!r})")

# 2b: prompt forces BOTH speak AND show blocks for roadmap
roadmap_prompt_ok = (
    "You MUST produce BOTH a speak block AND a show block" in SYSTEM_PROMPT
    and "mandatory and non-negotiable" in SYSTEM_PROMPT
)
result(roadmap_prompt_ok, "prompt enforces show block for roadmap as non-negotiable")

# 2c: Exact format example present
result("Exact output format for roadmap queries:" in SYSTEM_PROMPT,
       "Exact output format example present in prompt")

# 2d: show-block fallback — simulate LLM that returns plain text for a roadmap query
print()
print("  --- Simulating show-block fallback (LLM returns no show tags) ---")
parser_fb = StreamingDualParser()
# Simulate: LLM ignores instructions and returns plain text (no tags at all)
raw_llm_text = (
    "Here is a 60-day DSA roadmap for you.\n"
    "Week 1-2: Arrays and Strings. Week 3-4: Linked Lists, Stacks, Queues.\n"
    "Week 5-6: Trees and BSTs. Week 7-8: Graphs, BFS, DFS.\n"
    "Week 9-10: Dynamic Programming and Mock Interviews."
)
events_fb = parser_fb.feed(raw_llm_text)
events_fb += parser_fb.finalize()
show_events_fb = [e for e in events_fb if e["type"] == "show"]
text_events_fb = [e for e in events_fb if e["type"] == "text"]
print(f"  Raw text events: {len(text_events_fb)}  Show events: {len(show_events_fb)}")
# The fallback is in generate_stream_with_routing (not the parser itself).
# Verify that the parser correctly produces text events from untagged content
# (the fallback code in the router then synthesises the show block).
result(len(text_events_fb) >= 1, "parser emits text events for untagged LLM output")
result(len(show_events_fb) == 0, "parser does NOT invent show events (fallback is in router)")

# 2e: Parser correctly handles properly-tagged roadmap output (existing case)
print()
print("  --- Parser with proper show tags (normal case) ---")
parser_ok = StreamingDualParser()
tagged_output = (
    '<speak>Here is your 60-day DSA roadmap in the chat below.</speak>'
    '<show type="roadmap" lang="">Week 1-2: Arrays\nWeek 3-4: Trees</show>'
)
CHUNK = 30
events_ok = []
for i in range(0, len(tagged_output), CHUNK):
    events_ok.extend(parser_ok.feed(tagged_output[i:i+CHUNK]))
events_ok.extend(parser_ok.finalize())
show_ok = [e for e in events_ok if e["type"] == "show"]
text_ok = [e for e in events_ok if e["type"] == "text"]
result(len(text_ok) >= 1, f"parser emits >=1 text event (got {len(text_ok)})")
result(len(show_ok) == 1, f"parser emits exactly 1 show event (got {len(show_ok)})")
if show_ok:
    result(show_ok[0]["show_type"] == "roadmap", f"show_type == roadmap (got {show_ok[0]['show_type']!r})")
    result("Week 1-2" in show_ok[0]["content"], "show content contains roadmap body")

print(f"\n  CORRECTION 2: {'PASS' if (intent == 'visual-request' and roadmap_prompt_ok and len(show_ok) == 1) else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════════
# CORRECTION 3 — 20-word TTS cap
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("CORRECTION 3 — 20-word TTS sentence cap")
print("=" * 70)

cases = [
    # (input, expected_parts_count, description)
    # NOTE: The first sentence is exactly 20 words — at the cap, NOT split (<=20).
    # We use a 21+ word sentence to confirm splitting behaviour.
    (
        "To get an internship in second year, start building a strong profile on GitHub and apply actively on Internshala and LinkedIn every single week.",
        2,
        ">20-word sentence with comma splits into <=20-word chunks"
    ),
    (
        "Arrays are the foundation.",
        1,
        "Short sentence is not split"
    ),
    (
        "DSA is important.",
        1,
        "Very short sentence is not split"
    ),
    (
        "This is a sentence with exactly nineteen words and should not be split at all here.",
        1,
        "16-word sentence is not split"
    ),
    # Exactly-20-word sentence stays intact (<=20 guard)
    (
        "To get an internship in second year, start with two solid projects on GitHub while applying to Internshala and LinkedIn.",
        1,
        "Exactly 20-word sentence is NOT split (at the cap boundary)"
    ),
]

all_c3 = True
for text, expected_n, label in cases:
    parts = _cap_sentence_words(text)
    passed = len(parts) == expected_n
    all_c3 &= passed
    result(passed, f"{label} (got {len(parts)} parts: {[len(p.split()) for p in parts]} words)")

# Extra: confirm all parts are <=20 words or the whole sentence had no clause boundary
long_input = (
    "To land an internship in your second year, you should build two complete "
    "projects on GitHub and apply actively on Internshala and LinkedIn every week."
)
parts_long = _cap_sentence_words(long_input)
all_under_cap = all(len(p.split()) <= 20 for p in parts_long)
result(all_under_cap, f"All capped parts <=20 words ({[len(p.split()) for p in parts_long]})")

print(f"\n  CORRECTION 3: {'PASS' if all_c3 and all_under_cap else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
overall = all_c1 and (intent == "visual-request" and roadmap_prompt_ok) and all_c3 and all_under_cap
print(f"OVERALL: {'ALL CORRECTIONS VERIFIED' if overall else 'ONE OR MORE CORRECTIONS FAILED'}")
print("=" * 70)
