"""
verify_all.py — EduMentor 6-test verification suite.
Run with: python verify_all.py
Requires the FastAPI server to be running at http://localhost:8000
"""
import sys
import time
import json

# Fix Windows cp1252 console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import sqlite3
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx not available, using urllib")
    httpx = None

BASE = "http://localhost:8000"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []

def log(label, ok, detail=""):
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}")
    if detail:
        print(f"         {detail}")
    results.append((label, ok))

def post(url, body):
    import urllib.request, urllib.error
    data = json.dumps(body).encode()
    req  = urllib.request.Request(BASE + url, data=data,
                                   headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

def get(url, token=None):
    import urllib.request, urllib.error
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

print("\n" + "="*60)
print(" EduMentor Verification Suite")
print("="*60 + "\n")

# ── TEST 1 — Register + login ────────────────────────────────────────────────
print("TEST 1 — Register and login")
ts  = int(time.time())
uname = f"testuser_{ts}"

status, body = post("/auth/register", {"username": uname, "password": "test123", "year": "3rd year"})
ok1a = status == 201 and "student_id" in body
log("Register returns student_id", ok1a, f"status={status} body={body}")

student_id = body.get("student_id", "")

status, body = post("/auth/login", {"username": uname, "password": "test123"})
ok1b = status == 200 and "access_token" in body
log("Login returns access_token", ok1b, f"status={status}")

token = body.get("access_token", "")
print()

# ── TEST 6 (early) — No token → 401 ─────────────────────────────────────────
print("TEST 6 — Unauthenticated access blocked")
if student_id:
    status, body = get(f"/dashboard/{student_id}/stats")  # no token
    ok6 = status == 401
    log("GET /dashboard/stats with no token → 401", ok6, f"status={status}")
else:
    log("GET /dashboard/stats with no token → 401", False, "No student_id (register failed)")
print()

# ── TEST 2 — Authenticated query ─────────────────────────────────────────────
print("TEST 2 — Authenticated query → turn saved to DB")
if token and student_id:
    import urllib.request
    data = json.dumps({"question": "how do i prepare for amazon", "session_id": f"sess_{ts}"}).encode()
    req  = urllib.request.Request(
        BASE + "/edmentor/query", data=data,
        headers={"Content-Type":"application/json", "Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            q_status = r.status
            q_body   = json.loads(r.read())
    except Exception as e:
        q_status, q_body = 0, {"error": str(e)}

    ok2a = q_status == 200 and "response" in q_body
    log("Query returns response", ok2a, f"status={q_status} words={len(q_body.get('response','').split())}")

    # Give background tracker time to write
    time.sleep(1.5)

    import urllib.request as ur2
    req2 = ur2.Request(
        BASE + f"/dashboard/{student_id}/timeline?limit=5",
        headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with ur2.urlopen(req2, timeout=10) as r2:
            tl = json.loads(r2.read())
    except Exception as e:
        tl = []

    ok2b = isinstance(tl, list) and len(tl) > 0
    log("Turn appears in /timeline", ok2b, f"turns_found={len(tl)}")
else:
    log("Query returns response",     False, "Token missing (login failed)")
    log("Turn appears in /timeline",  False, "Token missing")
print()

# ── TEST 3 — 5 queries across 2 topics ───────────────────────────────────────
print("TEST 3 — 5 queries → correct stats breakdown")
if token and student_id:
    sess3 = f"sess3_{ts}"
    queries = [
        "explain binary search tree for interview",
        "how to solve dp problems on leetcode",
        "which companies should i target for placement",
        "how to crack amazon oa round",
        "what is time complexity of quicksort",
    ]
    for q in queries:
        import urllib.request as ur3
        data3 = json.dumps({"question": q, "session_id": sess3}).encode()
        req3  = ur3.Request(BASE + "/edmentor/query", data=data3,
                            headers={"Content-Type":"application/json",
                                     "Authorization": f"Bearer {token}"})
        try:
            with ur3.urlopen(req3, timeout=30) as r3:
                pass
        except Exception:
            pass
        time.sleep(0.3)

    time.sleep(2.0)  # wait for background tracker

    import urllib.request as ur4
    req4 = ur4.Request(BASE + f"/dashboard/{student_id}/stats",
                       headers={"Authorization": f"Bearer {token}"})
    try:
        with ur4.urlopen(req4, timeout=10) as r4:
            stats = json.loads(r4.read())
    except Exception as e:
        stats = {}

    total = stats.get("total_turns", 0)
    breakdown = stats.get("topic_breakdown", {})
    ok3a = total >= 5
    log(f"total_turns >= 5 (got {total})", ok3a, f"breakdown={breakdown}")

    # Check dsa and placement are both represented
    has_dsa       = breakdown.get("dsa", 0) > 0
    has_placement = breakdown.get("placement", 0) > 0
    ok3b = has_dsa and has_placement
    log(f"dsa + placement both in breakdown", ok3b,
        f"dsa={breakdown.get('dsa',0)} placement={breakdown.get('placement',0)}")
else:
    log("total_turns >= 5",               False, "Token missing")
    log("dsa + placement in breakdown",    False, "Token missing")
print()

# ── TEST 4 — Follow-up on 3rd turn ───────────────────────────────────────────
print("TEST 4 — 3rd turn triggers follow-up question")
if token and student_id:
    sess4 = f"sess4_{ts}"
    followup_detected = False
    for i, q in enumerate([
        "i am struggling with graph problems",
        "can you explain bfs and dfs",
        "which problems should i solve first on leetcode",
    ]):
        import urllib.request as ur5
        data5 = json.dumps({"question": q, "session_id": sess4}).encode()
        req5  = ur5.Request(BASE + "/edmentor/query", data=data5,
                            headers={"Content-Type":"application/json",
                                     "Authorization": f"Bearer {token}"})
        try:
            with ur5.urlopen(req5, timeout=30) as r5:
                resp_body = json.loads(r5.read())
                resp_text = resp_body.get("response","")
                if i == 2:  # 3rd query (0-indexed)
                    followup_detected = resp_text.strip().endswith("?")
        except Exception as e:
            if i == 2:
                resp_text = f"Error: {e}"
        time.sleep(0.4)

    log("3rd response ends with '?'", followup_detected,
        f"3rd response: {resp_text[:120] if 'resp_text' in dir() else 'N/A'}")
else:
    log("3rd response ends with '?'", False, "Token missing")
print()

# ── TEST 5 — Dashboard HTML loads ────────────────────────────────────────────
print("TEST 5 — Dashboard HTML accessible")
import urllib.request as ur6
try:
    with ur6.urlopen(BASE + "/dashboard.html", timeout=5) as r6:
        html_ok = r6.status == 200 and b"EduMentor" in r6.read()
    log("GET /dashboard.html returns 200 + HTML", html_ok)
except Exception as e:
    log("GET /dashboard.html returns 200 + HTML", False, str(e))

try:
    with ur6.urlopen(BASE + "/login.html", timeout=5) as rl:
        login_ok = rl.status == 200 and b"EduMentor" in rl.read()
    log("GET /login.html returns 200 + HTML", login_ok)
except Exception as e:
    log("GET /login.html returns 200 + HTML", False, str(e))
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("="*60)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f" Results: {passed}/{total} checks passed")
print("="*60)
for label, ok in results:
    print(f"  {'✓' if ok else '✗'} {label}")
print()
sys.exit(0 if passed == total else 1)
