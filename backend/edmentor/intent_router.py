import sys
from pathlib import Path

# Ensure edmentor directory is in sys.path
EDMENTOR_DIR = Path(__file__).resolve().parent
if str(EDMENTOR_DIR) not in sys.path:
    sys.path.append(str(EDMENTOR_DIR))

try:
    from guard import guard as edmentor_guard
except ImportError:
    # Fallback to local import if needed
    import guard
    edmentor_guard = guard.guard

import re

def is_off_domain(query: str) -> bool:
    """
    Evaluates if the query is outside Edmentor's mentoring domain.
    Returns True if off-domain (blocked), False if on-domain.
    """
    query_lower = query.lower().strip()
    
    # Check explicit blocklist patterns (which might otherwise bypass DomainGuard because of helper keywords like 'how')
    safety_blocklist = [
        r"\bhack(ing|er|s)?\b", r"\bexploits?\b", r"\bbypass\s+security\b", r"\bpolitics?\b", 
        r"\belections?\b", r"\bspam\b", r"\brecipes?\b", r"\bmovies?\b", r"\bgames?\b", 
        r"\bbake\s+a\s+cake\b", r"\bcake\b", r"\bbake\b", r"\btell\s+a\s+joke\b", r"\bweather\b"
    ]
    for pattern in safety_blocklist:
        if re.search(pattern, query_lower):
            return True
            
    is_blocked, _, _ = edmentor_guard.check(query)
    return is_blocked

