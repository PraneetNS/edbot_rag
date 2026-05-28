import re

class ResponseCleaner:
    """
    Cleans up response text from LLM or offline fallback, ensuring high-quality formatting,
    proper casing, and removal of robotic RAG prefix leaks.
    """
    def __init__(self):
        pass

    def clean(self, text: str) -> str:
        """
        Polishes response text, removing internal duplication or robotic prefixes.
        """
        if not text:
            return ""

        cleaned = text.strip()

        # 1. Strip Q: / A: and Question: / Answer: remnants
        cleaned = re.sub(r'\b(Q|Question)\s*:\s*.*?\b(A|Answer)\s*:\s*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r'\b(Q|Question|A|Answer)\s*:\s*', '', cleaned, flags=re.IGNORECASE)

        # 2. Strip leading robotic system prefixes (such as "Based on the context...")
        robotic_prefixes = [
            r"^based\s+on\s+[^,.]*(,\s*|\.\s*|(?=\b[a-zA-Z]))",
            r"^according\s+to\s+[^,.]*(,\s*|\.\s*|(?=\b[a-zA-Z]))",
            r"^retrieved\s+information:?\s*",
            r"^q\s*:\s*",
            r"^a\s*:\s*",
            r"^question\s*:\s*",
            r"^answer\s*:\s*"
        ]

        for pattern in robotic_prefixes:
            match = re.match(pattern, cleaned, re.IGNORECASE)
            if match:
                cleaned = cleaned[match.end():]
                break

        cleaned = cleaned.strip()

        # 3. Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]

        # 4. Remove technical system terms leaks
        system_leaks = [
            r'\bRetrievalFallbackMode\b', r'\bchunk retrieval\b', 
            r'\bOllama\b', r'\bChromaDB\b', r'\bLlamaIndex\b',
            r'\bvector database\b', r'\bdatabase score\b', r'\btechnical error\b'
        ]
        for leak in system_leaks:
            cleaned = re.sub(leak, '', cleaned, flags=re.IGNORECASE)

        # 5. Trim duplicated periods or whitespace
        cleaned = re.sub(r'\.+', '.', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned
