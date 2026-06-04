import json
import re
import logging
from pathlib import Path
from llama_index.core import Document
from typing import List

logger = logging.getLogger(__name__)

# ── Topic inference keywords (used when 'topic' field is absent in the JSONL) ─
_TOPIC_KEYWORDS = {
    "dsa":          ["dsa", "algorithm", "data structure", "tree", "graph", "sort", "dynamic programming", "recursion"],
    "placement":    ["placement", "internship", "job", "interview", "resume", "recruit", "offer"],
    "cloud":        ["cloud", "aws", "gcp", "azure", "ec2", "s3", "deployment"],
    "frontend":     ["frontend", "react", "html", "css", "javascript", "web dev"],
    "backend":      ["backend", "api", "server", "database", "sql", "django", "flask", "node"],
    "ml":           ["machine learning", "ml", "deep learning", "nlp", "ai", "neural"],
    "programming":  ["python", "java", "c++", "golang", "rust", "code", "programming", "language"],
    "career":       ["career", "roadmap", "switch", "path", "skills", "goal", "mentor"],
    "productivity": ["procrastinat", "motivation", "focus", "habit", "routine", "balance", "study"],
    "open source":  ["open source", "github", "contribution", "pr", "pull request"],
}

def _infer_topic(text: str) -> str:
    """Return the most likely topic label based on keyword presence."""
    text_lower = text.lower()
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return topic.title()
    return "General"

def _strip_id_suffix(question: str) -> str:
    """
    Strip synthetic duplicate index tags like '[1446]' from the end of a question.
    Example: 'should i learn cloud during college [1446]' → 'should i learn cloud during college'
    """
    return re.sub(r"\s*\[\d+\]\s*$", "", question).strip()


def load_jsonl_conversations(file_path: str, deduplicate: bool = False) -> List[Document]:
    """
    Reads the JSONL dataset, validates conversations, filters out system messages,
    extracts student questions and mentor responses, and returns them as formatted
    educational documents.

    Args:
        file_path:    Absolute path to the .jsonl file.
        deduplicate:  If True, strips [NNNN] suffixes and skips duplicate canonical
                      questions. Recommended for edumentor_synthetic_5k.jsonl which
                      contains many synthetic variants of the same base question.
    """
    documents = []
    seen_canonical: set = set()   # tracks deduplicated question keys
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset JSONL not found at: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                messages = data.get("messages", [])

                # Gracefully handle missing topic/source (common in the 5k file)
                raw_topic  = data.get("topic", "").strip()
                source     = data.get("source", "Edumentor Dataset").strip()

                if not isinstance(messages, list):
                    continue

                user_questions    = []
                assistant_answers = []

                for msg in messages:
                    role    = msg.get("role")
                    content = msg.get("content", "").strip()
                    if not content:
                        continue
                    if role == "user":
                        user_questions.append(content)
                    elif role == "assistant":
                        assistant_answers.append(content)

                if not user_questions or not assistant_answers:
                    continue   # skip empty / malformed conversations

                student_question   = "\n".join(user_questions)
                mentor_explanation = "\n".join(assistant_answers)

                # ── Deduplication logic ───────────────────────────────────────
                if deduplicate:
                    canonical = _strip_id_suffix(student_question).lower()
                    if canonical in seen_canonical:
                        continue
                    seen_canonical.add(canonical)
                    # Use cleaned question for the stored document text
                    student_question = _strip_id_suffix(student_question)
                # ─────────────────────────────────────────────────────────────

                # Infer topic when field is absent
                topic = raw_topic if raw_topic else _infer_topic(
                    student_question + " " + mentor_explanation
                )

                # Format as conversational mentor chunk (spec: Student/Mentor voice format)
                # This ensures retrieved text already sounds like Edmentor — the LLM
                # distills guidance from it rather than translating documentation prose.
                formatted_text = (
                    f"Student: {student_question}\n"
                    f"Mentor: {mentor_explanation}"
                )

                doc = Document(
                    text=formatted_text,
                    metadata={
                        "topic":             topic,
                        "source":            source,
                        "original_question": student_question,
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.warning(f"Error parsing line {idx}: {e}")

    skipped = idx - len(documents) - len(seen_canonical) if deduplicate else 0
    logger.info(
        f"Loaded {len(documents)} clean mentoring documents "
        f"({'deduplication ON' if deduplicate else 'deduplication OFF'})."
    )
    return documents

