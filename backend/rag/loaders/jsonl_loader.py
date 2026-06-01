import json
import logging
from pathlib import Path
from llama_index.core import Document
from typing import List

logger = logging.getLogger(__name__)

def load_jsonl_conversations(file_path: str) -> List[Document]:
    """
    Reads the JSONL dataset, validates conversations, filters out system messages,
    extracts student questions and mentor responses, and returns them as formatted
    educational documents.
    """
    documents = []
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
                topic = data.get("topic", "General").strip()
                source = data.get("source", "Unknown").strip()

                if not isinstance(messages, list):
                    continue

                user_questions = []
                assistant_answers = []

                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content", "").strip()
                    if not content:
                        continue
                    if role == "user":
                        user_questions.append(content)
                    elif role == "assistant":
                        assistant_answers.append(content)

                if not user_questions or not assistant_answers:
                    # Skip invalid or empty conversations
                    continue

                student_question = "\n".join(user_questions)
                mentor_explanation = "\n".join(assistant_answers)

                # Format as pure educational mentor knowledge document
                formatted_text = (
                    "Title:\n"
                    "Engineering Mentor Knowledge\n\n"
                    "Student Question:\n"
                    f"{student_question}\n\n"
                    "Mentor Explanation:\n"
                    f"{mentor_explanation}\n\n"
                    f"Topic:\n{topic}\n\n"
                    f"Source:\n{source}"
                )

                doc = Document(
                    text=formatted_text,
                    metadata={
                        "topic": topic,
                        "source": source,
                        "original_question": student_question
                    }
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Error parsing line {idx}: {e}")

    logger.info(f"Loaded {len(documents)} clean mentoring documents.")
    return documents
