import logging

logger = logging.getLogger(__name__)

# The strict RAG restricted prompt requested by the user
RAG_PROMPT = """
You are EduMentor.

You receive retrieved knowledge chunks.

Your job:
Use the context only if it directly answers the student's question.

Rules:

1. First understand the student's intent.

2. Check if the retrieved context actually matches the question.

3. If context is unrelated, ignore it completely.

4. Never force an answer from weak context.

5. Do not mention:
"retrieved context"
"knowledge base"
"source"

6. Answer naturally like a senior engineering mentor.

Student Question:
{question}


Knowledge:
{context}


Answer:
"""

# Default system prompt for direct LLM response when RAG is bypassed or rejected
DIRECT_MENTOR_SYSTEM_PROMPT = (
    "You are EduMentor, a dedicated and wise senior engineering academic mentor.\n"
    "Your tone is helpful, encouraging, and clear. Since there is no verified local context, "
    "explain concepts naturally like a senior engineering mentor using your own reasoning.\n"
    "Give clear explanations, practical steps, practical coding examples, learning paths, "
    "and mistakes to avoid when teaching engineering subjects."
)

class PromptBuilder:
    """
    Constructs strict, restricted prompts and manages system-level templates.
    """
    def __init__(self):
        pass

    def build_rag_prompt(self, question: str, context: str) -> str:
        return RAG_PROMPT.format(question=question, context=context)

    def build_direct_prompt(self, question: str, session_state: dict = None) -> str:
        topic = session_state.get("active_topic", "general") if session_state else "general"
        audience = "engineering student"
        
        # Adjust prompt structure dynamically by user metadata
        if session_state and "memory_profile" in session_state:
            td = session_state["memory_profile"].get("target_domain", {}).get("value", "engineering")
            ws = session_state["memory_profile"].get("weak_subject", {}).get("value", "None")
            audience = f"{session_state.get('mode', 'academic_mentor')} with interest in {td}"
            
        prompt = (
            f"You are speaking to a student mapped as a {audience}.\n"
            f"Please address the student question naturally as their engineering academic mentor.\n\n"
            f"Student Question:\n"
            f"{question}"
        )
        return prompt

    def get_direct_system_prompt(self) -> str:
        return DIRECT_MENTOR_SYSTEM_PROMPT
