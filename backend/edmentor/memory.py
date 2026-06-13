import re
import logging
import time
import json
import os
from collections import deque
from threading import Timer
from typing import Dict, Any
from langchain_classic.memory import ConversationBufferWindowMemory

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 1800  # 30 minutes
session_last_active: dict[str, float] = {}

PROFILES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

def touch_session(session_id: str):
    if not session_id:
        session_id = "default"
    session_last_active[session_id] = time.time()

def get_last_responses(session_id: str) -> deque:
    if not session_id:
        session_id = "default"
    if session_id not in memory.last_responses:
        memory.last_responses[session_id] = deque(maxlen=3)
    return memory.last_responses[session_id]

def cleanup_expired_sessions():
    try:
        now = time.time()
        expired = [
            sid for sid, t in session_last_active.items()
            if now - t > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            memory.sessions.pop(sid, None)
            memory.profiles.pop(sid, None)
            memory.last_responses.pop(sid, None)
            session_last_active.pop(sid, None)
            
            # Delete profile from disk
            profile_path = os.path.join(PROFILES_DIR, f"{sid}.json")
            if os.path.exists(profile_path):
                try:
                    os.remove(profile_path)
                except Exception as e:
                    logger.error(f"Failed to delete profile file {profile_path}: {e}")
            
            print(f"[SESSION CLEANUP] Expired: {sid}")
    except Exception as e:
        logger.error(f"Error in cleanup_expired_sessions: {e}")
    # Schedule next cleanup in 10 minutes
    t = Timer(600, cleanup_expired_sessions)
    t.daemon = True
    t.start()

class EdmentorMemory:
    """
    LangChain ConversationBufferWindowMemory wrapper.
    Manages session-specific conversation memories and student profiles.
    """
    def __init__(self):
        self.sessions: Dict[str, ConversationBufferWindowMemory] = {}
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.last_responses: Dict[str, deque] = {}

    def get_or_create_session(self, session_id: str) -> ConversationBufferWindowMemory:
        """Get or create the ConversationBufferWindowMemory for a session."""
        if not session_id:
            session_id = "default"
        if session_id not in self.sessions:
            logger.info(f"Creating new LangChain memory session for ID: {session_id}")
            self.sessions[session_id] = ConversationBufferWindowMemory(
                k=4,
                return_messages=True,
                memory_key="chat_history",
                input_key="input",
                output_key="output"
            )
        return self.sessions[session_id]

    def get_profile(self, session_id: str) -> Dict[str, Any]:
        """Get the profile dictionary for a session (with defaults)."""
        if not session_id:
            session_id = "default"
        if session_id not in self.profiles:
            profile_path = os.path.join(PROFILES_DIR, f"{session_id}.json")
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, "r", encoding="utf-8") as f:
                        self.profiles[session_id] = json.load(f)
                    logger.info(f"Loaded profile from disk for session {session_id}")
                except Exception as e:
                    logger.error(f"Failed to load profile for {session_id}: {e}")
                    self._set_default_profile(session_id)
            else:
                self._set_default_profile(session_id)
        return self.profiles[session_id]

    def _set_default_profile(self, session_id: str) -> None:
        self.profiles[session_id] = {
            "year": "Not specified",
            "goal": "Engineering guidance",
            "weak_areas": "None specified"
        }
        self._save_profile_to_disk(session_id)

    def _save_profile_to_disk(self, session_id: str) -> None:
        if session_id in self.profiles:
            profile_path = os.path.join(PROFILES_DIR, f"{session_id}.json")
            try:
                with open(profile_path, "w", encoding="utf-8") as f:
                    json.dump(self.profiles[session_id], f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save profile for {session_id}: {e}")

    def update_profile(self, session_id: str, message: str) -> None:
        """
        Scan a user message to dynamically detect and update profile fields.
        - Year: sophomore/junior/senior/BTech/BE/2nd/3rd/4th year
        - Goal: placement/internship/GATE/MS/GRE/job
        - Weak areas: struggling/weak/stuck/confused with CS topics
        """
        profile = self.get_profile(session_id)
        msg_lower = message.lower()

        # 1. Year detection
        year_match = None
        if "2nd year" in msg_lower or "second year" in msg_lower or "sophomore" in msg_lower:
            year_match = "2nd Year BTech/BE"
        elif "3rd year" in msg_lower or "third year" in msg_lower or "junior" in msg_lower:
            year_match = "3rd Year BTech/BE"
        elif "4th year" in msg_lower or "fourth year" in msg_lower or "senior" in msg_lower or "final year" in msg_lower:
            year_match = "4th Year BTech/BE"
        elif "btech" in msg_lower or "be degree" in msg_lower:
            # General engineering
            if profile["year"] == "Not specified":
                year_match = "BTech/BE Student"
        
        if year_match:
            profile["year"] = year_match
            logger.info(f"Updated session {session_id} profile year to: {year_match}")

        # 2. Goal detection
        goal_match = None
        if "placement" in msg_lower or "campus recruit" in msg_lower:
            goal_match = "Placements"
        elif "internship" in msg_lower:
            goal_match = "Internships"
        elif "gate" in msg_lower:
            goal_match = "GATE Prep"
        elif "ms" in msg_lower or "masters" in msg_lower or "gre" in msg_lower:
            goal_match = "Higher Studies"
        elif "job" in msg_lower or "off-campus" in msg_lower or "career" in msg_lower:
            goal_match = "Software Engineering Career"

        if goal_match:
            profile["goal"] = goal_match
            logger.info(f"Updated session {session_id} profile goal to: {goal_match}")

        # 3. Weak areas detection
        # Look for phrases signifying struggle combined with CS terms
        struggle_phrases = ["struggling", "weak in", "bad at", "stuck on", "confused with", "trouble with", "don't understand", "hard time with"]
        if any(phrase in msg_lower for phrase in struggle_phrases):
            # Try to identify what CS/engineering topics they are talking about
            topics = []
            cs_keywords = [
                "dsa", "graph", "tree", "recursion", "dynamic programming", "dp", "linked list",
                "pointer", "sorting", "binary search", "array", "stack", "queue", "hashing",
                "sql", "database", "system design", "operating system", "networking"
            ]
            for kw in cs_keywords:
                if kw in msg_lower:
                    topics.append(kw.upper() if kw in ("dsa", "dp", "sql") else kw)
            if topics:
                weak_areas = ", ".join(topics)
                profile["weak_areas"] = weak_areas
                logger.info(f"Updated session {session_id} profile weak areas to: {weak_areas}")
                
        # Save to disk if updated
        if year_match or goal_match or (any(phrase in msg_lower for phrase in struggle_phrases) and topics):
            self._save_profile_to_disk(session_id)

    def save_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Save a completed conversation turn to LangChain memory."""
        # First detect any updates to the student profile from user message
        self.update_profile(session_id, user_msg)
        
        # Save to memory instance
        mem = self.get_or_create_session(session_id)
        # ConversationBufferWindowMemory accepts inputs as dictionaries
        mem.save_context({"input": user_msg}, {"output": assistant_msg})
        logger.info(f"Saved turn to memory for session {session_id}.")

    def clear_session(self, session_id: str) -> None:
        """Clear the memory and profile for a session."""
        if not session_id:
            session_id = "default"
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            del self.sessions[session_id]
        if session_id in self.profiles:
            del self.profiles[session_id]
        self.last_responses.pop(session_id, None)
        session_last_active.pop(session_id, None)
        
        # Delete from disk
        profile_path = os.path.join(PROFILES_DIR, f"{session_id}.json")
        if os.path.exists(profile_path):
            try:
                os.remove(profile_path)
            except Exception as e:
                logger.error(f"Failed to delete profile file {profile_path}: {e}")
                
        logger.info(f"Cleared memory and profile for session {session_id}.")

# Module-level singleton
memory = EdmentorMemory()

# Call cleanup_expired_sessions() once on module import to start the recurring loop
cleanup_expired_sessions()
