"""
edmentor/topic_classifier.py
─────────────────────────────
Centroid-cosine topic classifier for ChromaDB metadata filtering.
Uses the same pattern as ai_core/intent/classifier.py but with
Edmentor-specific topic centroids.

Architecture:
    - At init: embed 6-8 seed phrases per topic, average into a centroid vector
    - At classify(): embed query once, compute cosine vs all centroids → argmax
    - Total latency: ~10ms (MiniLM already loaded)

Topics map 1-to-1 to ChromaDB 'topic' metadata field for where-filter retrieval.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# ── Topic seed phrases ──────────────────────────────────────────────────────
# Each list = 6-8 representative queries for that topic domain.
# The average embedding of these becomes the centroid for cosine comparison.

TOPIC_SEEDS = {
    "Dsa": [
        "how do I learn data structures and algorithms",
        "explain binary search tree time complexity",
        "dynamic programming approach for subproblems",
        "graph traversal using BFS and DFS",
        "sorting algorithm comparison quicksort mergesort",
        "recursion base case and memoization",
        "linked list reversal in place",
        "how to crack DSA interviews at product companies",
    ],
    "Placement": [
        "how to prepare for campus placements",
        "what is the placement preparation roadmap",
        "crack product based company interview",
        "off campus placement strategy as final year student",
        "mock interview practice for software jobs",
        "placement drive preparation tips",
        "how to get shortlisted in placement",
        "technical interview rounds preparation",
    ],
    "Resume": [
        "how to write a good engineering resume",
        "resume tips for fresher software engineer",
        "ATS friendly resume format for placements",
        "what to put on resume for internship applications",
        "resume review for product company applications",
        "how to list projects on a resume",
        "resume mistakes to avoid as a student",
        "make my resume stand out for Google Amazon",
    ],
    "Career": [
        "which domain should I choose for my career",
        "career roadmap for software engineer",
        "how to switch from ECE to software development",
        "which skills should I learn for future job market",
        "career advice for engineering students",
        "how to plan my engineering career",
        "should I go for GATE or placements",
        "career path comparison backend vs frontend vs ML",
    ],
    "Programming": [
        "learn Python programming from scratch",
        "object oriented programming concepts explained",
        "how does memory management work in C++",
        "explain JavaScript event loop and closures",
        "best way to learn a new programming language",
        "functional programming vs object oriented programming",
        "how to write clean maintainable code",
        "debugging techniques for beginners",
    ],
    "Projects": [
        "what projects should I build as a student",
        "final year project ideas for computer science",
        "how to build a full stack web application",
        "beginner project ideas to learn machine learning",
        "how to deploy a project on GitHub",
        "project ideas to improve my resume",
        "how to contribute to open source projects",
        "capstone project planning for engineering students",
    ],
    "Ml": [
        "how to get started with machine learning",
        "explain neural networks for beginners",
        "difference between supervised and unsupervised learning",
        "NLP project ideas for students",
        "deep learning vs traditional ML when to use which",
        "how to build a recommendation system",
        "best ML libraries Python scikit learn tensorflow",
        "machine learning engineer career path",
    ],
    "Higher Studies": [
        "should I do masters after engineering",
        "how to prepare for GRE GMAT",
        "GATE preparation strategy for CSE",
        "MS in US vs job after BTech which is better",
        "how to write statement of purpose for MS",
        "research internship application tips",
        "PhD after engineering pros and cons",
        "how to get research publications as undergrad",
    ],
    "Open Source": [
        "how to start contributing to open source",
        "find good first issues on GitHub",
        "how to submit a pull request",
        "GSoC Google Summer of Code application tips",
        "open source contribution for resume",
        "fork and clone a repository workflow",
        "how to write a good commit message",
        "open source projects for beginners to contribute",
    ],
}

# Fallback topic when no centroid is close enough
DEFAULT_TOPIC = "General"
SIMILARITY_THRESHOLD = 0.25  # below this → General (no topic filter applied)


class EdmentorTopicClassifier:
    """
    Centroid-cosine topic classifier.
    Shares the embed_model instance to avoid double-loading MiniLM.
    """

    def __init__(self, embed_model=None):
        if embed_model is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            logger.info("TopicClassifier: loading MiniLM embedding model...")
            self.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            self.embed_model = embed_model

        self.centroids: dict[str, np.ndarray] = {}
        self._build_centroids()

    def _build_centroids(self) -> None:
        """Embed all seed phrases and average into per-topic centroid vectors."""
        logger.info("TopicClassifier: computing topic centroids ...")
        for topic, seeds in TOPIC_SEEDS.items():
            embeddings = []
            for seed in seeds:
                try:
                    emb = np.array(
                        self.embed_model.get_text_embedding(seed), dtype=np.float32
                    )
                    embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Failed to embed seed '{seed}': {e}")
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                # Normalise centroid for fast cosine via dot product
                norm = np.linalg.norm(centroid)
                self.centroids[topic] = centroid / norm if norm > 0 else centroid
        logger.info(
            f"TopicClassifier: centroids ready for {list(self.centroids.keys())}"
        )

    def classify(self, query: str) -> str:
        """
        Classify query into a topic label.

        Returns:
            Topic string matching ChromaDB 'topic' metadata field,
            or 'General' if below similarity threshold (no filter applied).
        """
        try:
            q_emb = np.array(
                self.embed_model.get_query_embedding(query), dtype=np.float32
            )
            norm = np.linalg.norm(q_emb)
            if norm == 0:
                return DEFAULT_TOPIC
            q_emb = q_emb / norm  # normalise for dot-product cosine

            best_topic = DEFAULT_TOPIC
            best_score = SIMILARITY_THRESHOLD  # minimum bar to pass

            for topic, centroid in self.centroids.items():
                score = float(np.dot(q_emb, centroid))
                if score > best_score:
                    best_score = score
                    best_topic = topic

            logger.debug(
                f"TopicClassifier: '{query[:60]}' → '{best_topic}' (score={best_score:.4f})"
            )
            return best_topic

        except Exception as e:
            logger.error(f"TopicClassifier.classify error: {e}")
            return DEFAULT_TOPIC
