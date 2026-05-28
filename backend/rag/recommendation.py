# Edutainer asset catalog to score and recommend
ASSETS_CATALOG = [
    {
        "id": "ielts_course",
        "type": "course",
        "title": "English IELTS Readiness",
        "description": "Elevate your spoken and written English. Join online classes for effective IELTS prep.",
        "badge": "Language Exam",
        "action_text": "Learn About IELTS",
        "query_trigger": "What does the IELTS readiness course cover?",
        "tags": ["english", "ielts", "course"]
    },
    {
        "id": "vtu_cert",
        "type": "certification",
        "title": "VTU Virtual Internship",
        "description": "Earn 4-8 weeks of hands-on internship hours and an industry-aligned joint VTU certificate.",
        "badge": "LMS Certificate",
        "action_text": "Download Certificate Info",
        "query_trigger": "How do I download my VTU certificate?",
        "tags": ["vtu", "certificate", "credentials", "internship"]
    },
    {
        "id": "placement_roadmap",
        "type": "roadmap",
        "title": "Engineering Placement Roadmap",
        "description": "Step-by-step guideline on resume mentoring, interview questions, and placement roadmaps.",
        "badge": "Career Guide",
        "action_text": "View Placement Prep",
        "query_trigger": "Tell me about the placement preparation roadmap",
        "tags": ["placement", "job", "career", "interview", "resume"]
    },
    {
        "id": "support_portal",
        "type": "support",
        "title": "LMS Technical Support",
        "description": "Direct contact details, email links, and instructions to resolve course login issues.",
        "badge": "Support Helpline",
        "action_text": "Contact Helpdesk",
        "query_trigger": "How can I contact the academic support team?",
        "tags": ["support", "contact", "login", "password", "help"]
    }
]

def generate_recommendations(intent: str, active_topic: str, memory_profile: dict) -> list[dict]:
    """
    Priority-scores assets catalog based on intent, topic, and active memory profile.
    Hides cards below score 0.5. Returns top 2 relevant recommendation cards.
    """
    scored_assets = []
    
    # Extract keywords from memory visual profile for scoring
    target_domain = memory_profile.get("target_domain", {}).get("value", "") if memory_profile.get("target_domain") else ""
    weak_subject = memory_profile.get("weak_subject", {}).get("value", "") if memory_profile.get("weak_subject") else ""
    
    memory_keywords = []
    if target_domain:
        memory_keywords.extend(target_domain.lower().split("/"))
    if weak_subject:
        memory_keywords.append(weak_subject.lower())
        
    for asset in ASSETS_CATALOG:
        score = 0.0
        
        # 1. Intent Match (0.4 weight)
        intent_map = {
            "COURSE_QUERY": ["course"],
            "PLACEMENT_GUIDANCE": ["roadmap", "course"],
            "INTERNSHIP_GUIDANCE": ["certification", "roadmap"],
            "LMS_SUPPORT": ["support"],
            "CERTIFICATION_SUPPORT": ["certification", "support"],
            "EXAM_ASSISTANCE": ["course", "roadmap"]
        }
        
        allowed_types = intent_map.get(intent, [])
        if asset["type"] in allowed_types:
            score += 0.4
            
        # 2. Topic Match (0.4 weight)
        topic_lower = active_topic.lower()
        topic_words = topic_lower.split()
        
        topic_match_found = False
        for tag in asset["tags"]:
            # Direct match
            if tag in topic_lower:
                topic_match_found = True
                break
            # Word-level overlap
            for tw in topic_words:
                if len(tw) > 3 and tw in tag:
                    topic_match_found = True
                    break
                    
        if topic_match_found:
            score += 0.4
            
        # 3. Active Memory Match (0.2 weight)
        memory_match_found = False
        for tag in asset["tags"]:
            if any(mk in tag for mk in memory_keywords if len(mk) >= 2):
                memory_match_found = True
                break
        if memory_match_found:
            score += 0.2
            
        # Filter: Only keep relevant recommendations above the 0.5 confidence threshold
        if score >= 0.5:
            # Create a copy and insert score
            asset_copy = asset.copy()
            asset_copy["score"] = score
            scored_assets.append(asset_copy)
            
    # Sort by score in descending order
    scored_assets.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top 2 assets (clean of score metadata for clean payload API)
    final_cards = []
    for sa in scored_assets[:2]:
        final_cards.append({
            "title": sa["title"],
            "description": sa["description"],
            "badge": sa["badge"],
            "action_text": sa["action_text"],
            "query_trigger": sa["query_trigger"]
        })
        
    return final_cards
