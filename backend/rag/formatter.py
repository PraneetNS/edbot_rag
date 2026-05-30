import re

def clean_response(text: str) -> str:
    """
    Cleans response text from LLM or fallback by stripping unwanted prefixes,
    headings, duplicates, and database formatting artifacts.
    """
    if not text:
        return ""
        
    cleaned = text.strip()
    
    # Strip any inline Q: ... A: questions and their prefixes in the middle of text
    cleaned = re.sub(r'\b(Q|Question)\s*:\s*.*?\b(A|Answer)\s*:\s*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Strip any remaining standalone internal "Q:" or "A:" or "Question:" or "Answer:" prefixes
    cleaned = re.sub(r'\b(Q|Question|A|Answer)\s*:\s*', '', cleaned, flags=re.IGNORECASE)
    
    # 1. Clean common robotic RAG prefixes at the very start of the text
    # Matches "Based on..." or "According to..." up to the first comma/period, or transition word
    prefixes = [
        r"^based\s+on\s+[^,.]*(,\s*|\.\s*|(?=\b[a-zA-Z]))",
        r"^according\s+to\s+[^,.]*(,\s*|\.\s*|(?=\b[a-zA-Z]))",
        r"^retrieved\s+information:?\s*",
        r"^q\s*:\s*",
        r"^a\s*:\s*",
        r"^question\s*:\s*",
        r"^answer\s*:\s*"
    ]
    
    for p in prefixes:
        match = re.match(p, cleaned, re.IGNORECASE)
        if match:
            cleaned = cleaned[match.end():]
            break # Clean one leading prefix
            
    cleaned = cleaned.strip()
    
    # 2. Capitalize first letter if it was lowercased by prefix removal
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
        
    # 3. Trim internal double punctuation (like duplicated periods)
    cleaned = re.sub(r'\.+', '.', cleaned)
    
    return cleaned

def clean_text_chunks(hits: list, confidence_threshold: float = 0.25) -> tuple[list[str], list[float]]:
    """
    Cleans raw retrieved chunks, filters out navigation noise, duplicate lines,
    UI labels, short symbols, and returns a list of substantive text lines.
    """
    valid_hits = []
    scores = []
    
    if not hits:
        return [], []
        
    for h in hits:
        score = h.score if h.score is not None else 1.0
        # Filter by confidence threshold to avoid low-quality hallucinations
        if score >= confidence_threshold:
            valid_hits.append(h)
            scores.append(score)
            
    if not valid_hits:
        return [], []
        
    substantive_lines = []
    seen_lines = set()
    
    # Detailed case-insensitive navigation and UI label keywords
    navigation_keywords = {
        "home", "about", "about us", "courses", "contact", "contact us", 
        "sign in", "register", "login", "menu", "search", "navigation", 
        "copyright", "all rights reserved", "elms", "platform", "legal", 
        "privacy policy", "refund policy", "terms & conditions", "terms of service", 
        "terms", "category", "categories", "dashboard", "free", "logout", 
        "sign out", "view all", "discover courses", "learn more", "get started", 
        "enroll now", "buy course", "add to cart", "cart", "price", "rating", 
        "ratings", "reviews", "testimonials", "social media", "facebook", 
        "twitter", "linkedin", "instagram", "youtube", "vtu link", "verify",
        "popular courses", "start your journey", "expert instructors", "expert instructor",
        "our collaboration", "vtu collaboration", "hands-on learning"
    }
    
    # Pattern to match pure numeric lines, buttons, ratings, stats
    stat_pattern = re.compile(r'^(\d+|[\d\+\%\/\-kK]+|\d+K\+?|\d+\s*expert\s*instructors|\d+\s*hours?|\bfree\b|\bfree\s+trial\b)$', re.IGNORECASE)
    
    for h in valid_hits:
        lines = h.text.split("\n")
        for line in lines:
            cleaned_line = line.strip()
            if not cleaned_line:
                continue
                
            # Filter out lines that are too short (e.g. less than 3 chars or statistical symbols)
            if len(cleaned_line) < 3:
                continue
                
            # Check if line is purely UI statistical badge or ratings
            if stat_pattern.match(cleaned_line):
                continue
                
            # Clean internal excessive whitespace
            normalized_line = re.sub(r'\s+', ' ', cleaned_line)
            
            # Check if the line is exactly a navigation/UI keyword or contains repeated navigation labels
            line_lower = normalized_line.lower()
            
            # Simple keyword checks for navigation links (under 6 words)
            words = line_lower.split()
            if len(words) < 6:
                # If all words in the line are navigation/UI keywords, skip it
                if all(w in navigation_keywords or re.match(r'^[^\w\s]$', w) for w in words):
                    continue
                # If the line itself matches any navigation keyword, skip it
                if line_lower in navigation_keywords:
                    continue
                    
            # De-duplicate lines
            if line_lower not in seen_lines:
                seen_lines.add(line_lower)
                substantive_lines.append(normalized_line)
                
    return substantive_lines, scores

def extract_sentences(substantive_lines: list[str], question: str) -> list[str]:
    """
    Extracts high-quality sentences from substantive lines and prioritizes
    sentences that contain keywords from the user's question.
    """
    sentences = []
    seen_sentences = set()
    
    # Extract query keywords for relevance boosting (words over 3 characters)
    query_words = [w.lower() for w in re.findall(r'\b\w{4,}\b', question)]
    
    for line in substantive_lines:
        # Pre-process line to split Q: and A: on newlines to ensure sentence splitting handles them
        processed_line = re.sub(r'\b(Q|Question)\s*:', r'\n\1:', line, flags=re.IGNORECASE)
        processed_line = re.sub(r'\b(A|Answer)\s*:', r'\n\1:', processed_line, flags=re.IGNORECASE)
        
        lines_split = processed_line.split('\n')
        for l in lines_split:
            l_clean = l.strip()
            if not l_clean:
                continue
                
            # Split line into sentences if punctuation is present
            parts = re.split(r'(?<=[.!?])\s+', l_clean)
            for part in parts:
                part_clean = part.strip()
                # Filter out short fragments (under 15 characters)
                if len(part_clean) < 15:
                    continue
                    
                # Discard any sentence that is a question (starts with Q: or ends with ?)
                if re.match(r'^(Q|Question)\s*:\s*', part_clean, re.IGNORECASE) or part_clean.endswith('?'):
                    continue
                    
                # If the sentence starts with A: or Answer:, strip it
                if re.match(r'^(A|Answer)\s*:\s*', part_clean, re.IGNORECASE):
                    part_clean = re.sub(r'^(A|Answer)\s*:\s*', '', part_clean, flags=re.IGNORECASE).strip()
                    if not part_clean or len(part_clean) < 15:
                        continue
                
                # Capitalize first letter and ensure proper spacing
                part_clean = part_clean[0].upper() + part_clean[1:]
                
                # Remove any technical system terms that might have leaked
                technical_terms = [
                    r'\bRetrievalFallbackMode\b', r'\bchunk retrieval\b', 
                    r'\bOllama\b', r'\bChromaDB\b', r'\bLlamaIndex\b',
                    r'\bvector database\b', r'\bdatabase score\b', r'\btechnical error\b'
                ]
                contains_tech = False
                for term in technical_terms:
                    if re.search(term, part_clean, re.IGNORECASE):
                        contains_tech = True
                        break
                if contains_tech:
                    continue
                    
                sentence_lower = part_clean.lower()
                if sentence_lower not in seen_sentences:
                    seen_sentences.add(sentence_lower)
                    
                    # Calculate a relevance score based on keyword match
                    match_count = sum(1 for qw in query_words if qw in sentence_lower)
                    sentences.append((part_clean, match_count))
                
    # Sort sentences so that those matching the user's query keywords are first
    sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Return the text of the sentences
    return [s[0] for s in sentences]

def conversational_fallback(question: str, hits: list) -> str:
    """
    Generates a clean, concise, conversational, and student-friendly response
    using retrieved chunks when the LLM is offline. Supports soft thresholds
    and dynamic semantic fallbacks.
    """
    # 0. Fast rule-based conversational/greeting handlers for offline mode
    q_clean = question.strip().lower().rstrip("?!.")
    greetings = ["hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening"]
    if any(g == q_clean or q_clean.startswith(g + " ") for g in greetings):
        return "Hello! I am EduBot, your Edutainer AI Engineering Academic Mentor. How can I help you with your engineering courses, placements, internships, or certifications today?"
        
    identity_triggers = ["who are you", "what is your name", "what's your name", "tell me about yourself", "your identity"]
    if any(trigger in q_clean for trigger in identity_triggers):
        return "I am EduBot, your dedicated Edutainer AI Engineering Academic Mentor. I am here to guide 2nd, 3rd, and 4th-year engineering students on academics, technical skills (DSA, OOP, Web Dev, AI/ML), placements, and VTU certifications!"
        
    status_triggers = ["how are you", "how's it going", "how are you doing", "how do you do"]
    if any(trigger in q_clean for trigger in status_triggers):
        return "I am doing great, thank you! Ready to support your engineering learning journey today. What year of engineering are you in, or what concepts can I help you explore?"

    # 0.5. Soft Confidence Threshold & Dynamic Fallbacks
    highest_score = max(h.score if h.score is not None else 0.0 for h in hits) if hits else 0.0
    
    is_educational = any(kw in q_clean for kw in [
        "course", "placement", "internship", "certif", "lms", "support", "help",
        "contact", "learn", "study", "exam", "admission", "fee", "price", "duration",
        "react", "python", "java", "coding", "partner", "vtu", "job", "career",
        "syllabus", "curriculum", "schedule", "class", "project", "assignment", "bot", 
        "edutainer", "edubot", "programming", "code", "developer", "development", 
        "software", "database", "sql", "computer", "science", "engineering", 
        "math", "algorithm", "data", "structure", "second", "2nd", "third", "3rd", 
        "fourth", "4th", "final", "sophomore", "junior", "senior", "capstone", 
        "dsa", "oop", "system design", "resume", "interview", "mentor", "roadmaps"
    ])
    if not is_educational:
        return (
            "I am EduBot, your dedicated AI Engineering Academic Mentor. I can only assist you with questions "
            "related to engineering courses, core CS concepts (DSA, OOP, SQL), placements, internships, VTU certifications, "
            "and LMS support.\n\n"
            "Please let me know how I can help you with your academic or career development in engineering!"
        )

    if is_educational and (not hits or highest_score < 0.25):
        if any(k in q_clean for k in ["second year", "2nd year", "sophomore"]):
            return (
                "For 2nd-year engineering students, my primary goal is to help you build ironclad CS foundations, "
                "including Data Structures & Algorithms (DSA), Object-Oriented Programming (OOP using Java/C++), "
                "Database Management Systems (SQL), and Operating Systems. Maintaining a strong CGPA and working "
                "on foundational mini-projects during this stage is highly recommended. \n\n"
                "What foundational concept or language can I help you explore today?"
            )
        elif any(k in q_clean for k in ["third year", "3rd year", "junior"]):
            return (
                "For 3rd-year engineering students, our focus is professional upskilling and career readiness, "
                "specifically targeting Full-Stack Web Development (React JS, Node.js), introductory System Design, "
                "Software Engineering, and securing summer internships. Preparing your resume and completing VTU "
                "virtual internships/certifications will significantly boost your profile. \n\n"
                "Would you like guidance on building your resume, or information about VTU virtual internship requirements?"
            )
        elif any(k in q_clean for k in ["fourth year", "4th year", "final year", "senior", "capstone"]):
            return (
                "For 4th-year/final-year engineering students, our focus shifts to advanced specializations like "
                "Artificial Intelligence, Machine Learning, Cloud Systems, preparing for Capstone projects, and rigorous "
                "placement preparation including mock technical interviews, resume workshops, and campus drives. \n\n"
                "Would you like me to guide you to our placement preparation roadmap, or discuss capstone project execution?"
            )
        elif any(k in q_clean for k in ["placement", "job", "career", "interview", "resume", "recruit", "internship"]):
            return (
                "While our direct placement preparation records are not fully detailed here, Edutainer provides "
                "industry-aligned career guidance, resume mentoring workshops, mock technical interviews, and engineering "
                "internships to prepare technology students for their dream roles. \n\n"
                "Would you like to explore our engineering placement roadmap, or learn about our internship requirements?"
            )
        elif any(k in q_clean for k in ["certif", "vtu", "stamp", "badge", "credentials", "diploma"]):
            return (
                "While specific certification codes are being updated, Edutainer offers verified upskilling "
                "certifications—including joint VTU virtual internship credentials—to validate your hands-on coding and "
                "technology skills. \n\n"
                "I can explain the passing requirements or show you how to download your verified VTU certificate. "
                "What would you like to do next?"
            )
        elif any(k in q_clean for k in ["support", "contact", "help", "issue", "portal", "login", "mail", "phone"]):
            return (
                "If you are having trouble accessing the Learning Management System (LMS) or logging in, you can reach out "
                "directly to our dedicated portal support team at support@edutainer.in, and we will get your account issues "
                "resolved right away. \n\n"
                "Would you like the contact phone number and support hours, or should I guide you on how to submit a ticket "
                "for login issues?"
            )
        elif any(k in q_clean for k in ["course", "syllabus", "curriculum", "program", "class", "react", "python", "java", "coding", "learn", "study", "engineering", "academic"]):
            return (
                "Edutainer currently appears to offer several industry-focused technical programs including "
                "Artificial Intelligence, React JS, Python programming, and other professional upskilling courses designed "
                "to advance your engineering and technology skills. \n\n"
                "Would you like me to guide you to the course registration page on your dashboard, or would you like "
                "more details about a specific syllabus?"
            )
        else:
            return "I currently do not have enough verified course information available regarding that topic. Please contact our support team at support@edutainer.in for further assistance!"

    if not hits:
        return "I currently do not have enough verified course information available regarding that topic. Please contact our support team at support@edutainer.in for further assistance!"

    # 1. Clean and filter chunks
    substantive_lines, scores = clean_text_chunks(hits, confidence_threshold=0.15) # Relaxed to 0.15 to allow semantically close matches
    
    # 2. Extract and prioritize sentences
    sentences = extract_sentences(substantive_lines, question)
    
    # 3. Handle low-confidence / insufficient information
    if not sentences:
        return "I currently do not have enough verified course information available regarding that topic. Please contact our support team at support@edutainer.in for further assistance!"
        
    # 4. Select top 2-3 sentences to make a concise paragraph
    selected_sentences = sentences[:3]
    
    # To keep document logic, sort selected sentences by their original order in substantive_lines
    def get_original_index(s):
        for idx, line in enumerate(substantive_lines):
            if s.lower() in line.lower():
                return idx
        return 999
        
    selected_sentences.sort(key=get_original_index)
    summary_paragraph = " ".join(selected_sentences)
    
    # 5. Determine active topic to format custom student-friendly conversational endings
    q_lower = question.lower()
    topic = "general"
    if any(k in q_lower for k in ["second year", "2nd year", "sophomore"]):
        topic = "2nd_year"
    elif any(k in q_lower for k in ["third year", "3rd year", "junior"]):
        topic = "3rd_year"
    elif any(k in q_lower for k in ["fourth year", "4th year", "final year", "senior", "capstone"]):
        topic = "4th_year"
    elif any(k in q_lower for k in ["placement", "job", "career", "interview", "resume", "recruit"]):
        topic = "placements"
    elif any(k in q_lower for k in ["react", "python", "java", "course", "syllabus", "curriculum", "learn", "teach", "class"]):
        topic = "courses"
    elif any(k in q_lower for k in ["certif", "vtu", "stamp", "badge", "diploma"]):
        topic = "certifications"
    elif any(k in q_lower for k in ["support", "contact", "help", "issue", "portal", "login", "mail", "phone"]):
        topic = "support"
        
    # 6. Begin direct and naturally (no robotic prefixes)
    full_response = clean_response(summary_paragraph)
    
    # 7. Add a student-friendly educational guide closing and exactly one follow-up question
    if topic == "2nd_year":
        full_response += (
            "\n\nMaintaining a solid CGPA and starting foundational mini-projects during your sophomore year is highly advised. "
            "What computer science topic or language are you looking to master next?"
        )
    elif topic == "3rd_year":
        full_response += (
            "\n\nSecuring summer internships and obtaining verified industry certificates will significantly improve your profile. "
            "Would you like guidance on how to structure your resume or prep for mock technical interviews next?"
        )
    elif topic == "4th_year":
        full_response += (
            "\n\nPreparing for rigorous placement drives and executing a strong Capstone project are critical final year milestones. "
            "What specific role or technology specialization are you targeting for your placements?"
        )
    elif topic == "courses":
        full_response += (
            "\n\nWould you like me to guide you to the course registration page on your dashboard, "
            "or would you like more details about a specific syllabus?"
        )
    elif topic == "placements":
        full_response += (
            "\n\nWould you like to explore our placement preparation roadmap, or learn about "
            "upcoming engineering internship opportunities?"
        )
    elif topic == "certifications":
        full_response += (
            "\n\nI can also explain the passing requirements or show you how to download your "
            "verified VTU certificate. What would you like to do next?"
        )
    elif topic == "support":
        full_response += (
            "\n\nWould you like the contact phone number and support hours, or should I guide you "
            "on how to submit a ticket for login issues?"
        )
    else:
        full_response += (
            "\n\nI hope this helps support your engineering learning journey! What other questions or topic "
            "would you like to explore next?"
        )
        
    return full_response
