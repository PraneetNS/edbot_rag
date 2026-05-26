import re
from pathlib import Path

# Setup paths relative to the script
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
CLEANED_DIR = BASE_DIR / "cleaned_docs"

# Exact navigation, UI, and marketing noise to skip
BANNED_EXACT_STRINGS = {
    "elms",
    "courses",
    "about us",
    "contact us",
    "sign in",
    "register",
    "browse courses",
    "discover courses",
    "view all",
    "category",
    "search",
    "free",
    "0",
    "5",
    "platform",
    "legal",
    "privacy policy",
    "refund policy",
    "terms & conditions",
    "terms of service",
    "terms",
    "dashboard",
    "logout",
    "sign out",
    "get started",
    "enroll now",
    "buy course",
    "add to cart",
    "cart",
    "price",
    "rating",
    "ratings",
    "reviews",
    "testimonials",
    "social media",
    "facebook",
    "twitter",
    "linkedin",
    "instagram",
    "youtube",
    "vtu link",
    "verify",
    "popular courses",
    "start your journey",
    "expert instructors",
    "expert instructor",
    "our collaboration",
    "vtu collaboration",
    "hands-on learning",
    "personalized support",
    "flexible courses",
    "virtual internships",
    "our services",
    "our story",
    "our journey of innovation and impact",
    "strengthening industry-academic collaboration",
    "expanding opportunities through innovation",
    "our values",
    "innovation",
    "accessibility",
    "excellence",
    "collaboration",
    "for universities",
    "for students",
    "for career opportunities",
    "start your journey with edutainer",
    "not sure where to start?",
    "have questions or need guidance? we’re here to help. talk to our team to choose the right course, understand internships, or get personalized support for your learning journey.",
    "reshaping learning for the modern world, where education meets accessibility and inclusivity.",
    "learning for the",
    "modern world",
    "flexible learning",
    "industry driven internship",
    "cutting - edge courses",
    "cutting-edge courses",
    "go to programs",
    "24/7",
    "access",
    "learn at your own pace, anytime, anywhere.",
    "400+",
    "expert instructors",
    "guided by seasoned educators and industry leaders.",
    "100%",
    "placement support",
    "gain real-world experience with top companies.",
    "50+",
    "new courses",
    "stay ahead with the latest technology trends.",
    "edutainer",  # Repeated Catalog Prefix
}

def clean_line(line: str) -> str:
    return line.strip()

def should_skip(line: str, seen_lines: set) -> bool:
    line_lower = line.lower()
    
    # 1. Skip empty lines
    if not line:
        return True
        
    # 2. Skip if it is exactly a banned string (case-insensitive)
    if line_lower in BANNED_EXACT_STRINGS:
        return True
        
    # 3. Skip footer / copyright lines
    if "all rights reserved" in line_lower or "©" in line or "copyright" in line_lower:
        return True
        
    # 4. Skip pure numbers or ratings
    if re.match(r'^\d+$', line):
        return True
        
    # 5. Skip very short fragments under 5 characters (unless it's list items or FAQ labels like Q/A)
    if len(line) < 5:
        # Check if it starts with Q: or A: or is a number list like "1."
        if not (re.match(r'^(Q|A)\s*:', line, re.IGNORECASE) or re.match(r'^\d+\.?$', line)):
            return True
            
    # 6. Skip duplicate lines (case-insensitive) to prevent testimonial and catalog pollution
    if line_lower in seen_lines:
        return True
        
    return False

def clean_file(input_path: Path, output_path: Path):
    print(f"Cleaning: {input_path.name}")
    content = input_path.read_text(encoding="utf-8")
    lines = content.split("\n")
    
    cleaned_lines = []
    seen_lines = set()
    
    for l in lines:
        cleaned = clean_line(l)
        if not should_skip(cleaned, seen_lines):
            cleaned_lines.append(cleaned)
            # Add to deduplication set
            seen_lines.add(cleaned.lower())
            
    output_path.write_text("\n".join(cleaned_lines), encoding="utf-8")
    print(f"Saved: {output_path.name} ({len(cleaned_lines)} substantive lines)")

def clean_all_documents():
    CLEANED_DIR.mkdir(exist_ok=True, parents=True)
    
    for doc_file in DOCS_DIR.glob("*.txt"):
        out_file = CLEANED_DIR / doc_file.name
        clean_file(doc_file, out_file)
        
    print("\nAll documents cleaned successfully.")

if __name__ == "__main__":
    clean_all_documents()
