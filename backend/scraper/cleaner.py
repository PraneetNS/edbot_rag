import re
from bs4 import BeautifulSoup

def clean_html(html_content: str, remove_nav_footer: bool = False) -> str:
    """
    Cleans raw HTML content by removing scripts, styles, and optionally
    navigation and footer elements, then extracts and normalizes the text.
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content tags
    tags_to_remove = ["script", "style", "noscript", "svg", "iframe", "code"]
    if remove_nav_footer:
        tags_to_remove.extend(["nav", "footer", "header"])

    for tag in soup(tags_to_remove):
        tag.extract()

    # Get raw text with newline separator
    text = soup.get_text(separator="\n")

    # Clean and normalize lines
    cleaned_lines = []
    for line in text.splitlines():
        trimmed = line.strip()
        # Filter out extremely short lines, social media icons, or generic noise
        if not trimmed:
            continue
        # Remove multiple spaces inside lines
        normalized = re.sub(r'\s+', ' ', trimmed)
        cleaned_lines.append(normalized)

    # Join with newlines
    clean_text = "\n".join(cleaned_lines)
    
    # Remove excessive blank lines
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    
    return clean_text
