import sys
import re
import requests
from bs4 import BeautifulSoup
import trafilatura
from pathlib import Path

# Add parent path to import config
sys.path.append(str(Path(__file__).resolve().parent))
import config

class EducationalScraper:
    """
    Highly targeted educational content scraper that extracts semantic documents,
    removing ads, headers, footers, and scripts using trafilatura and BeautifulSoup.
    """
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def scrape_url(self, url: str) -> tuple[str, str]:
        """
        Scrapes a URL. Uses trafilatura as primary, falling back to BeautifulSoup.
        Returns: (extracted_text, title)
        """
        print(f"Scraping targeted URL: {url}...")
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                print(f"Error: Non-200 response ({response.status_code}) for {url}")
                return "", ""
                
            html_content = response.text
            
            # Extract title using BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            title = soup.title.string.strip() if soup.title else "Untitled Educational Document"
            
            # 1. Primary Extraction: Trafilatura
            extracted = trafilatura.extract(
                html_content, 
                include_comments=False, 
                include_tables=True,
                no_fallback=False
            )
            
            if extracted:
                print(f"Successful extraction via Trafilatura from {url}")
                return extracted, title
                
            # 2. Secondary Extraction: BeautifulSoup Fallback (semantic content only)
            print(f"Trafilatura returned empty. Falling back to BeautifulSoup for {url}...")
            
            # Remove scripts, styles, heads, navigation elements, footers, and ads
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                element.decompose()
                
            # Find the main body or substantive content
            main_content = soup.find("main") or soup.find("article") or soup.find("div", {"class": "content"}) or soup.body
            if not main_content:
                return "", ""
                
            # Get text and clean redundant spaces
            text = main_content.get_text(separator="\n")
            cleaned_text = re.sub(r'\n+', '\n', text).strip()
            
            return cleaned_text, title
        except Exception as e:
            print(f"Exception raised scraping {url}: {e}")
            return "", ""

    def clean_text(self, text: str) -> str:
        """
        Removes redundant symbols, lines, duplicate sentences, and UI boilerplate tags.
        """
        if not text:
            return ""
            
        # Clean multiple spaces and blank lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        # De-duplicate adjacent redundant lines
        cleaned_lines = []
        for l in lines:
            # Skip very short generic items
            if len(l) < 6 and l.lower() in ["home", "menu", "close", "next", "prev", "back"]:
                continue
            if not cleaned_lines or l != cleaned_lines[-1]:
                cleaned_lines.append(l)
                
        return "\n".join(cleaned_lines)

    def save_educational_document(self, category: str, filename: str, content: str, source_url: str):
        """
        Saves scraped content under targeted data/ subdirectories.
        """
        folder = config.DATA_DIR / category
        folder.mkdir(parents=True, exist_ok=True)
        
        file_path = folder / f"{filename}.txt"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Source URL: {source_url}\n")
            f.write(content)
            
        print(f"Saved educational corpus: {file_path}")

if __name__ == "__main__":
    scraper = EducationalScraper()
    # Test scrape
    text, title = scraper.scrape_url("https://roadmap.sh/ai")
    if text:
        scraper.save_educational_document("roadmaps", "ai_roadmap", text, "https://roadmap.sh/ai")
