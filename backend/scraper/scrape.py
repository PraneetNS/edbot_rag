import os
import sys
import time
import argparse
from pathlib import Path
import requests

# Add parent directory to path so we can import cleaner
sys.path.append(str(Path(__file__).parent.parent))
from scraper.cleaner import clean_html

# Default target URLs mapping
TARGET_PAGES = {
    "homepage": "https://beta.edutainer.in",
    "about": "https://beta.edutainer.in/about",
    "courses": "https://beta.edutainer.in/courses",
    "support": "https://beta.edutainer.in/contact"
}

def scrape_with_requests(url):
    print(f"Scraping {url} using requests...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.text

def scrape_with_selenium(url):
    print(f"Scraping {url} using Selenium (Headless Chrome)...")
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    # Silence logs
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    try:
        driver.get(url)
        # Wait for dynamic React content to render
        time.sleep(5)
        html = driver.page_source
        return html
    finally:
        driver.quit()

def main():
    parser = argparse.ArgumentParser(description="Web Scraper for Edutainer Website")
    parser.add_argument("--mode", choices=["requests", "selenium", "auto"], default="auto",
                        help="Scraping mode (requests, selenium, or auto fallback)")
    parser.add_argument("--pages", nargs="+", help="Specific pages to scrape (homepage, about, courses, support)")
    args = parser.parse_args()

    # Determine pages to scrape
    pages_to_scrape = TARGET_PAGES
    if args.pages:
        pages_to_scrape = {name: url for name, url in TARGET_PAGES.items() if name in args.pages}

    # Output directory
    output_dir = Path(__file__).parent.parent / "docs"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Also support output/ dir in scraper
    scraper_output_dir = Path(__file__).parent / "output"
    scraper_output_dir.mkdir(exist_ok=True, parents=True)

    for name, url in pages_to_scrape.items():
        print(f"\n--- Starting Scrape: {name} ({url}) ---")
        html_content = ""
        mode_used = ""
        
        # Try Selenium first if 'selenium' or 'auto' mode is set
        if args.mode in ["selenium", "auto"]:
            try:
                html_content = scrape_with_selenium(url)
                mode_used = "selenium"
            except Exception as e:
                print(f"Selenium scrape failed for {name}: {e}")
                if args.mode == "auto":
                    print("Falling back to requests...")
                else:
                    continue
        
        # Try requests if 'requests' or if selenium failed in 'auto' mode
        if not html_content and args.mode in ["requests", "auto"]:
            try:
                html_content = scrape_with_requests(url)
                mode_used = "requests"
            except Exception as e:
                print(f"Requests scrape failed for {name}: {e}")
                continue

        if not html_content:
            print(f"Error: Could not retrieve content for {name}")
            continue

        # Clean text
        clean_text = clean_html(html_content)

        # Print quick preview of what we got
        word_count = len(clean_text.split())
        print(f"Successfully scraped using {mode_used}. Got {word_count} words.")
        
        # Save to both docs/ and scraper/output/
        doc_path = output_dir / f"{name}.txt"
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
        print(f"Saved cleaned text to {doc_path}")

        scraper_path = scraper_output_dir / f"{name}.txt"
        with open(scraper_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
            
    print("\nScraping workflow completed.")

if __name__ == "__main__":
    main()
