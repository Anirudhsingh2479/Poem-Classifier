import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import time
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt
import sqlite3
import nltk
import ssl

# Fix SSL certificate issue on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('vader_lexicon')

# ---- Configuration ----
SOURCES = {
    "PoetryFoundation": {"pages": 5},      # Reduced for testing (change to 50 for full scrape)
    "Gutenberg": {"ids": [2600, 4800]},    # Reduced for testing (add more IDs for full scrape)
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# ---- Genre Classifier ----
def classify_poem(text):
    text_lower = text.lower()
    if "love" in text_lower or "heart" in text_lower:
        return "affection"
    elif "music" in text_lower or "song" in text_lower:
        return "music"
    elif "river" in text_lower or "mountain" in text_lower:
        return "environment"
    elif "death" in text_lower or "grave" in text_lower:
        return "death"
    elif "tears" in text_lower or "grief" in text_lower:
        return "sorrow"
    else:
        return random.choice(["affection", "music", "environment", "death", "sorrow"])

# ---- Scrapers ----
@retry(stop=stop_after_attempt(3))
def scrape_poetry_foundation(page):
    url = f"https://www.poetryfoundation.org/poems/browse#page={page}"
    response = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')
    poems = []
    for poem in soup.find_all('div', class_='c-poem'):
        text = poem.get_text(strip=True)
        poems.append([classify_poem(text), f'"{text}"'])
    time.sleep(1.5)
    return poems

@retry(stop=stop_after_attempt(3))
def scrape_gutenberg(id):
    url = f"https://www.gutenberg.org/files/{id}/{id}-0.txt"
    response = requests.get(url, headers=HEADERS, timeout=10)
    poems = []
    for stanza in response.text.split('\n\n'):
        if len(stanza) > 50:
            poems.append([classify_poem(stanza), f'"{stanza[:300]}"'])
    return poems

# ---- Main ----
def main():
    all_poems = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Poetry Foundation
        pf_futures = [executor.submit(scrape_poetry_foundation, page) 
                     for page in range(1, SOURCES["PoetryFoundation"]["pages"] + 1)]
        
        # Gutenberg
        g_futures = [executor.submit(scrape_gutenberg, id) 
                    for id in SOURCES["Gutenberg"]["ids"]]

        # Collect results
        for future in pf_futures + g_futures:
            all_poems.extend(future.result())

    # Save to CSV
    df = pd.DataFrame(all_poems, columns=["genre", "poem"])
    df.to_csv("dataset.csv", index=False)
    print(f"Saved {len(all_poems)} poems to dataset.csv")

if __name__ == "__main__":
    main()