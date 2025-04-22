# -*- coding: utf-8 -*-
# hybrid_scraper_v1_8.py

"""
This script is a hybrid web scraper designed to automate the process of extracting abstracts and publication dates from academic papers.
It integrates multiple techniques, including rule-based scraping, JSON-LD parsing, and optional fallbacks using Crossref API and
a local language model (LLM) for abstract extraction. The script is built using Streamlit for the user interface, allowing users
to upload a CSV file containing bibliographic information and process it to retrieve the required data.

### Key Features:
1. **Rule-Based Scraping**: The script uses a set of predefined CSS selectors and structural rules to extract abstracts and
   publication dates from web pages. It prioritizes JSON-LD metadata, meta tags, and structural HTML elements.
2. **JSON-LD Parsing**: The script extracts abstracts and dates from JSON-LD scripts embedded in web pages, which are often
   used for structured data in scholarly articles.
3. **Crossref API Fallback**: If the input CSV does not contain DOIs or URLs, the script can query the Crossref API to find
   DOIs based on the paper titles. This fallback requires the `habanero` library and a valid email for the API requests.
4. **LLM Fallback**: If the rule-based scraping and Crossref fallback do not yield satisfactory results, the script can use a
   local language model (LLM) to extract abstracts from the filtered HTML content. This fallback requires the `transformers`
   library and a suitable LLM model.
5. **Truncation Detection and Retry**: The script includes logic to detect potentially truncated abstracts and retry the
   extraction using structural rules if truncation is suspected.
6. **Robust Date Parsing**: The script uses the `dateutil` library for robust date parsing, handling various date formats and
   falling back to regex patterns if necessary.
7. **Streamlit UI**: The script provides a user-friendly interface built with Streamlit, allowing users to upload a CSV file,
   configure options, and view the processing results. Users can also download the results as CSV or Excel files.
8. **Logging and Error Handling**: The script includes comprehensive logging and error handling to provide feedback on the
   processing status and help diagnose issues.

### Dependencies:
- streamlit
- pandas
- requests
- beautifulsoup4
- lxml (optional, for faster HTML parsing)
- habanero (optional, for Crossref API)
- fuzzywuzzy (optional, for improved title matching with Crossref)
- transformers (optional, for LLM fallback)
- torch or tensorflow (optional, required by transformers)
- python-dateutil (for robust date parsing)

### Configuration:
The script includes various configuration options, such as request delays, domain-specific delays, headers for HTTP requests,
and thresholds for title similarity and abstract truncation detection. These options can be adjusted to fine-tune the scraping
behavior and ensure compliance with the target websites' rate limits and terms of service.

### Usage:
1. Install the required dependencies using `pip install -r requirements.txt`.
2. Run the script using `streamlit run hybrid_scraper_v1_8.py`.
3. Upload a CSV file containing bibliographic information through the Streamlit UI.
4. Configure the options in the sidebar, such as enabling Crossref and LLM fallbacks.
5. Click the "Start Processing" button to begin the scraping process.
6. View the processing results and download the output as CSV or Excel files.

### Note:
This script is designed for educational and research purposes. Ensure that your use of this script complies with the terms of
service and legal guidelines of the websites being scraped.
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment # Import Comment
import time
import re
import json # Added for JSON-LD parsing
from datetime import datetime
from io import BytesIO
import logging
import os
from urllib.parse import urljoin, urlparse
from dateutil import parser as dateutil_parser # Use dateutil for robust parsing

# --- Library Availability Checks & Setup ---
try: from transformers import pipeline; TRANSFORMERS_AVAILABLE = True
except ImportError: TRANSFORMERS_AVAILABLE = False
try: from habanero import Crossref; HABANERO_AVAILABLE = True
except ImportError: HABANERO_AVAILABLE = False
try: from fuzzywuzzy import fuzz; FUZZYWUZZY_AVAILABLE = True
except ImportError: FUZZYWUZZY_AVAILABLE = False
try: import lxml; LXML_AVAILABLE = True
except ImportError: LXML_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DEFAULT_REQUEST_DELAY = 1.8
# Added more common academic domains and adjusted delays based on perceived sensitivity/rate limits
DOMAIN_SPECIFIC_DELAYS = {
    'dl.acm.org': 5.0, 'ieeexplore.ieee.org': 4.0, 'sciencedirect.com': 3.0,
    'linkinghub.elsevier.com': 3.0, 'link.springer.com': 2.5, 'onlinelibrary.wiley.com': 2.5,
    'nature.com': 2.0, 'pubs.acs.org': 3.0, 'tandfonline.com': 2.5,
    'cambridge.org': 2.0, 'oxfordjournals.org': 2.0, 'sagepub.com': 2.5,
    'plos.org': 1.5, 'mdpi.com': 1.5, 'frontiersin.org': 1.5,
    'arxiv.org': 1.0, 'ssrn.com': 3.0, 'nber.org': 2.0,
    'degruyter.com': 2.5, 'emerald.com': 2.5, 'iopscience.iop.org': 3.0,
    'bmj.com': 2.0, 'jamanetwork.com': 2.5, 'karger.com': 2.0
}
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
           'Accept-Language': 'en-US,en;q=0.9'}
MAX_ROWS_TO_PROCESS = None # Set to a number (e.g., 10) for testing, None for all
LLM_MODEL_NAME = "distilbert-base-cased-distilled-squad"
MAX_TEXT_LENGTH_FOR_LLM = 4000 # For extracted plain text - Used by filter_html_for_llm now
MAX_FILTERED_HTML_TEXT_LENGTH = 5000 # Limit for filtered HTML text passed to LLM
REQUEST_TIMEOUT = 35 # Increased slightly
CROSSREF_EMAIL = "g.rizzo@example.com" # *** REPLACE WITH YOUR ACTUAL EMAIL ***
CROSSREF_TITLE_SIMILARITY_THRESHOLD = 85
LLM_MIN_ABSTRACT_WORDS = 9
LLM_MIN_SHORT_ABSTRACT_WORDS = 7
TRUNCATION_MARKERS = [r'\.\.\.(?:$|\s+(More|Read|View|Continue))', r'\[\.\.\.\]$'] # Regex patterns ending with ... or [...]
ENDS_WITH_PUNCTUATION = r'[.?!]"?\'?\)\]\}]$' # Regex for ending punctuation (slightly more robust)
TRUNCATION_WORD_COUNT_THRESHOLD = 75 # Word count below which truncation is more likely if ending improperly

# Global QA pipeline
qa_pipeline = None

# --- Helper Functions ---

def extract_doi_url(row):
    # --- IDENTICAL to v1.7 ---
    doi_link_col = next((c for c in row.index if c.lower() == 'doi link'), None)
    if doi_link_col and pd.notna(row[doi_link_col]) and isinstance(row[doi_link_col], str):
        url_str = row[doi_link_col]; match = re.search(r'(https?://doi\.org/[^ ]+)', url_str)
        if match: return match.group(1).strip(), "DOI Link"
        if url_str.startswith('doi.org/'): return f"https://{url_str.strip()}", "DOI Link (Schema Added)"
        doi_match = re.search(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', url_str, re.IGNORECASE)
        if doi_match: return f"https://doi.org/{doi_match.group(1).strip()}", "DOI Found in Link"
        # Allow generic URLs only if they look valid
        if re.match(r'^https?://\S+\.\S+$', url_str): return url_str.strip(), "URL" # Check for domain
        if re.match(r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$', url_str, re.IGNORECASE): return f"https://doi.org/{url_str.strip()}", "DOI Link (DOI Only)"
    doi_col = next((c for c in row.index if c.lower() == 'doi'), None)
    if doi_col and pd.notna(row[doi_col]):
        doi = str(row[doi_col]).strip()
        if re.match(r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$', doi, re.IGNORECASE):
            if doi.startswith(('https://doi.org/', 'http://doi.org/')): return doi, "DOI Column (URL)"
            if doi.startswith('doi.org/'): return f"https://{doi}", "DOI Column (Schema Added)"
            return f"https://doi.org/{doi}", "DOI Column"
    url_col = next((c for c in row.index if c.lower() == 'url'), None)
    if url_col and pd.notna(row[url_col]) and isinstance(row[url_col], str) and row[url_col].startswith('http'):
        if re.match(r'^https?://\S+\.\S+$', row[url_col].strip()): # Check for domain
             return row[url_col].strip(), "URL"
    return None, "No Valid URL/DOI Found"

def find_doi_via_crossref(title, first_author=None):
    # --- IDENTICAL to v1.7 ---
    if not HABANERO_AVAILABLE: return None, "Habanero library not installed"
    if not CROSSREF_EMAIL or "@example.com" in CROSSREF_EMAIL:
        logging.warning("Crossref email not set. Using a placeholder, which is discouraged.")
        # Keep the placeholder for now, but ideally, it should error out or disable
    if not title or not isinstance(title, str) or len(title.strip()) < 10: return None, "Insufficient title for Crossref search"
    clean_title = re.sub(r'\s+', ' ', title).strip(); cr = Crossref(); query_params = {'query.bibliographic': clean_title}
    if first_author and isinstance(first_author, str): clean_author = first_author.strip(); query_params['query.author'] = clean_author if len(clean_author) > 2 else None
    logging.info(f"Querying Crossref: title='{clean_title[:50]}...', author='{query_params.get('query.author', 'N/A')}'")
    try:
        limit = 5 if FUZZYWUZZY_AVAILABLE else 1; sort_by = 'score' if FUZZYWUZZY_AVAILABLE else 'relevance'
        results = cr.works(**query_params, limit=limit, mailto=CROSSREF_EMAIL, sort=sort_by, order='desc')
        if results and results['message'] and results['message']['items']:
            items = results['message']['items']
            # Try exact title match first if multiple results
            for item in items:
                 retrieved_title_list = item.get('title')
                 if retrieved_title_list and retrieved_title_list[0].strip().lower() == clean_title.lower():
                     if 'DOI' in item and item['DOI']:
                         logging.info(f"  Crossref exact title match DOI: {item['DOI']}")
                         return item['DOI'], "DOI Found (Exact Title Match)"
            # Fallback to fuzzy matching or first result
            for item in items:
                if 'DOI' in item and item['DOI']:
                    doi = item['DOI']; retrieved_title_list = item.get('title')
                    retrieved_title = retrieved_title_list[0] if retrieved_title_list else None
                    if FUZZYWUZZY_AVAILABLE and retrieved_title:
                         similarity = fuzz.token_sort_ratio(clean_title.lower(), retrieved_title.lower())
                         logging.info(f"  Crossref candidate DOI: {doi}, Title: '{retrieved_title[:50]}...', Similarity: {similarity}")
                         if similarity >= CROSSREF_TITLE_SIMILARITY_THRESHOLD: return doi, f"DOI Found (Similarity: {similarity})"
                    else: logging.info(f"  Crossref candidate DOI: {doi} (No similarity check)"); return doi, "DOI Found (First Result)"
            return None, f"Crossref results found, but similarity < {CROSSREF_TITLE_SIMILARITY_THRESHOLD}" if FUZZYWUZZY_AVAILABLE else "Crossref result found, but no matching DOI/title"
        else: return None, "No results found on Crossref"
    except requests.exceptions.RequestException as e: logging.error(f"Crossref API request failed: {e}"); return None, f"Crossref API Request Error: {str(e)[:100]}..."
    except Exception as e: logging.exception("Crossref API query failed (unexpected)"); return None, f"Crossref API Error: {str(e)[:100]}..."


def safe_request_get(url, headers, timeout=REQUEST_TIMEOUT):
    session = requests.Session()
    session.headers.update(headers)
    session.max_redirects = 10 # Allow more redirects for complex paths
    try:
        # Basic URL validation/cleaning
        if not isinstance(url, str): return None, f"Invalid URL type: {type(url)}"
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            if url.startswith('//'): url = 'https:' + url
            else: url = 'https://' + url
        if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', url): # Basic sanity check
             return None, f"URL appears invalid: {url[:100]}"

        response = session.get(url, timeout=timeout, allow_redirects=True)
        final_url = response.url
        content_type = response.headers.get('Content-Type', '').lower()

        # Check for non-HTML content types first
        if 'pdf' in content_type:
            logging.warning(f"Content is PDF, skipping scraping: {final_url}")
            return None, f"Skipped (Content is PDF)"
        if 'xml' in content_type and 'html' not in content_type:
            logging.warning(f"Content is XML (not HTML), skipping scraping: {final_url}")
            return None, f"Skipped (Content is XML)"
        # Check for other common non-text types
        if any(ct in content_type for ct in ['image/', 'video/', 'audio/', 'application/zip', 'application/octet-stream']):
            logging.warning(f"Content is non-HTML/XML ({content_type}), skipping: {final_url}")
            return None, f"Skipped (Content Type: {content_type.split(';')[0]})"

        # Raise HTTP errors (4xx, 5xx)
        response.raise_for_status()

        # Check for signs of JS challenges or bot protection after successful status
        text_lower = response.text.lower()
        if "checking if the site connection is secure" in text_lower or \
           "enable javascript and cookies" in text_lower or \
           "radware" in text_lower or "validate.perfdrive.com" in final_url or \
           "incapsula" in text_lower or "cloudflare" in response.headers.get('Server', '').lower():
            logging.warning(f"Detected possible JS challenge/Bot protection page: {final_url}")
            # Return response but add a note - maybe some data is still usable
            # Or return None, f"Failed (JS Challenge/Bot Protection?)" - Safer bet
            return None, f"Failed (JS Challenge/Bot Protection?)"

        # Paywall check
        paywall_terms = ["purchase access", "institutional login", "sign in to view", "subscribe to read", "get access", "log in required"]
        paywall_detected = any(term in text_lower for term in paywall_terms)
        note = "Note: Possible Paywall/Login Page Detected" if paywall_detected else None

        return response, note # Return response and potential paywall note

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 403: return None, f"Access Forbidden (403)"
        if status_code == 404: return None, f"Not Found (404)"
        if status_code == 429: return None, f"Too Many Requests (429)"
        if status_code == 503: return None, f"Service Unavailable (503)"
        return None, f"Request Error: Status {status_code}"
    except requests.exceptions.Timeout: return None, f"Timeout Error ({timeout}s)"
    except requests.exceptions.TooManyRedirects: return None, f"Too Many Redirects"
    except requests.exceptions.ConnectionError as e: return None, f"Connection Error: {str(e)[:100]}..."
    except requests.exceptions.RequestException as e: return None, f"Request Error: {str(e)[:100]}..."
    except Exception as e: logging.exception(f"Unexpected error during request for {url}"); return None, f"Unexpected Request Error: {str(e)[:100]}..."


def parse_date(date_str):
    # Use dateutil.parser for more robust parsing, fallback to regex
    if not date_str or not isinstance(date_str, str): return None, None
    current_year = datetime.now().year
    min_year = 1800
    max_year = current_year + 2 # Allow slightly into the future

    try:
        # Try dateutil parser first
        dt = dateutil_parser.parse(date_str, ignoretz=True, fuzzy=False) # fuzzy=False is stricter
        if dt and min_year < dt.year <= max_year:
            return dt.year, dt.month
    except (ValueError, OverflowError, TypeError):
        pass # If dateutil fails, fall through to regex

    # Regex patterns (kept for specific cases or fallbacks)
    patterns = [
        # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
        r'(\d{4})[-/.]([01]?\d)[-/.]([0-3]?\d)',
        # Month Day, Year (e.g., Jan 15, 2023 or January 15 2023)
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-zA-Z]*\.?\s+([0-3]?\d),?\s+(\d{4})',
        # Day Month Year (e.g., 15 Jan 2023 or 15 January, 2023)
        r'([0-3]?\d)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-zA-Z]*\.?,?\s+(\d{4})',
        # YYYY-MM or YYYY/MM
        r'(\d{4})[-/]([01]?\d)',
        # Month Year (e.g., January 2023 or Jan 2023)
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-zA-Z]*\.?\s+(\d{4})\b',
        # Keywords followed by Year (e.g., published: 2023)
        r'\b(published|issued|publication date|received|accepted|created|modified)\s*[:\-]?\s*(\d{4})\b',
         # Just Year (less reliable, use last)
        r'\b(\d{4})\b'
    ]
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

    for pattern in patterns:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            groups = match.groups()
            year, month = None, None
            try:
                g_len = len(groups)
                if g_len == 3: # YYYY-MM-DD variations
                    if groups[0].isdigit() and groups[1].isdigit(): year, month = int(groups[0]), int(groups[1])
                    elif groups[0].isalpha(): month = month_map.get(groups[0][:3].lower()); year = int(groups[2])
                    else: month = month_map.get(groups[1][:3].lower()); year = int(groups[2])
                elif g_len == 2: # YYYY-MM or Month Year variations
                    if groups[0].isdigit() and groups[1].isdigit(): year, month = int(groups[0]), int(groups[1])
                    elif groups[0].isalpha() and groups[1].isdigit(): month = month_map.get(groups[0][:3].lower()); year = int(groups[1])
                    elif any(kw in groups[0].lower() for kw in ["published", "issued", "publication date", "received", "accepted", "created", "modified"]) and groups[1].isdigit(): year = int(groups[1])
                elif g_len == 1 and groups[0].isdigit(): # Just YYYY
                     potential_year = int(groups[0])
                     # Add stricter check for standalone year to avoid random 4-digit numbers
                     if min_year < potential_year <= max_year and ('year' in date_str.lower() or len(date_str) < 15): # Heuristic
                          year = potential_year; month = None

                if year and (min_year < year <= max_year):
                     if month and not (1 <= month <= 12): month = None
                     return year, month
                else: year, month = None, None # Reset if year invalid
            except (ValueError, TypeError, IndexError):
                year, month = None, None; continue # Try next pattern

    # Final fallback: Try fuzzy parsing if strict failed
    try:
        dt = dateutil_parser.parse(date_str, ignoretz=True, fuzzy=True) # fuzzy=True is more lenient
        if dt and min_year < dt.year <= max_year:
            return dt.year, dt.month
    except (ValueError, OverflowError, TypeError, AttributeError): # Added AttributeError
        pass

    logging.warning(f"Could not parse date from string: {date_str[:100]}")
    return None, None


@st.cache_resource
def load_qa_pipeline():
    # --- IDENTICAL to v1.7 ---
    if not TRANSFORMERS_AVAILABLE: return None
    try:
        logging.info(f"Attempting to load QA model: {LLM_MODEL_NAME}")
        qa_pipe = pipeline("question-answering", model=LLM_MODEL_NAME, tokenizer=LLM_MODEL_NAME)
        logging.info("QA model loaded successfully.")
        return qa_pipe
    except NameError as ne: st.error(f"Error loading QA model ({LLM_MODEL_NAME}): {ne}. Check library imports. LLM fallback disabled."); logging.error(f"LLM Load Error (NameError): {ne}"); return None
    except Exception as e: st.error(f"Error loading QA model ({LLM_MODEL_NAME}): {e}. LLM fallback disabled."); logging.exception("LLM Load Error"); return None

def get_llm_abstract(text_content, context_type="Text"):
    # --- IDENTICAL to v1.7 ---
    global qa_pipeline
    if qa_pipeline is None: return None, "LLM Skipped (Not Loaded)"
    if not text_content or len(text_content) < 50: return None, f"LLM Skipped ({context_type} too short)"

    # Truncate context if it exceeds the model's likely capacity (adjust based on model if known)
    # DistilBERT max sequence length is 512 tokens, roughly corresponds to text length
    max_context_len = 3000 # Be conservative to avoid tokenization issues
    truncated_context = text_content[:max_context_len]
    if len(text_content) > max_context_len:
        logging.warning(f"LLM context truncated from {len(text_content)} to {max_context_len} chars.")

    try:
        question = "What is the abstract of this document?"
        result = qa_pipeline(question=question, context=truncated_context, max_answer_len=600, handle_impossible_answer=True) # Handle cases where answer isn't found
        status_prefix = f"LLM ({context_type})" # Add context info to status

        # Check for null answer or very low score
        if result is None or result['answer'] is None or result['score'] < 0.01: # Lowered threshold slightly, rely more on word count
            return None, f"{status_prefix} Failed (Not found or low confidence)"

        answer = result['answer'].strip()
        # More robust cleaning of LLM answer artifacts
        answer = re.sub(r'^(abstract|summary|introduction|background)\s*[:\-]?\s*', '', answer, flags=re.IGNORECASE).strip()
        answer = re.sub(r'^(here is|the abstract is)\s*[:\-]?\s*', '', answer, flags=re.IGNORECASE).strip()
        # Remove potential incomplete sentence starts if it looks like context bleed
        if answer.startswith("..."): answer = answer[3:].lstrip()

        if "abstract of this document" in answer.lower() or "provide an abstract" in answer.lower() or "cannot provide an abstract" in answer.lower() or "context does not contain" in answer.lower():
            return None, f"{status_prefix} Failed (Found question/noise/refusal)"

        answer_words = answer.split(); answer_word_count = len(answer_words)

        if answer_word_count >= LLM_MIN_ABSTRACT_WORDS:
            return answer, f"Success ({status_prefix}, Score: {result['score']:.2f})"
        elif answer_word_count >= LLM_MIN_SHORT_ABSTRACT_WORDS:
            return answer, f"Success ({status_prefix} - Short, Score: {result['score']:.2f})"
        else:
            return None, f"{status_prefix} Failed (Answer too short, Score: {result['score']:.2f})"
    except Exception as e:
        error_message = f"LLM ({context_type}) Error: {str(e)[:100]}..."
        logging.exception("LLM Inference Error")
        return None, error_message


# --- NEW Function: Extract JSON-LD ---
def extract_json_ld_abstract_and_date(soup):
    """Extracts abstract and date from JSON-LD scripts."""
    abstract, date_str = None, None
    try:
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            if script.string:
                try:
                    data = json.loads(script.string)
                    # Handle both single object and list of objects
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        # Check common types for scholarly articles
                        item_type = item.get('@type')
                        if item_type and ('Article' in item_type or 'ScholarlyArticle' in item_type or 'PublicationIssue' in item_type or 'WebPage' in item_type):
                            # Find abstract - common keys: 'abstract', 'description'
                            if not abstract: # Only take the first good one
                                potential_abstract = item.get('abstract') or item.get('description')
                                if isinstance(potential_abstract, dict): # Sometimes it's nested e.g. {'@value': '...'}
                                    potential_abstract = potential_abstract.get('@value')
                                if potential_abstract and isinstance(potential_abstract, str) and len(potential_abstract.split()) > 10:
                                    abstract = potential_abstract.strip()
                                    logging.info("Found abstract in JSON-LD.")

                            # Find date - common keys: 'datePublished', 'dateCreated', 'dateModified'
                            if not date_str: # Prioritize published, then created, then modified
                                potential_date = item.get('datePublished') or item.get('dateCreated') or item.get('dateModified')
                                if potential_date and isinstance(potential_date, str):
                                    date_str = potential_date.strip()
                                    logging.info(f"Found date in JSON-LD: {date_str}")

                            # If we found both, we can stop searching this script block
                            if abstract and date_str:
                                break # Exit inner loop (items)
                    if abstract and date_str:
                         break # Exit outer loop (scripts)
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON-LD parsing failed: {e} - Content: {script.string[:100]}...")
                except Exception as e:
                    logging.exception(f"Unexpected error processing JSON-LD: {e}")
        return abstract, date_str
    except Exception as e:
        logging.exception(f"Error finding/processing JSON-LD tags: {e}")
        return None, None


# --- Rule-Based Scraping Function (v5 Logic - incorporating JSON-LD and more selectors) ---
def _search_abstract_rules(soup, response_url):
    """Internal function to find abstract using various rule-based methods."""
    json_ld_abstract, json_ld_date_str = None, None
    meta_abstract, structural_abstract = None, None
    best_structural_source = "Unknown"
    heading_sibling_abstract = None

    # 1. JSON-LD (Highest Priority)
    json_ld_abstract, json_ld_date_str = extract_json_ld_abstract_and_date(soup)
    if json_ld_abstract:
         logging.info("Using abstract found via JSON-LD.")
         # We'll return this separately and let the orchestrator decide

    # 2. Meta Tags (High Priority - Specific tags first)
    # Prioritize citation_abstract, then og:description, then general description
    meta_priority_tags = ['citation_abstract', 'og:description', 'description', 'DC.Description', 'dcterms.abstract']
    for tag_name in meta_priority_tags:
        meta_tag = soup.find('meta', attrs={'name': tag_name}) or \
                   soup.find('meta', attrs={'property': tag_name}) or \
                   soup.find('meta', attrs={'itemprop': tag_name}) # Added itemprop
        if meta_tag and meta_tag.get('content'):
            potential_meta = meta_tag['content'].strip()
            word_count = len(potential_meta.split())
            # Basic quality check: needs reasonable length, avoid short titles/descriptions
            if word_count > 15 and not potential_meta.lower().startswith("keywords:"):
                meta_abstract = potential_meta
                logging.info(f"Found potential abstract in Meta Tag: {tag_name}")
                break # Stop after finding the first good meta abstract

    # 3. Structural Selectors (Broad search, categorized)
    # More specific / reliable selectors first
    known_selectors = [
        'div.abstractSection',                 # Common generic
        'section.abstract',                   # Common semantic HTML5
        'div.abstract',                       # Common generic
        '#abstract',                          # Common ID
        'div#abs',                            # Common ID variation
        '#Abs1-content',                      # Springer?
        'div.abstract-text',                  # IEEE, Thieme?
        'div.article__abstract',              # Newer generic?
        '.abstract-group .abstract',          # Some journals
        '#abstracts .abstract-content',       # ScienceDirect?
        'div.abstract-content',               # MDPI, Preprints?
        'div.art-abstract',                   # MDPI specific
        'div.abstract.module',                # Wiley?
        'div.abstractInFull',                 # NowPublishers (T&F)?
        'div.AbstractText',                   # Frontiers
        'div.abstract.markup',                # Emerald
        'section[aria-labelledby="abstract-heading"]', # Nature?
        'section[aria-labelledby="Abs1-title"]',   # BMC?
        'div.abstract > div > p',             # Cambridge? (might be too broad)
        'section.article-abstract',           # OJS platforms
        'div#abstract > p',                   # OJS/FirstMonday variations
        'div.abstractbox',                    # IJRASet?
        'div.tab-content[role="tabpanel"]',   # Check if contains 'Abstract' header - Lexxion?
        'div.talk-the-abstract',              # HStalks?
    ]
    # More general selectors (use with caution)
    general_selectors = [
        'div[class*="abstract"]',             # Contains 'abstract' in class (can be noisy)
        'section[class*="abstract"]',
        'div[id*="abstract"]',                # Contains 'abstract' in ID
        'section[id*="abstract"]',
    ]

    # Special case for ArXiv blockquote
    if response_url and 'arxiv.org' in response_url:
        arxiv_selector = 'blockquote.abstract'
        known_selectors.insert(0, arxiv_selector) # Add to front for ArXiv

    all_selectors = known_selectors + general_selectors
    best_structural_abstract = None

    for selector in all_selectors:
        container = None
        try:
            container = soup.select_one(selector)
        except Exception as e:
            logging.warning(f"CSS Selector failed: '{selector}' - Error: {e}")
            continue

        if container:
            # Extract text carefully, joining paragraphs if necessary
            text_parts = []
            # Prioritize paragraphs within the container
            paragraphs = container.find_all('p', recursive=False) # Direct children first
            if paragraphs:
                 for p in paragraphs: text_parts.append(p.get_text(" ", strip=True))
            else:
                 # If no direct 'p' children, get all text, but be wary of noise
                 text_parts.append(container.get_text(" ", strip=True))

            text = " ".join(text_parts).strip()

            # Clean common prefixes
            text = re.sub(r'^\s*(abstract|summary)\s*[:\-]?\s*', '', text, flags=re.IGNORECASE).strip()

            if len(text.split()) > 15: # Minimum length check
                 # Prefer longer abstracts or those from known selectors
                 is_better = False
                 if best_structural_abstract is None:
                     is_better = True
                 else:
                     # Prefer if longer
                     if len(text) > len(best_structural_abstract): is_better = True
                     # Prefer if current is from known selector and previous was general
                     if selector in known_selectors and best_structural_source not in known_selectors: is_better = True
                     # Overwrite if same length but current is from known selector
                     if len(text) == len(best_structural_abstract) and selector in known_selectors and best_structural_source not in known_selectors: is_better = True

                 if is_better:
                     best_structural_abstract = text
                     best_structural_source = f"Selector ({selector})"
                     logging.info(f"Found potential structural abstract via: {best_structural_source}")

    structural_abstract = best_structural_abstract

    # 4. Heading + Sibling (Lower Priority Fallback)
    # Only run if structural search yielded nothing substantial
    if not structural_abstract or len(structural_abstract.split()) < 30:
        try:
            # Find headings like Abstract, Summary
            heading = soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'strong', 'div', 'span'],
                                string=re.compile(r'^\s*(Abstract|Summary)\s*$', re.IGNORECASE))
            if heading:
                logging.info("Found abstract heading, looking for siblings...")
                # Look for the next <p> or <div> sibling containing substantial text
                next_sibling = heading.find_next_sibling()
                potential_heading_abstract = ""
                while next_sibling and len(potential_heading_abstract.split()) < 15:
                    if next_sibling.name == 'p' or next_sibling.name == 'div':
                        sibling_text = next_sibling.get_text(" ", strip=True)
                        if len(sibling_text.split()) > 10: # Check if sibling itself has enough text
                             potential_heading_abstract = sibling_text
                             break # Found a good sibling
                    # Stop if we hit another heading
                    if next_sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5']:
                         break
                    next_sibling = next_sibling.find_next_sibling()

                if potential_heading_abstract:
                     logging.info("Found potential abstract via Heading+Sibling.")
                     heading_sibling_abstract = potential_heading_abstract
                     # Prefer this only if structural was very short or non-existent
                     if structural_abstract is None or len(heading_sibling_abstract) > len(structural_abstract) * 1.5 :
                          # This logic is now handled in the orchestrator function
                          pass # Just make it available
        except Exception as e:
             logging.exception("Error during Heading+Sibling search.")


    # Return all found parts, let the orchestrator prioritize
    return json_ld_abstract, json_ld_date_str, meta_abstract, structural_abstract, best_structural_source, heading_sibling_abstract


# --- Targeted Structural Retry Function ---
# Simplified: We now rely on the main search having better selectors,
# the main value of retry is if the *first* attempt was truncated meta/json.
def retry_structural_search(soup, response_url):
    """Performs a focused structural search, typically used if initial meta/json seems truncated."""
    logging.info("Retrying structural search...")
    # Use the same structural search logic, but without meta/json-ld/heading checks
    _, _, _, structural_abstract, best_structural_source, _ = _search_abstract_rules(soup, response_url)
    # We only care about the structural result here
    if structural_abstract:
        logging.info(f"Retry structural search found abstract via {best_structural_source}")
        return structural_abstract, f"Retry {best_structural_source}"
    else:
        logging.warning("Retry structural search did not find an abstract.")
        return None, "Retry Failed"


# --- Function: Filter HTML ---
def filter_html_for_llm(soup):
    """Removes common noise elements from HTML for better LLM context."""
    if not soup: return None
    soup_copy = BeautifulSoup(str(soup), 'lxml' if LXML_AVAILABLE else 'html.parser')

    selectors_to_remove = [
        "script", "style", "header", "footer", "nav", "aside", "form", "figure",
        "figcaption", "button", "input", "iframe", "noscript", "svg", "img", "audio", "video",
        ".related-articles", ".sidebar", ".references", ".citations", ".supplementary-material",
        ".banner", ".ad", ".cookie-consent", ".site-header", ".site-footer", ".navigation",
        "#acknowledgments", "#supplementary-material", "#comments", ".comment-section",
        ".footer", ".header", ".menu", ".breadcrumb", ".tabs", ".pagination",
        "[role='navigation']", "[role='banner']", "[role='contentinfo']", "[role='complementary']",
        "[role='dialog']", "[role='menu']", "[aria-hidden='true']",
        # Common ad/noise patterns
        '[id*="ad"]', '[class*="ad"]', '[id*="banner"]', '[class*="banner"]',
        '[id*="promo"]', '[class*="promo"]', '[id*="sponsor"]', '[class*="sponsor"]',
        '[id*="cookie"]', '[class*="cookie"]', '[class*="social"]', '[class*="share"]',
        '[class*="related"]', '[class*="popular"]', '[class*="trending"]', '[class*="modal"]',
        '[class*="popup"]'
    ]
    # Remove comments first
    for comment in soup_copy.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    # Remove selected tags/elements
    for selector in selectors_to_remove:
        try:
            for element in soup_copy.select(selector):
                element.decompose()
        except Exception as e:
            logging.warning(f"Error removing elements with selector '{selector}': {e}")

    # Attempt to find the main content area (heuristic)
    main_content = soup_copy.find('main') or \
                   soup_copy.find('article') or \
                   soup_copy.select_one('div[role="main"]') or \
                   soup_copy.select_one('div#main') or \
                   soup_copy.select_one('div.main') or \
                   soup_copy.select_one('div#content') or \
                   soup_copy.select_one('div.content')

    if main_content:
        target_element = main_content
        logging.info("Using main content area for LLM context.")
    else:
        target_element = soup_copy.find("body") or soup_copy # Fallback to body or root
        logging.info("Using body/root for LLM context.")

    if not target_element: return None

    main_text = target_element.get_text(" ", strip=True)
    # Aggressive whitespace cleaning
    main_text = re.sub(r'\s\s+', ' ', main_text).strip()
    main_text = re.sub(r'(\n\s*){2,}', '\n', main_text) # Reduce multiple blank lines

    logging.info(f"Filtered HTML text length for LLM: {len(main_text)}")
    # Use MAX_FILTERED_HTML_TEXT_LENGTH here, as it was defined for this purpose
    return main_text[:MAX_FILTERED_HTML_TEXT_LENGTH]


# --- Main Scraping Orchestrator Function (v1.8) ---
def scrape_abstract_and_date(soup, response_url):
    """Orchestrates scraping: JSON-LD -> Rules -> Truncation Check -> Retry -> Date Extraction."""
    final_abstract, scraped_year, scraped_month = None, None, None
    final_status = "Initial State"
    source = "Unknown"
    was_truncated = False # Flag if initial finding seems truncated

    # 1. Initial Rule-Based Search (including JSON-LD inside)
    json_ld_abstract, json_ld_date_str, meta_abstract, structural_abstract, structural_source, heading_sibling_abstract = _search_abstract_rules(soup, response_url)

    # 2. Prioritization Logic
    initial_abstract = None
    if json_ld_abstract:
        initial_abstract = json_ld_abstract
        source = "JSON-LD"
    elif meta_abstract and ('citation_abstract' in meta_abstract or 'dcterms.abstract' in meta_abstract): # Prioritize specific meta tags
         initial_abstract = meta_abstract
         source = "Meta (Specific)"
    elif structural_abstract and structural_source != "Unknown": # Any identified structural element
         initial_abstract = structural_abstract
         source = structural_source
    elif meta_abstract: # Fallback to other meta tags (like description)
        initial_abstract = meta_abstract
        source = "Meta (General)"
    elif heading_sibling_abstract: # Fallback to heading search
        initial_abstract = heading_sibling_abstract
        source = "Heading+Sibling"

    # 3. Date Extraction - Prioritize JSON-LD date if found
    date_str = json_ld_date_str # Start with JSON-LD date if available
    date_source = "JSON-LD" if date_str else "Unknown"

    # If JSON-LD didn't provide a date, try meta tags
    if not date_str:
        meta_date_tags = [
            'citation_publication_date', 'datePublished', 'article:published_time',
            'DC.Date', 'dcterms.issued', 'dcterms.created', 'dcterms.date',
            'og:published_time'
        ]
        for tag_name in meta_date_tags:
            meta_tag = soup.find('meta', attrs={'name': tag_name}) or \
                       soup.find('meta', attrs={'property': tag_name}) or \
                       soup.find(itemprop=tag_name)
            if meta_tag:
                 date_val = meta_tag.get('content', meta_tag.get('datetime', meta_tag.string))
                 if date_val and isinstance(date_val, str):
                     date_str = date_val.strip()
                     if date_str:
                         date_source = f"Meta ({tag_name})"
                         logging.info(f"Found date string via Meta: {date_str}")
                         break # Stop searching meta tags for date
    # If still no date, try <time> tag
    if not date_str:
        time_tag = soup.find('time', datetime=True)
        if time_tag and time_tag.has_attr('datetime') and time_tag['datetime']:
            date_str = time_tag['datetime'].strip()
            if date_str:
                 date_source = "Time Tag"
                 logging.info(f"Found date string via Time Tag: {date_str}")

    # If still no date, try ArXiv specific dateline
    if not date_str and response_url and 'arxiv.org' in response_url:
        dateline = soup.find('div', class_='dateline')
        if dateline:
             # Updated Regex for ArXiv dates (V1, V2, Submitted, etc.)
             match = re.search(r'\(\s*(?:Submitted on|v\d+\s*(?:last revised|announced))\s+([^)]+)\)', dateline.get_text(strip=True), re.IGNORECASE)
             if match:
                 date_str = match.group(1).strip()
                 if date_str:
                      date_source = "ArXiv Dateline"
                      logging.info(f"Found date string via ArXiv Dateline: {date_str}")

    # Add more specific date rules here if needed (e.g., searching text patterns)
    # Example: Look for "Published: DD Month YYYY" in specific divs if other methods fail
    # if not date_str:
    #    pub_info_div = soup.select_one('.publication-dates') or soup.select_one('.article-meta')
    #    if pub_info_div:
    #        pub_text = pub_info_div.get_text(" ", strip=True)
    #        match = re.search(r'(Published|Online|Issued):\s*(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})', pub_text, re.IGNORECASE)
    #        if match:
    #             date_str = match.group(2).strip()
    #             date_source = "Text Pattern"
    #             logging.info(f"Found date string via Text Pattern: {date_str}")


    # Parse the found date string
    if date_str:
        scraped_year, scraped_month = parse_date(date_str)
        if not scraped_year:
             logging.warning(f"Failed to parse date '{date_str}' found via {date_source}.")
             date_source += " (Parse Failed)"
        else:
             logging.info(f"Parsed date: Year={scraped_year}, Month={scraped_month} from {date_source}")
    else:
        date_source = "Not Found"
        logging.info("Date string not found by any rule.")

    # 4. Truncation Check and Retry Logic (Applied to the 'initial_abstract')
    final_abstract = initial_abstract # Start with the best abstract found so far

    if initial_abstract:
         word_count = len(initial_abstract.split())
         # Ends improperly check: not ending with standard punctuation
         ends_improperly = not bool(re.search(ENDS_WITH_PUNCTUATION, initial_abstract))
         # Ends with truncation marker check
         ends_with_marker = any(re.search(pattern, initial_abstract, re.IGNORECASE) for pattern in TRUNCATION_MARKERS)

         # Determine if truncation is suspected
         # Suspect if:
         # - Ends with a specific marker (strong indicator)
         # - OR (From Meta/JSON-LD OR word count is low) AND it ends improperly
         is_potentially_truncated = ends_with_marker or \
             ( (source in ["JSON-LD", "Meta (Specific)", "Meta (General)"] or word_count < TRUNCATION_WORD_COUNT_THRESHOLD) and ends_improperly )

         if is_potentially_truncated:
            logging.warning(f"Truncation suspected for '{source}' abstract (ends_marker={ends_with_marker}, ends_improperly={ends_improperly}, words={word_count}).")
            was_truncated = True # Mark that truncation was suspected

            # Only retry structural search if the truncated source wasn't already structural
            if source not in ["Selector", "Heading+Sibling"] and source.startswith("Selector") == False:
                 retry_abstract, retry_source = retry_structural_search(soup, response_url)
                 if retry_abstract and len(retry_abstract) > len(initial_abstract) * 1.15: # Use if retry is clearly better
                     logging.info(f"Retry successful: Replacing truncated '{source}' abstract with longer one from {retry_source}")
                     final_abstract = retry_abstract
                     source = retry_source # Update source to reflect retry success
                     final_status = f"Success ({source} after Retry)" # Set status reflecting successful retry
                 else:
                     logging.warning(f"Retry failed to find a significantly longer abstract. Keeping potentially truncated abstract from '{source}'.")
                     # Keep initial abstract but update status
                     final_status = f"Success ({source} - Truncation Suspected, Retry Failed)"
            else:
                 # The suspected abstract was already structural, no retry needed, just flag status
                 logging.warning(f"Truncation suspected for structural abstract '{source}'. No retry attempted.")
                 final_status = f"Success ({source} - Truncation Suspected)"
         else: # Initial abstract seems okay (not truncated)
             final_status = f"Success ({source})"
    else: # No abstract found by any rule
         final_status = "Abstract Not Found (Rules)"
         source = "Not Found"


    # 5. Combine Final Status with Date Info
    abstract_found = bool(final_abstract)
    date_found = bool(scraped_year)

    if abstract_found and date_found:
         # Status is already set based on abstract source and truncation check
         final_status += f" / Date ({date_source})"
    elif abstract_found and not date_found:
         final_status = f"{final_status.replace('Success (','').replace(')','')} / Date Not Found" # Clean up abstract status part if needed
    elif not abstract_found and date_found:
         final_status = f"Abstract Not Found / Date ({date_source})"
    else: # Neither found
         final_status = "Abstract & Date Not Found (Rules)"

    # Return the final abstract, date components, status, and the truncation flag
    return final_abstract, scraped_year, scraped_month, final_status, was_truncated


# --- convert_df_to_excel - Identical to v1.5 / v1.7 ---
def convert_df_to_excel(df):
    output = BytesIO()
    df_excel = df.copy() # Work on a copy
    # Ensure potentially long string columns are handled before writing
    string_cols = ['Scraped_Abstract', 'Title', 'Authors', 'Venue', 'Scrape_Status', 'Processed_URL', 'DOI', 'DOI link'] # Add any other potential long strings
    for col in string_cols:
        if col in df_excel.columns:
            # Convert to string, fill NA, slice. Handle potential non-string data gracefully.
            df_excel[col] = df_excel[col].apply(lambda x: str(x)[:32766] if pd.notna(x) else '')

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_excel.to_excel(writer, index=False, sheet_name='Sheet1')
        worksheet = writer.sheets['Sheet1']
        # Adjust column widths (optional but helpful)
        for idx, col in enumerate(df_excel.columns):
            series = df_excel[col]
            # Calculate max length robustly
            try:
                 max_len = max((series.astype(str).map(len).max(), len(str(series.name)))) + 1
                 # Clamp max width to avoid excessively wide columns
                 width = min(max(max_len, 10), 70) # Min width 10, max 70
                 worksheet.set_column(idx, idx, width)
            except Exception as e:
                 logging.warning(f"Could not set width for column '{col}': {e}")
                 worksheet.set_column(idx, idx, 20) # Default width on error

    return output.getvalue()


# --- Streamlit App UI and Main Logic ---

st.set_page_config(layout="wide", page_title="Hybrid Abstract Scraper v1.8")
st.title("Hybrid Bibliography Abstract & Date Scraper (v1.8)")
st.markdown("""
Upload a CSV file containing bibliographic information. The script requires **DOI**, **DOI link**, or a valid **URL** column to find articles.
It attempts to scrape the **Abstract** and **Publication Year/Month**.
- **Enhancements:** Prioritizes `JSON-LD` metadata, uses expanded CSS selectors, robust date parsing (`dateutil`), improved truncation detection with retry, and clearer status messages.
- **Fallbacks:** Uses **Crossref** API (if enabled and DOI/URL is missing, requires 'Title') and optional **LLM Fallback** (if rules/retry fail or find a truncated abstract).
""")

# Placeholders
error_placeholder = st.empty()
info_placeholder = st.empty()
warning_placeholder = st.empty()

# --- Sidebar Options ---
st.sidebar.header("Options & Status")
libs_ok = True
if not TRANSFORMERS_AVAILABLE: st.sidebar.warning("`transformers` or `torch`/`tensorflow` not found. LLM option disabled."); libs_ok = False
if not HABANERO_AVAILABLE: st.sidebar.warning("`habanero` not found. Crossref option disabled."); libs_ok = False
if not FUZZYWUZZY_AVAILABLE: st.sidebar.info("`fuzzywuzzy` not found. Crossref title matching will be less precise.")
if not LXML_AVAILABLE: st.sidebar.info("`lxml` not found. Using Python's built-in HTML parser (might be slower/less robust).")
try: import dateutil; st.sidebar.success("`python-dateutil` found.") # Check for dateutil explicitly
except ImportError: st.sidebar.warning("`python-dateutil` not found. Date parsing may be less reliable."); libs_ok = False


use_llm_fallback = False
if TRANSFORMERS_AVAILABLE:
    use_llm_fallback = st.sidebar.checkbox("Enable LLM Fallback (Slow, Experimental)", value=False, key="llm_checkbox",
                                           help="Use local AI model if rules/retry fail to find an abstract, OR if a potentially truncated abstract is found and structural retry doesn't improve it significantly. Requires significant compute resources.")

use_crossref = False
if HABANERO_AVAILABLE:
    # Make Crossref email editable in sidebar
    st.sidebar.subheader("Crossref Configuration")
    # crossref_email_input = st.sidebar.text_input("Crossref Email (Required for API)", value=CROSSREF_EMAIL, key="crossref_email_input")
    # Use a placeholder if not provided, but strongly recommend setting one.
    # if crossref_email_input and "@" in crossref_email_input:
    #     CROSSREF_EMAIL = crossref_email_input
    # else:
    #     st.sidebar.warning("Valid Crossref email recommended for reliable API access.")
    #     CROSSREF_EMAIL = "anonymous@example.com" # Default anonymous if not set

    use_crossref = st.sidebar.checkbox("Enable Crossref DOI Search (Requires Title)", value=True, key="crossref_checkbox",
                                       help=f"Query Crossref API if DOI/URL is missing in the input. Requires a 'Title' column. Uses email: {CROSSREF_EMAIL}")


with st.sidebar.expander("Help & Tips"):
    st.markdown(f"""
*   **Input CSV:** Must have headers. Key columns (case-insensitive): `DOI`, `DOI link`, `URL`, `Title`, `Authors`, `Year`, `Venue`. Presence of `DOI`, `DOI link`, or `URL` is crucial for finding the article page.
*   **Scraping Priority:** JSON-LD > Meta Tags (Specific) > Structural (Known Selectors) > Meta Tags (General) > Structural (General) > Heading+Sibling.
*   **Truncation Check:** Script checks if the found abstract ends abruptly (e.g., missing punctuation, ends with '...') especially if short or from meta/JSON sources. If suspected, it retries with structural rules.
*   **LLM Fallback Trigger:** Activated if (1) no abstract is found by any rule/retry, OR (2) an abstract is found but flagged as potentially truncated *and* the structural retry did not yield a significantly better result. It uses the filtered HTML text as context.
*   **Crossref:** Uses 'Title' and optionally 'Authors' to find a DOI if none is provided in the input. Accuracy depends on Crossref data and title similarity ({CROSSREF_TITLE_SIMILARITY_THRESHOLD}% threshold with `fuzzywuzzy`).
*   **LLM Performance:** Can be slow, especially without a GPU. Accuracy varies based on the webpage content and model capability. Model used: `{LLM_MODEL_NAME}`.
*   **Output Columns:** Includes `Scraped_Abstract`, `Scraped_Year`, `Scraped_Month`, `Scrape_Status` (detailed outcome), `Processed_URL` (the URL used for scraping), `DOI` (original or from Crossref).
*   **Delays:** Uses a default delay of {DEFAULT_REQUEST_DELAY}s between requests, with longer delays for specific domains (see `DOMAIN_SPECIFIC_DELAYS`) to be polite to servers.
*   **Limitations:** Cannot scrape PDFs, sites heavily reliant on JavaScript for content rendering, pages behind hard paywalls/logins, or sites with strong bot protection. Status messages will indicate these issues.
*   **Status Codes:** Look for codes like `(JSON-LD)`, `(Selector:...)`, `(Meta)`, `(Skipped...)`, `(Failed...)`, `(Truncation Suspected)`, `(LLM Success/Failed)`.
    """)

# --- File Upload ---
uploaded_file = st.file_uploader("Choose a bibliography CSV file", type="csv", key="file_uploader")

# --- Main Processing Logic ---
if uploaded_file is not None:
    # Clear previous messages
    error_placeholder.empty()
    info_placeholder.empty()
    warning_placeholder.empty()

    try:
        # Attempt to read CSV with standard UTF-8, fallback to latin1
        try:
            input_df_original = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            try:
                input_df_original = pd.read_csv(uploaded_file, encoding='latin1')
                info_placeholder.info("Read CSV with 'latin1' encoding due to UTF-8 decode error.")
            except Exception as e_read:
                 error_placeholder.error(f"Error reading CSV file (tried UTF-8 and latin1): {e_read}")
                 st.stop()
        except pd.errors.EmptyDataError:
             error_placeholder.error("Error: Uploaded CSV file is empty.")
             st.stop()
        except Exception as e_read:
            error_placeholder.error(f"Error reading CSV file: {e_read}")
            st.stop()

        if input_df_original.empty:
            error_placeholder.error("Uploaded CSV file contains no data.")
            st.stop()

        input_df = input_df_original.copy() # Work on a copy
        st.write("Uploaded Bibliography Preview (first 5 rows):")
        st.dataframe(input_df.head(5))

        # --- Column Identification & Renaming (More robust) ---
        def find_and_rename_col(df, standard_name, variations):
            for var in variations:
                found_col = next((c for c in df.columns if c.lower() == var.lower()), None)
                if found_col:
                    if found_col != standard_name: # Only rename if not already the standard name
                        if standard_name in df.columns:
                             warning_placeholder.warning(f"Both '{found_col}' and '{standard_name}' found. Using '{standard_name}'.")
                             return standard_name # Prefer existing standard name
                        else:
                             logging.info(f"Renaming column '{found_col}' to '{standard_name}'")
                             df.rename(columns={found_col: standard_name}, inplace=True)
                             return standard_name
                    else:
                        return standard_name # Already has standard name
            return None # Column not found

        # Define standard names and possible variations
        cols_map = {
            'Title': ['title', 'article title'],
            'Authors': ['authors', 'author', 'author(s)'],
            'DOI': ['doi', 'digital object identifier'],
            'DOI link': ['doi link', 'doi_link', 'link', 'url', 'source url', 'article url'], # URL is often used for DOI link
            'Year': ['year', 'publication year', 'pub year'],
            'Venue': ['venue', 'journal', 'journal name', 'conference', 'conference name', 'publication name']
            # Add 'URL' as a separate potential source if 'DOI link' variations don't capture it
            # We handle this in extract_doi_url implicitly by checking 'url' column if others fail
        }

        required_for_crossref = ['Title']
        required_for_direct_scrape = ['DOI', 'DOI link'] # Need at least one of these or a generic 'URL'

        found_columns = {}
        for std_name, variations in cols_map.items():
            found = find_and_rename_col(input_df, std_name, variations)
            if found: found_columns[std_name] = found

        # Check if we have enough info to proceed
        has_url_source = any(c in input_df.columns for c in required_for_direct_scrape) or 'URL' in input_df.columns # Check generic URL column too
        can_use_crossref_fallback = use_crossref and HABANERO_AVAILABLE and all(c in input_df.columns for c in required_for_crossref)

        if not has_url_source and not can_use_crossref_fallback:
             missing_direct = [c for c in required_for_direct_scrape if c not in input_df.columns]
             missing_cr = [c for c in required_for_crossref if c not in input_df.columns]
             err_msg = "Error: Cannot proceed. Need at least one URL source column ('DOI', 'DOI link', 'URL')"
             if use_crossref: err_msg += f" OR Crossref fallback enabled AND required columns ({', '.join(required_for_crossref)})."
             else: err_msg += "."
             if not has_url_source: err_msg += f" Missing URL source columns: {', '.join(missing_direct)}."
             if use_crossref and not all(c in input_df.columns for c in required_for_crossref): err_msg += f" Missing columns for Crossref: {', '.join(missing_cr)}."

             error_placeholder.error(err_msg)
             st.stop()

        # Keep track of original columns for final output ordering
        original_columns = input_df_original.columns.tolist() # Use original before renaming
        processed_columns = input_df.columns.tolist() # Columns after renaming

        # --- Start Button ---
        if st.button(f"Start Processing ({len(input_df)} rows)", key='start_button'):
            llm_ready = False
            if use_llm_fallback: # Load LLM only if enabled AND needed libraries are present
                if qa_pipeline is None and 'llm_load_failed' not in st.session_state:
                    with st.spinner(f"Loading QA model ({LLM_MODEL_NAME})... Please wait."):
                        qa_pipeline = load_qa_pipeline() # Function handles internal errors
                        if qa_pipeline is None:
                            st.session_state['llm_load_failed'] = True
                            error_placeholder.error("LLM model loading failed. LLM fallback will be disabled for this run.")
                        else:
                            info_placeholder.success("QA model loaded successfully.")
                            llm_ready = True
                elif qa_pipeline is not None:
                    llm_ready = True # Already loaded
                    info_placeholder.success("QA model already loaded.")
                else: # Load previously failed
                    error_placeholder.error("LLM model loading previously failed. LLM fallback disabled.")


            # --- Initialization for Processing ---
            results = []
            total_rows = len(input_df)
            rows_to_process = total_rows if MAX_ROWS_TO_PROCESS is None else min(total_rows, MAX_ROWS_TO_PROCESS)
            if MAX_ROWS_TO_PROCESS is not None and MAX_ROWS_TO_PROCESS < total_rows:
                 info_placeholder.info(f"Processing the first {rows_to_process} of {total_rows} entries based on MAX_ROWS_TO_PROCESS setting.")
            else:
                 info_placeholder.info(f"Processing {rows_to_process} entries...")

            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()
            processed_count = 0

            # --- Main Processing Loop ---
            for index, row in input_df.head(rows_to_process).iterrows():
                processed_count += 1
                progress = processed_count / rows_to_process if rows_to_process > 0 else 1.0
                try:
                     progress_bar.progress(progress)
                except Exception: pass # Ignore potential errors if element removed

                # Get title for status updates, handle missing title
                title_str = str(row.get('Title', f'Row {index + 1}')).strip()
                if not title_str: title_str = f'Row {index + 1} (No Title)'
                status_text.text(f"Processing {processed_count}/{rows_to_process}: {title_str[:65]}...")

                # Initialize variables for each row
                url, doi_source_status = None, "From Input"
                final_abstract, scraped_year, scraped_month, scrape_status = None, None, None, "Pending"
                result_row = row.to_dict() # Start with original row data
                processed_url_for_row = None # Store the URL actually used
                was_truncated = False # Reset truncation flag for each row

                # --- Step 1: Get URL (Input or Crossref) ---
                url, url_type = extract_doi_url(row) # Use standardized column names
                if url:
                    doi_source_status = url_type
                    processed_url_for_row = url
                elif can_use_crossref_fallback: # Only try Crossref if enabled and no URL/DOI found in input
                    query_title = row.get('Title')
                    first_author = None
                    authors_str = row.get('Authors')
                    # Extract first author more carefully
                    if authors_str and isinstance(authors_str, str):
                        # Split by common delimiters, handle 'et al.'
                        authors_list = re.split(r'\s*;\s*|\s+and\s+|\s*,\s*(?![^()]*\))', authors_str.replace(' et al.', '')) # Avoid splitting within affiliations
                        first_author = authors_list[0].strip() if authors_list else None

                    if query_title:
                        status_text.text(f"... {title_str[:60]}... (Searching Crossref)")
                        time.sleep(0.1) # Small delay for UI update
                        doi, cr_status = find_doi_via_crossref(query_title, first_author)
                        if doi:
                             # Validate DOI format before constructing URL
                             if re.match(r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$', doi, re.IGNORECASE):
                                 url = f"https://doi.org/{doi}"
                                 processed_url_for_row = url
                                 doi_source_status = f"Crossref ({cr_status})"
                                 result_row['DOI'] = doi # Update DOI column if found via Crossref
                                 scrape_status = "URL Found via Crossref"
                             else:
                                 logging.warning(f"Crossref returned invalid DOI format: {doi}")
                                 doi_source_status = f"Crossref (Invalid DOI Format: {cr_status})"
                                 scrape_status = "Crossref DOI Invalid"
                        else:
                             doi_source_status = f"Crossref ({cr_status})"
                             scrape_status = "URL Not Found (Crossref Failed)"
                        status_text.text(f"Processing {processed_count}/{rows_to_process}: {title_str[:65]}...") # Reset status text
                    else:
                         scrape_status = "URL Not Found (Missing Title for Crossref)"
                         doi_source_status = "Input (No URL/DOI) / Crossref (No Title)"
                else: # No URL in input and Crossref not usable/enabled
                     scrape_status = "URL Not Found (No Input Source & Crossref Disabled/Unavailable)"
                     doi_source_status = "Input (None) / Crossref (N/A)"


                # --- Step 2: Scrape Page Content (if URL exists) ---
                if url:
                    scrape_status = "URL Ready, Fetching..." # Initial status before request
                    possible_paywall_note = ""
                    response = None # Initialize response

                    try:
                        parsed_url = urlparse(url)
                        domain = parsed_url.netloc
                        # Ensure domain extraction is safe
                        if not domain and '://' in url:
                             domain = url.split('://')[1].split('/')[0]

                        current_delay = DOMAIN_SPECIFIC_DELAYS.get(domain, DEFAULT_REQUEST_DELAY)
                        logging.info(f"Attempting fetch for: {url} (Domain: {domain}, Delay: {current_delay}s)")
                        time.sleep(current_delay) # Apply delay before request

                        response, fetch_status_msg = safe_request_get(url, HEADERS, timeout=REQUEST_TIMEOUT)

                        if response: # Request succeeded, even if paywall detected
                            scrape_status = "HTML Received, Parsing..."
                            if fetch_status_msg and "Note:" in fetch_status_msg:
                                possible_paywall_note = f" ({fetch_status_msg.replace('Note: ', '')})" # Store paywall note separately
                                logging.warning(f"Possible paywall detected for {response.url}")

                            # --- Step 3: Parse HTML and Extract Data ---
                            try:
                                soup = None
                                parser_type = 'lxml' if LXML_AVAILABLE else 'html.parser'
                                soup = BeautifulSoup(response.text, parser_type)
                                response_url = response.url # Use the final URL after redirects

                                # Call the enhanced orchestrator function
                                final_abstract, scraped_year, scraped_month, scrape_status, was_truncated = scrape_abstract_and_date(soup, response_url)

                                # --- Step 4: LLM Fallback Logic (Conditional) ---
                                if llm_ready:
                                    llm_trigger_reason = None
                                    if final_abstract is None:
                                        llm_trigger_reason = "No Abstract Found"
                                    elif was_truncated: # Abstract found but suspected truncated
                                        # Check if the status already reflects a successful retry that overcame truncation
                                        if "after Retry" not in scrape_status:
                                            llm_trigger_reason = "Truncated Abstract"

                                    if llm_trigger_reason:
                                        status_text.text(f"... {title_str[:60]}... (LLM Fallback: {llm_trigger_reason})")
                                        logging.info(f"Triggering LLM fallback for '{title_str[:50]}...'. Reason: {llm_trigger_reason}")
                                        time.sleep(0.1) # UI update pause

                                        # Prepare context using filtered HTML
                                        llm_context = filter_html_for_llm(soup)
                                        if llm_context:
                                            llm_abstract, llm_status = get_llm_abstract(llm_context, "Filtered HTML")
                                            if llm_abstract:
                                                # Compare LLM result with potentially truncated rule result if it existed and was the trigger
                                                use_llm_result = True
                                                if llm_trigger_reason == "Truncated Abstract" and final_abstract:
                                                     # Only replace if LLM abstract is significantly longer or rule-based was very short
                                                     if len(llm_abstract) < len(final_abstract) * 1.1 and len(final_abstract.split()) > 30:
                                                          use_llm_result = False
                                                          logging.info("LLM found abstract, but keeping rule-based one (not significantly better).")
                                                          # Append LLM status without replacing abstract
                                                          scrape_status += f" / {llm_status.replace('Success', 'LLM Found Similar/Shorter')}"
                                                     else:
                                                          logging.info("Using LLM abstract (replaced potentially truncated rule-based abstract).")

                                                if use_llm_result:
                                                    final_abstract = llm_abstract
                                                    # Update status to reflect LLM success, keeping date info if found
                                                    date_part = ""
                                                    if "Date (" in scrape_status:
                                                         date_part = scrape_status[scrape_status.find(" / Date ("):]
                                                    elif "Date Not Found" in scrape_status:
                                                         date_part = " / Date Not Found"
                                                    scrape_status = f"{llm_status}{date_part}"
                                            else: # LLM failed
                                                logging.warning(f"LLM fallback failed to find abstract. Status: {llm_status}")
                                                scrape_status += f" / {llm_status}" # Append LLM failure details
                                        else: # Could not generate LLM context
                                             logging.error("Could not generate filtered HTML context for LLM.")
                                             scrape_status += " / LLM Skipped (Context Error)"
                                        status_text.text(f"Processing {processed_count}/{rows_to_process}: {title_str[:65]}...") # Reset status text
                                else: # LLM not ready or not enabled
                                    if (final_abstract is None or was_truncated) and use_llm_fallback:
                                         # Log if LLM *would* have run but wasn't ready
                                         logging.warning(f"LLM fallback condition met for '{title_str[:50]}...' but LLM is not available/loaded.")
                                         if final_abstract is None: scrape_status += " / LLM Skipped (Unavailable)"
                                         elif "after Retry" not in scrape_status : scrape_status += " / LLM Skipped (Unavailable)"


                            except Exception as e_parse:
                                scrape_status = f"Parsing/Processing Error: {str(e_parse)[:100]}..."
                                logging.exception(f"Error parsing/processing URL {url}")

                        else: # Safe request failed (returned None for response)
                            scrape_status = fetch_status_msg # Use the error message from safe_request_get

                    except Exception as e_req_outer:
                         # Catch any unexpected errors during the request/processing block
                         scrape_status = f"Outer Request/Process Error: {str(e_req_outer)[:100]}"
                         logging.exception(f"Unexpected outer error processing URL {url}")

                    # Append paywall note if present
                    if possible_paywall_note:
                        scrape_status += possible_paywall_note

                # --- Step 5: Populate Final Result Row ---
                result_row['Scraped_Abstract'] = final_abstract
                # Handle original year (ensure it's numeric if present)
                original_year_value = pd.to_numeric(row.get('Year', None), errors='coerce')
                # Use scraped year if found, otherwise fallback to original numeric year
                scraped_year_num = pd.to_numeric(scraped_year, errors='coerce')
                result_row['Scraped_Year'] = int(scraped_year_num) if pd.notna(scraped_year_num) else (int(original_year_value) if pd.notna(original_year_value) else None)
                # Handle month
                scraped_month_num = pd.to_numeric(scraped_month, errors='coerce')
                result_row['Scraped_Month'] = int(scraped_month_num) if pd.notna(scraped_month_num) else None
                # Combine overall status with the source of the URL/DOI
                result_row['Scrape_Status'] = f"{scrape_status} (URL Source: {doi_source_status})"
                result_row['Processed_URL'] = processed_url_for_row # Add the URL that was actually scraped

                # Ensure all original columns are present in the final dict, even if empty
                for col in original_columns:
                     if col not in result_row and col in row: # Check if it existed in the original row
                         result_row[col] = row.get(col)

                results.append(result_row)
                # Minimal delay even if domain specific delay wasn't hit (e.g., Crossref fail)
                # time.sleep(0.1) # Reduced this as delay happens *before* request now

            # --- End Processing Loop ---

            end_time = time.time()
            duration = end_time - start_time
            status_text.text(f"Processing complete for {processed_count} entries in {duration:.2f} seconds.")
            progress_bar.empty() # Remove progress bar

            # --- Display Results ---
            if results:
                try:
                    output_df = pd.DataFrame(results)

                    # Define core output columns and preserve others
                    core_cols = ['Title', 'Authors', 'Year', 'Scraped_Year', 'Scraped_Month', 'Venue',
                                 'Scraped_Abstract', 'Scrape_Status', 'Processed_URL', 'DOI', 'DOI link', 'URL']
                    # Get unique columns present in the output DataFrame
                    output_cols_present = output_df.columns.tolist()
                    # Start with core columns that actually exist in the output
                    final_cols_order = [c for c in core_cols if c in output_cols_present]
                    # Add any other original columns that were carried over and aren't already in the core list
                    other_original_cols = [c for c in processed_columns if c in output_cols_present and c not in final_cols_order] # Use processed_columns names
                    final_cols_order.extend(other_original_cols)
                    # Add any *new* columns created during processing if not already included (should be covered)
                    new_cols = [c for c in output_cols_present if c not in final_cols_order]
                    final_cols_order.extend(new_cols)

                    # Reindex DataFrame
                    output_df = output_df[final_cols_order]

                    st.write("Processing Results Preview (first 10 rows):")
                    st.dataframe(output_df.head(10))
                    st.success("Processing finished!")

                    # --- Summary Statistics ---
                    abstract_found_count = output_df['Scraped_Abstract'].notna().sum()
                    date_found_count = output_df['Scraped_Year'].notna().sum()
                    llm_success_count = output_df['Scrape_Status'].str.contains(r'LLM \(.*\) Success', case=False, na=False, regex=True).sum()
                    crossref_success_count = output_df['Scrape_Status'].str.contains(r'Crossref \(DOI Found', case=False, na=False, regex=True).sum()
                    truncation_suspected_count = output_df['Scrape_Status'].str.contains('Truncation Suspected', case=False, na=False).sum()
                    pdf_skipped_count = output_df['Scrape_Status'].str.contains('Skipped \(Content is PDF', case=False, na=False).sum()
                    failed_count = output_df['Scrape_Status'].str.contains(r'(Failed|Error|Not Found)', case=False, na=False, regex=True).sum() - pdf_skipped_count # Exclude PDF skips from general fails


                    st.info(f"""**Summary:**
- Processed Entries: **{processed_count}**
- Abstracts Found: **{abstract_found_count}** ({abstract_found_count/processed_count:.1%})
- Dates Found (Year): **{date_found_count}** ({date_found_count/processed_count:.1%})
- Abstracts found via LLM: **{llm_success_count}**
- DOIs found via Crossref: **{crossref_success_count}**
- Truncation Suspected (kept/retried): **{truncation_suspected_count}**
- Skipped (PDF): **{pdf_skipped_count}**
- Other Failures/Not Found: **{failed_count}**""")

                    # --- Download Buttons ---
                    col1, col2 = st.columns(2)
                    with col1:
                        try:
                             csv_data = output_df.to_csv(index=False).encode('utf-8')
                             st.download_button(label="Download Results as CSV (UTF-8)",
                                              data=csv_data,
                                              file_name='bibliography_processed_utf8.csv',
                                              mime='text/csv',
                                              key='csv_download_utf8')
                        except Exception as e_csv:
                             error_placeholder.error(f"Could not generate CSV file: {e_csv}")
                             logging.exception("CSV generation error")
                    with col2:
                        try:
                             excel_data = convert_df_to_excel(output_df)
                             st.download_button(label="Download Results as Excel",
                                              data=excel_data,
                                              file_name='bibliography_processed.xlsx',
                                              mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                              key='excel_download')
                        except Exception as e_excel:
                             error_placeholder.error(f"Could not generate Excel file: {e_excel}")
                             logging.exception("Excel generation error")
                except Exception as e_results:
                     error_placeholder.error(f"An error occurred while preparing results: {e_results}")
                     logging.exception("Error preparing results for display/download.")

            else: # No results generated (e.g., all rows failed very early)
                warning_placeholder.warning("No results were generated. Check input file and logs.")

    except pd.errors.EmptyDataError: # Catch error if file uploaded but pandas reads it as empty
        error_placeholder.error("Error: Uploaded CSV file is empty or could not be parsed correctly.")
    except Exception as e:
        error_placeholder.error(f"An critical unexpected error occurred during processing: {e}")
        logging.exception("Critical error in main processing block.")
        st.exception(e) # Show detailed traceback in Streamlit during development/debugging

# --- Footer ---
st.markdown("---")
st.markdown("Hybrid Scraper v1.8 | Enhanced Rules, JSON-LD, DateUtil, Crossref, Truncation Retry, Conditional LLM | Developed with extreme diligence and precision.")
