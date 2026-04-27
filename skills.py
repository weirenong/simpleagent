

from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
import shutil
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RAG_MODEL_PATH = BASE_DIR / "models" / "rag-all-minilm-l6-v2" / "model"
_EMBEDDING_MODEL: Any | None = None
DEBUG_SEARCH_FETCH = False
USE_PLAYWRIGHT_FETCH = True
PLAYWRIGHT_TIMEOUT_MS = 12_000
SEARCH_DEFAULT_MAX_RESULTS = 8
SEARCH_FETCH_TOP_PAGES = 5
SEARCH_VARIANT_LIMIT = 3


class DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture_title = False
        self._capture_snippet = False
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        class_name = attr_map.get("class", "")

        if tag == "a" and "result__a" in class_name:
            self._current = {
                "title": "",
                "url": self._clean_url(attr_map.get("href", "")),
                "description": "",
            }
            self._capture_title = True
            self._title_parts = []
            return

        if self._current is not None and tag in {"a", "div"} and "result__snippet" in class_name:
            self._capture_snippet = True
            self._snippet_parts = []

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._title_parts.append(data)
        if self._capture_snippet:
            self._snippet_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title and self._current is not None:
            self._current["title"] = self._normalise_text(" ".join(self._title_parts))
            self._capture_title = False
            return

        if self._capture_snippet and tag in {"a", "div"} and self._current is not None:
            self._current["description"] = self._normalise_text(" ".join(self._snippet_parts))
            self._capture_snippet = False
            if self._current.get("title") and self._current.get("url"):
                self.results.append(self._current)
            self._current = None

    def _clean_url(self, url: str) -> str:
        decoded = html.unescape(url)
        if "uddg=" in decoded:
            parsed = urllib.parse.urlparse(decoded)
            params = urllib.parse.parse_qs(parsed.query)
            if "uddg" in params and params["uddg"]:
                return params["uddg"][0]
        return decoded

    def _normalise_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", html.unescape(value)).strip()


# --- PageTextParser for lightweight page excerpt extraction ---
class PageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self._skip_depth = 0
        self._capture_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg", "footer", "nav"}:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in {"h1", "h2", "h3", "p", "li"}:
            self._capture_tag = tag

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0 or self._capture_tag is None:
            return
        cleaned = re.sub(r"\s+", " ", html.unescape(data)).strip()
        if len(cleaned) >= 20:
            self.text_parts.append(cleaned)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "footer", "nav"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag == self._capture_tag:
            self._capture_tag = None

    def get_text(self) -> str:
        seen: set[str] = set()
        deduped: list[str] = []
        for part in self.text_parts:
            key = part.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(part)
        return "\n".join(deduped)


def get_all_skills() -> str:
    return "\n".join(
        [
            "0: no_skill - Do not call any skill. Use this when internet data is unnecessary.",
            "1: internet_search - Search the internet for current or factual information and return ranked snippets as grounding context.",
            "2: scrape_url - Extract relevant information from one or more user-provided URLs and return page excerpts as grounding context.",
            "3: memory_rag - Retrieve relevant past conversation messages for context when the user refers to previous discussion.",
            "4: attachment_vision - Analyse attached images or videos with the local VL model when the user attaches visual files or asks about an attachment.",
            "5: text_file_reader - Read attached plain-text, markdown, code, config, CSV, JSON, HTML, CSS, SQL, shell, YAML, and other text-like files from local file paths.",
            "6: pdf_reader - Read text from attached PDF files using local PDF text extraction.",
            "7: code_reader - Read attached code files and provide coding, debugging, refactoring, code review, implementation, and patch-style guidance.",
            "8: file_editor - Create or edit text/code files by producing an apply-patch style diff for attached files or files inside the current chat workspace."

        ]
    )

def get_valid_skill_ids() -> set[int]:
    return {0, 1, 2, 3, 4, 5, 6, 7, 8}


def execute_skill(skill_id: int, user_prompt: str, memory_items: list[dict[str, str]] | None = None) -> str:
    if skill_id == 1:
        return internet_search(user_prompt)
    if skill_id == 2:
        return scrape_url(user_prompt)
    if skill_id == 3:
        return memory_rag_search(user_prompt, memory_items or [])
    if skill_id == 4:
        # Attachment vision is executed inside simple_agent.py because it needs access to local attachment paths and the VL runtime.
        return ""
    if skill_id == 5:
        # Text file reading is executed inside simple_agent.py because it needs access to local attachment paths.
        return ""
    if skill_id == 6:
        # PDF reading is executed inside simple_agent.py because it needs access to local attachment paths.
        return ""
    if skill_id == 7:
        # Code reading is executed inside simple_agent.py because it needs access to local attachment paths and coding context.
        return ""
    if skill_id == 8:
        # File editing is executed inside simple_agent.py because it needs chat workspace paths, attachment paths, user approval, and local file access.
        return ""
    return ""


# --- MEMORY RAG SEARCH ---
def memory_rag_search(query: str, memory_items: list[dict[str, str]], max_items: int = 5) -> str:
    if not memory_items:
        return "No relevant memory found."

    model = _get_embedding_model()

    # Fallback to keyword method if embedding model not available
    if model is None:
        return _memory_rag_keyword_fallback(query, memory_items, max_items)

    try:
        documents = [
            f"{item.get('user_summary','')} {item.get('assistant_summary','')}"
            for item in memory_items
        ]

        query_embedding = model.encode([query], normalize_embeddings=True)[0]
        doc_embeddings = model.encode(documents, normalize_embeddings=True)

        scored = []
        for item, emb in zip(memory_items, doc_embeddings):
            score = float(sum(float(a) * float(b) for a, b in zip(query_embedding, emb)))
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [item for score, item in scored[:max_items] if score > 0.2]

        if not top:
            return "No strongly relevant past messages found."

    except Exception:
        return _memory_rag_keyword_fallback(query, memory_items, max_items)

    lines = [
        "Relevant past conversation context (semantic match):",
        "Use this ONLY if it directly helps answer the current prompt.",
    ]

    for i, item in enumerate(top, 1):
        lines.append(f"{i}. User: {item.get('user_summary','')}")
        lines.append(f"   Assistant: {item.get('assistant_summary','')}")

    return "\n".join(lines)


# --- Memory RAG keyword fallback ---
def _memory_rag_keyword_fallback(query: str, memory_items: list[dict[str, str]], max_items: int) -> str:
    query_terms = _tokenise(query)
    scored: list[tuple[float, dict[str, str]]] = []

    for item in memory_items:
        combined = f"{item.get('user_summary','')} {item.get('assistant_summary','')}"
        terms = _tokenise(combined)
        overlap = len(query_terms & terms)
        score = overlap + (len(terms) * 0.01)
        if overlap > 0:
            scored.append((score, item))

    if not scored:
        return "No strongly relevant past messages found."

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [item for _, item in scored[:max_items]]

    lines = [
        "Relevant past conversation context (keyword fallback):",
        "Use this ONLY if it directly helps answer the current prompt.",
    ]

    for i, item in enumerate(top, 1):
        lines.append(f"{i}. User: {item.get('user_summary','')}")
        lines.append(f"   Assistant: {item.get('assistant_summary','')}")

    return "\n".join(lines)

def _extract_urls(text: str) -> list[str]:
    raw_urls = re.findall(r"https?://[^\s)\]>\"']+", text)
    cleaned_urls: list[str] = []
    seen: set[str] = set()

    for raw_url in raw_urls:
        url = raw_url.rstrip(".,;:!?]")
        if url not in seen:
            seen.add(url)
            cleaned_urls.append(url)

    return cleaned_urls


def scrape_url(user_prompt: str, max_urls: int = 3) -> str:
    urls = _extract_urls(user_prompt)[:max_urls]
    if not urls:
        return "URL scraping skipped because no valid URL was found."

    lines = [
        "Direct URL scrape results:",
        "Extract and summarise ONLY the most relevant factual information from these pages.",
        "Do not repeat generic page descriptions.",
        "If specific answers (names, values, lists, facts) exist, prioritise them.",
        "Always include source URLs when referencing information.",
    ]

    for index, url in enumerate(urls, start=1):
        page_text = _fetch_page_text(url)

        if DEBUG_SEARCH_FETCH:
            print("\n===== DIRECT URL SCRAPE START =====")
            print(f"URL: {url}")
            print(f"Fetched characters: {len(page_text)}")
            if page_text:
                print(page_text[:4000])
            else:
                print("No page text extracted. The page may be blocking the request, requiring JavaScript, returning non-HTML, or failing to load.")
            print("===== DIRECT URL SCRAPE END =====\n")

        if not page_text:
            lines.append(
                f"{index}. URL: {url}\n"
                "   Page excerpt: No page text could be extracted."
            )
            continue

        excerpt = _best_page_excerpt(user_prompt, page_text, max_chars=1400)
        lines.append(
            f"{index}. URL: {url}\n"
            f"   Page excerpt: {excerpt}"
        )

    return "\n".join(lines)




def internet_search(query: str, max_results: int = SEARCH_DEFAULT_MAX_RESULTS) -> str:
    original_query = query.strip()
    if not original_query:
        return "Internet search skipped because the query was empty."

    search_queries = _build_search_queries(original_query)[:SEARCH_VARIANT_LIMIT]
    raw_results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for search_query in search_queries:
        variant_results = _search_duckduckgo(search_query, max_results=max(12, max_results * 3))
        for result in variant_results:
            url = result.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            enriched = dict(result)
            enriched["search_query"] = search_query
            raw_results.append(enriched)

    if not raw_results:
        return f"Internet search found no results for: {original_query}"

    ranked_results = _rank_results(original_query, raw_results)
    ranked_results = _apply_source_quality_scores(original_query, ranked_results)
    ranked_results = ranked_results[:max_results]
    ranked_results = _enrich_results_with_page_excerpts(
        original_query,
        ranked_results,
        max_pages=SEARCH_FETCH_TOP_PAGES,
    )

    print("\n===== INTERNET SEARCH RESULTS START =====")
    print(f"Original query: {original_query}")
    print("Search variants:")
    for search_query in search_queries:
        print(f"- {search_query}")
    for index, result in enumerate(ranked_results, start=1):
        print(f"{index}. {result.get('title', 'Untitled')}")
        print(f"   URL: {result.get('url', '')}")
        print(f"   Domain: {_domain_from_url(result.get('url', ''))}")
        print(f"   Description: {result.get('description', 'No description available.')}")
        print(f"   Relevance score: {float(result.get('score', 0.0)):.3f}")
        print(f"   Source quality: {float(result.get('source_quality', 0.0)):.3f}")
        print(f"   Final score: {float(result.get('final_score', 0.0)):.3f}")
        if result.get("page_excerpt"):
            print(f"   Page excerpt: {result.get('page_excerpt')}")
    print("===== INTERNET SEARCH RESULTS END =====\n")

    lines = [
        f"Internet search results for: {original_query}",
        "Search strategy: multiple query variants were used, results were deduplicated, source quality was scored, and the top pages were fetched for richer excerpts.",
        "Extract and summarise ONLY the most relevant factual information from the following results.",
        "Prefer reputable and primary sources over low-quality blogs, SEO pages, or suspicious domains.",
        "Do not repeat generic descriptions or page titles.",
        "If specific answers (names, values, lists, facts, dates) exist, prioritise them.",
        "Always include source URLs when referencing information.",
    ]
    for index, result in enumerate(ranked_results, start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        description = result.get("description", "No description available.")
        score = float(result.get("score", 0.0))
        source_quality = float(result.get("source_quality", 0.0))
        final_score = float(result.get("final_score", 0.0))
        page_excerpt = result.get("page_excerpt", "")
        result_text = (
            f"{index}. {title}\n"
            f"   URL: {url}\n"
            f"   Domain: {_domain_from_url(url)}\n"
            f"   Description: {description}\n"
            f"   Relevance score: {score:.3f}\n"
            f"   Source quality score: {source_quality:.3f}\n"
            f"   Final score: {final_score:.3f}"
        )
        if page_excerpt:
            result_text += f"\n   Page excerpt: {page_excerpt}"
        lines.append(result_text)
    return "\n".join(lines)
# --- Search helpers for internet_search ---

def _build_search_queries(query: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", query).strip()
    lowered = cleaned.lower()
    queries: list[str] = [cleaned]

    if _query_looks_scholarly(lowered):
        queries.extend([
            f"{cleaned} site:scholar.google.com OR site:arxiv.org OR site:semanticscholar.org",
            f"{cleaned} research paper study review",
        ])
    elif _query_looks_encyclopedic(lowered):
        queries.extend([
            f"{cleaned} Wikipedia overview",
            f"{cleaned} official source",
        ])
    elif _query_looks_news_or_current(lowered):
        queries.extend([
            f"{cleaned} Reuters AP BBC latest",
            f"{cleaned} site:reuters.com OR site:apnews.com OR site:bbc.com",
        ])
    elif _query_looks_technical(lowered):
        queries.extend([
            f"{cleaned} official documentation",
            f"{cleaned} GitHub documentation",
        ])
    elif _query_looks_financial(lowered):
        queries.extend([
            f"{cleaned} official investor relations SEC latest",
            f"{cleaned} Reuters MarketWatch Yahoo Finance",
        ])
    else:
        queries.extend([
            f"{cleaned} official source",
            f"{cleaned} overview reliable source",
        ])

    deduped: list[str] = []
    seen: set[str] = set()
    for item in queries:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _query_looks_news_or_current(lowered_query: str) -> bool:
    markers = {
        "latest", "current", "today", "recent", "news", "breaking", "war", "conflict",
        "price", "stock", "crypto", "earnings", "announcement", "update", "updates",
    }
    return any(marker in lowered_query for marker in markers)


def _query_looks_scholarly(lowered_query: str) -> bool:
    markers = {
        "paper", "papers", "study", "studies", "research", "journal", "scholar",
        "literature", "citation", "doi", "academic", "systematic review", "meta analysis",
    }
    return any(marker in lowered_query for marker in markers)


def _query_looks_technical(lowered_query: str) -> bool:
    markers = {
        "python", "javascript", "typescript", "api", "library", "framework", "documentation",
        "docs", "install", "error", "bug", "github", "package", "sdk", "cli",
    }
    return any(marker in lowered_query for marker in markers)


def _query_looks_financial(lowered_query: str) -> bool:
    markers = {
        "stock", "ticker", "shares", "etf", "earnings", "revenue", "market cap", "sec filing",
        "10-k", "10-q", "investor relations", "price target", "dividend", "crypto", "coin",
    }
    return any(marker in lowered_query for marker in markers)


def _query_looks_encyclopedic(lowered_query: str) -> bool:
    markers = {"what is", "who is", "history of", "overview", "definition", "meaning of", "explain"}
    return any(marker in lowered_query for marker in markers)


def _domain_from_url(url: str) -> str:
    try:
        domain = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _apply_source_quality_scores(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lowered_query = query.lower()
    scored_results: list[dict[str, Any]] = []
    for result in results:
        enriched = dict(result)
        domain = _domain_from_url(enriched.get("url", ""))
        source_quality = _source_quality_score(domain, lowered_query)
        relevance_score = float(enriched.get("score", 0.0))
        final_score = relevance_score + source_quality
        enriched["source_quality"] = source_quality
        enriched["final_score"] = final_score
        scored_results.append(enriched)
    return sorted(scored_results, key=lambda item: item.get("final_score", 0.0), reverse=True)


def _source_quality_score(domain: str, lowered_query: str) -> float:
    if not domain:
        return -0.25

    trusted_general = {
        "wikipedia.org", "britannica.com", "investopedia.com",
    }
    trusted_news = {
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "aljazeera.com", "theguardian.com",
        "nytimes.com", "washingtonpost.com", "ft.com", "bloomberg.com", "cnbc.com", "npr.org",
    }
    trusted_research = {
        "scholar.google.com", "arxiv.org", "semanticscholar.org", "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov", "nature.com", "science.org", "springer.com", "sciencedirect.com",
        "ieee.org", "acm.org", "jstor.org", "ssrn.com",
    }
    trusted_technical = {
        "docs.python.org", "developer.mozilla.org", "github.com", "stackoverflow.com", "pypi.org",
        "readthedocs.io", "huggingface.co", "pytorch.org", "numpy.org", "pandas.pydata.org",
        "openai.com", "microsoft.com", "apple.com", "google.com", "cloudflare.com",
    }
    trusted_finance = {
        "sec.gov", "investor.gov", "nasdaq.com", "nyse.com", "finance.yahoo.com", "marketwatch.com",
        "morningstar.com", "companiesmarketcap.com", "macrotrends.net", "tradingview.com",
    }

    quality = 0.0
    if any(domain == item or domain.endswith("." + item) for item in trusted_general):
        quality += 0.25
    if any(domain == item or domain.endswith("." + item) for item in trusted_news):
        quality += 0.55 if _query_looks_news_or_current(lowered_query) else 0.25
    if any(domain == item or domain.endswith("." + item) for item in trusted_research):
        quality += 0.55 if _query_looks_scholarly(lowered_query) else 0.25
    if any(domain == item or domain.endswith("." + item) for item in trusted_technical):
        quality += 0.45 if _query_looks_technical(lowered_query) else 0.20
    if any(domain == item or domain.endswith("." + item) for item in trusted_finance):
        quality += 0.45 if _query_looks_financial(lowered_query) else 0.20

    suspicious_markers = {
        "blogspot.", "wordpress.", "medium.com", "quora.com", "reddit.com", "pinterest.",
        "fandom.com", "wiki.gg", "answers.com", "slideshare.", "scribd.",
    }
    if any(marker in domain for marker in suspicious_markers):
        quality -= 0.15

    spam_markers = {"coupon", "casino", "betting", "free-download", "apk", "lyrics"}
    if any(marker in domain for marker in spam_markers):
        quality -= 0.35

    return quality


def _search_duckduckgo(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    encoded_query = urllib.parse.urlencode({"q": query})
    url = f"https://duckduckgo.com/html/?{encoded_query}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 SimpleAgent/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            html_text = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return [{"title": "Search failed", "url": "", "description": str(exc), "score": 0.0}]

    parser = DuckDuckGoHTMLParser()
    parser.feed(html_text)

    deduped: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for result in parser.results:
        url_value = result.get("url", "")
        if not url_value or url_value in seen_urls:
            continue
        seen_urls.add(url_value)
        deduped.append(result)
        if len(deduped) >= max_results:
            break
    return deduped


def _rank_results(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = _rank_with_sentence_transformer(query, results)
    if ranked is not None:
        return ranked
    return _rank_with_keyword_overlap(query, results)


# --- Enrich results with page excerpts ---
def _enrich_results_with_page_excerpts(
    query: str,
    results: list[dict[str, Any]],
    max_pages: int = 3,
) -> list[dict[str, Any]]:
    enriched_results: list[dict[str, Any]] = []
    pages_checked = 0

    for result in results:
        enriched = dict(result)
        url = enriched.get("url", "")
        if pages_checked < max_pages and url.startswith(("http://", "https://")):
            page_text = _fetch_page_text(url)

            if DEBUG_SEARCH_FETCH:
                print("\n===== FETCHED PAGE CONTENT START =====")
                print(f"URL: {url}")
                print(f"Fetched characters: {len(page_text)}")
                if page_text:
                    print(page_text[:4000])
                else:
                    print(
                        "No page text extracted. The page may be blocking the request, requiring JavaScript, returning non-HTML, or failing to load.")
                print("===== FETCHED PAGE CONTENT END =====\n")

            if page_text:
                excerpt = _best_page_excerpt(query, page_text)
                if excerpt:
                    enriched["page_excerpt"] = excerpt
                pages_checked += 1
        enriched_results.append(enriched)

    return enriched_results


def _fetch_page_text(url: str) -> str:
    page_text = ""

    if USE_PLAYWRIGHT_FETCH:
        page_text = _fetch_page_text_with_playwright(url)
        if page_text:
            return page_text

    return _fetch_page_text_with_urllib(url)


def _fetch_page_text_with_urllib(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36 SimpleAgent/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            content_type = response.headers.get("Content-Type", "")
            status_code = getattr(response, "status", "unknown")
            html_bytes = response.read(400_000)
            if DEBUG_SEARCH_FETCH:
                print("\n===== PAGE FETCH DEBUG urllib =====")
                print(f"URL: {url}")
                print(f"Status: {status_code}")
                print(f"Content-Type: {content_type}")
                print(f"Raw bytes read: {len(html_bytes)}")
                print("===== PAGE FETCH DEBUG urllib END =====")
            if "text/html" not in content_type.lower():
                return ""
            html_text = html_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        if DEBUG_SEARCH_FETCH:
            print("\n===== PAGE FETCH FAILED urllib =====")
            print(f"URL: {url}")
            print(f"Error: {exc}")
            print("===== PAGE FETCH FAILED urllib END =====\n")
        return ""

    return _extract_text_from_html(html_text)


def _fetch_page_text_with_playwright(url: str) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        if DEBUG_SEARCH_FETCH:
            print("\n===== PAGE FETCH SKIPPED Playwright =====")
            print(f"URL: {url}")
            print(f"Reason: playwright is not installed or unavailable: {exc}")
            print("Install with: pip install playwright && python -m playwright install chromium")
            print("===== PAGE FETCH SKIPPED Playwright END =====\n")
        return ""

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36 SimpleAgent/1.0"
                ),
                locale="en-US",
            )
            page.goto(url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_TIMEOUT_MS)
            try:
                page.wait_for_load_state("networkidle", timeout=PLAYWRIGHT_TIMEOUT_MS)
            except Exception:
                pass
            page_text = page.locator("body").inner_text(timeout=PLAYWRIGHT_TIMEOUT_MS)
            browser.close()

            page_text = _normalise_page_text(page_text)
            if DEBUG_SEARCH_FETCH:
                print("\n===== PAGE FETCH DEBUG Playwright =====")
                print(f"URL: {url}")
                print(f"Extracted characters: {len(page_text)}")
                print("===== PAGE FETCH DEBUG Playwright END =====")
            return page_text
    except Exception as exc:
        if DEBUG_SEARCH_FETCH:
            print("\n===== PAGE FETCH FAILED Playwright =====")
            print(f"URL: {url}")
            print(f"Error: {exc}")
            print("===== PAGE FETCH FAILED Playwright END =====\n")
        return ""


def _extract_text_from_html(html_text: str) -> str:
    parser = PageTextParser()
    try:
        parser.feed(html_text)
    except Exception:
        return ""
    return parser.get_text()


def _normalise_page_text(text: str) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", html.unescape(raw_line)).strip()
        if len(cleaned) < 20:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(cleaned)
    return "\n".join(lines)


def _best_page_excerpt(query: str, page_text: str, max_chars: int = 900) -> str:
    query_terms = _tokenise(query)
    chunks = [chunk.strip() for chunk in page_text.splitlines() if chunk.strip()]
    if not chunks:
        return ""

    scored_chunks: list[tuple[float, str]] = []
    for index, chunk in enumerate(chunks):
        chunk_terms = _tokenise(chunk)
        overlap = len(query_terms & chunk_terms)
        score = overlap + (0.05 / (index + 1))
        if overlap > 0 or index < 8:
            scored_chunks.append((score, chunk))

    if not scored_chunks:
        scored_chunks = [(0.0, chunk) for chunk in chunks[:8]]

    key_items = _extract_key_items(query, chunks)
    best_chunks = [chunk for _, chunk in sorted(scored_chunks, key=lambda item: item[0], reverse=True)[:8]]
    excerpt = " | ".join(best_chunks)
    if key_items:
        excerpt = "Key extracted items: " + ", ".join(key_items[:10]) + " | " + excerpt
    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    return excerpt[:max_chars]


def _extract_key_items(query: str, chunks: list[str], max_items: int = 12) -> list[str]:
    query_terms = _tokenise(query)
    candidates: list[str] = []

    for chunk in chunks[:80]:
        candidates.extend(_extract_list_like_items(chunk))
        candidates.extend(_extract_named_phrases(chunk))

    scored_candidates: list[tuple[float, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = _clean_candidate_item(candidate)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)

        candidate_terms = _tokenise(cleaned)
        overlap = len(query_terms & candidate_terms)
        score = overlap + min(len(cleaned) / 80, 1.0)
        scored_candidates.append((score, cleaned))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored_candidates[:max_items]]


def _extract_list_like_items(text: str) -> list[str]:
    items: list[str] = []

    numbered_match = re.match(r"^\s*(?:\d+|[ivxlcdm]+)[\).:-]\s+(.+)$", text, flags=re.IGNORECASE)
    if numbered_match:
        items.append(numbered_match.group(1))

    bullet_match = re.match(r"^\s*[-*•]\s+(.+)$", text)
    if bullet_match:
        items.append(bullet_match.group(1))

    heading_like = re.match(r"^\s*([A-Z0-9][^.!?]{2,90})\s*$", text)
    if heading_like and len(text.split()) <= 12:
        items.append(heading_like.group(1))

    return items


def _extract_named_phrases(text: str) -> list[str]:
    patterns = [
        r"\b([A-Z][a-zA-Z0-9&:'’\-]+(?:\s+[A-Z0-9][a-zA-Z0-9&:'’\-]+){0,5})\b",
        r"\b([A-Z]{2,}(?:\s+[A-Z]{2,}){0,4})\b",
    ]

    phrases: list[str] = []
    for pattern in patterns:
        phrases.extend(re.findall(pattern, text))
    return phrases


def _clean_candidate_item(value: str) -> str:
    cleaned = html.unescape(value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^[\W_]+|[\W_]+$", "", cleaned)

    if not cleaned:
        return ""
    if len(cleaned) < 3 or len(cleaned) > 120:
        return ""
    if cleaned.lower() in _generic_candidate_stopwords():
        return ""
    if re.fullmatch(r"\d+", cleaned):
        return ""
    if len(cleaned.split()) > 14:
        return ""

    return cleaned


def _generic_candidate_stopwords() -> set[str]:
    return {
        "home",
        "news",
        "reviews",
        "videos",
        "photos",
        "search",
        "login",
        "subscribe",
        "advertisement",
        "privacy policy",
        "terms of service",
        "contact us",
        "read more",
        "learn more",
        "see more",
        "view all",
        "skip to content",
        "sign in",
        "sign up",
    }


def _rank_with_sentence_transformer(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    model = _get_embedding_model()
    if model is None:
        return None

    try:
        documents = [f"{item.get('title', '')}. {item.get('description', '')}" for item in results]
        query_embedding = model.encode([query], normalize_embeddings=True)[0]
        document_embeddings = model.encode(documents, normalize_embeddings=True)
    except Exception:
        return None

    scored_results: list[dict[str, Any]] = []
    for result, embedding in zip(results, document_embeddings):
        score = float(sum(float(a) * float(b) for a, b in zip(query_embedding, embedding)))
        enriched = dict(result)
        enriched["score"] = score
        scored_results.append(enriched)

    return sorted(scored_results, key=lambda item: item.get("score", 0.0), reverse=True)


def _get_embedding_model() -> Any | None:
    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    if not RAG_MODEL_PATH.exists():
        return None

    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(str(RAG_MODEL_PATH))
    except Exception:
        _EMBEDDING_MODEL = None

    return _EMBEDDING_MODEL


def _rank_with_keyword_overlap(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_terms = _tokenise(query)
    scored_results: list[dict[str, Any]] = []

    for rank_index, result in enumerate(results):
        text = f"{result.get('title', '')} {result.get('description', '')}"
        result_terms = _tokenise(text)
        overlap = len(query_terms & result_terms)
        coverage = overlap / max(1, len(query_terms))
        rank_bonus = 1.0 / (rank_index + 1)
        score = coverage + (0.15 * rank_bonus)
        enriched = dict(result)
        enriched["score"] = score
        scored_results.append(enriched)

    return sorted(scored_results, key=lambda item: item.get("score", 0.0), reverse=True)


def _tokenise(text: str) -> set[str]:
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
        "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
        "was", "what", "when", "where", "which", "who", "why", "with", "you",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 1 and token not in stopwords
    }