"""
SearchNode: Searches the web for manufacturing facility information
Supports multiple search methods: Tavily API, SerpAPI, and web scraping fallback
"""
import os
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from src.models import SearchResult, AgentState
from src.utils.cache import ResultCache
from src.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)


class SearchNode:
    """Node for searching manufacturing facilities information"""

    def __init__(self):
        self.search_method = os.getenv("SEARCH_METHOD", "web_scraping")
        logger.info(f"SearchNode initialized with method: {self.search_method}")

        # Initialize cache
        self.cache = ResultCache(cache_dir=".cache/search_results")

        # Initialize search clients based on method
        if self.search_method == "tavily" and os.getenv("TAVILY_API_KEY"):
            try:
                from tavily import TavilyClient
                self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
                logger.info("Tavily Search API initialized")
            except ImportError:
                logger.warning("tavily-python not installed, falling back to web scraping")
                self.search_method = "web_scraping"

        elif self.search_method == "serpapi" and os.getenv("SERPAPI_API_KEY"):
            try:
                from serpapi import GoogleSearch
                self.serpapi_key = os.getenv("SERPAPI_API_KEY")
                logger.info("SerpAPI initialized")
            except ImportError:
                logger.warning("google-search-results not installed, falling back to web scraping")
                self.search_method = "web_scraping"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search for manufacturing facilities

        Args:
            state: Current agent state containing company name

        Returns:
            Updated state with search results
        """
        company = state.get("company", "")
        logger.info(f"Starting search for company: {company}")

        if not company:
            logger.error("No company name provided")
            state["errors"].append("No company name provided for search")
            return state

        # Check cache
        cache_key = ResultCache.make_key("search", company)
        cached_results = self.cache.get(cache_key)

        if cached_results:
            logger.info(f"Using cached search results for: {company}")
            state["search_results"] = [SearchResult(**r) for r in cached_results]
            return state

        # Define search queries optimized for manufacturing facilities
        queries = [
            f"{company} manufacturing facilities locations addresses",
            f"{company} factories plants worldwide addresses",
            f"{company} production sites manufacturing -headquarters -office -HQ"
        ]

        all_results = []
        for query in queries:
            try:
                if self.search_method == "tavily":
                    results = self._search_tavily(query)
                elif self.search_method == "serpapi":
                    results = self._search_serpapi(query)
                else:
                    # Default: web scraping
                    results = self._search_web_scraping(query)

                all_results.extend(results)
                logger.info(f"Query '{query}' returned {len(results)} results")

            except Exception as e:
                logger.error(f"Error searching with query '{query}': {e}")
                state["errors"].append(f"Search error for query '{query}': {str(e)}")

        # Convert to SearchResult objects
        search_results = []
        for result in all_results:
            try:
                search_results.append(SearchResult(**result))
            except Exception as e:
                logger.warning(f"Failed to parse search result: {e}")

        # Remove duplicates based on URL
        unique_results = {}
        for result in search_results:
            if result.url not in unique_results:
                unique_results[result.url] = result

        state["search_results"] = list(unique_results.values())
        logger.info(f"Total unique search results: {len(state['search_results'])}")

        # Save to cache
        serializable_results = [r.model_dump() for r in state["search_results"]]
        self.cache.set(cache_key, serializable_results)
        logger.debug(f"Saved search results to cache")

        return state

    def _search_tavily(self, query: str) -> List[Dict[str, str]]:
        """
        Search using Tavily Search API

        Args:
            query: Search query string

        Returns:
            List of search results
        """
        try:
            response = self.tavily.search(
                query=query,
                max_results=5,
                search_depth="advanced"
            )

            results = []
            for item in response.get("results", []):
                results.append({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "content": item.get("content", "")
                })
            return results

        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []

    def _search_serpapi(self, query: str) -> List[Dict[str, str]]:
        """
        Search using SerpAPI (Google Search)

        Args:
            query: Search query string

        Returns:
            List of search results
        """
        try:
            from serpapi import GoogleSearch

            search = GoogleSearch({
                "q": query,
                "api_key": self.serpapi_key,
                "num": 5
            })

            response = search.get_dict()
            results = []

            for item in response.get("organic_results", []):
                results.append({
                    "url": item.get("link", ""),
                    "title": item.get("title", ""),
                    "content": item.get("snippet", "")
                })
            return results

        except Exception as e:
            logger.error(f"SerpAPI search error: {e}")
            return []

    @retry_with_backoff(retries=3, exceptions=(requests.exceptions.RequestException,))
    def _search_web_scraping(self, query: str) -> List[Dict[str, str]]:
        """
        Search using web scraping (DuckDuckGo HTML)
        No API key required, fallback method

        Args:
            query: Search query string

        Returns:
            List of search results
        """
        try:
            # Use DuckDuckGo HTML search (no API required)
            search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            response = requests.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            # Parse search results
            for result_div in soup.find_all('div', class_='result')[:5]:
                try:
                    # Extract link
                    link_tag = result_div.find('a', class_='result__a')
                    if not link_tag:
                        continue

                    url = link_tag.get('href', '')
                    title = link_tag.get_text(strip=True)

                    # Extract snippet
                    snippet_tag = result_div.find('a', class_='result__snippet')
                    content = snippet_tag.get_text(strip=True) if snippet_tag else ""

                    if url and title:
                        results.append({
                            "url": url,
                            "title": title,
                            "content": content
                        })

                except Exception as e:
                    logger.debug(f"Error parsing individual result: {e}")
                    continue

            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Web scraping request error: {e}")
            return []
        except Exception as e:
            logger.error(f"Web scraping error: {e}")
            return []
