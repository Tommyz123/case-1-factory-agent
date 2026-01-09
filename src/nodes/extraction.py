"""
ExtractionNode: Uses LLM to extract structured facility information from search results
"""
import os
import json
import logging
from typing import List, Dict, Any
from langchain_openai import AzureChatOpenAI
from src.models import SearchResult, FacilityCandidate
from src.prompts import EXTRACTION_PROMPT
from src.utils.cache import ResultCache
from src.utils.facility_classifier import FacilityTypeClassifier

logger = logging.getLogger(__name__)


class ExtractionNode:
    """Node for extracting structured facility information using LLM"""

    def __init__(self, metrics_collector=None):
        """
        Initialize OpenAI client

        Args:
            metrics_collector: Optional MetricsCollector for tracking API usage
        """
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                temperature=0,  # Deterministic output
                max_tokens=4000
            )
            self.cache = ResultCache(cache_dir=".cache/extractions")
            self.metrics = metrics_collector
            self.model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            logger.info("ExtractionNode initialized with Azure OpenAI")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            raise

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract facility information from search results

        Args:
            state: Current agent state with search_results

        Returns:
            Updated state with extracted_facilities
        """
        search_results = state.get("search_results", [])
        company = state.get("company", "")

        logger.info(f"Extracting facilities from {len(search_results)} search results for {company}")

        if not search_results:
            logger.warning("No search results to extract from")
            state["extracted_facilities"] = []
            return state

        # Build context from search results
        context = self._build_context(search_results)

        # Call LLM to extract facilities
        try:
            facilities = self._extract_with_llm(company, context)
            state["extracted_facilities"] = facilities
            logger.info(f"Extracted {len(facilities)} facility candidates")

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            state["errors"].append(f"Extraction error: {str(e)}")
            state["extracted_facilities"] = []

        return state

    def _build_context(self, search_results: List[SearchResult]) -> str:
        """
        Build context string from search results for LLM

        Args:
            search_results: List of SearchResult objects

        Returns:
            Formatted context string
        """
        context_parts = []

        for idx, result in enumerate(search_results[:15], 1):  # Limit to 15 to avoid token limits
            context_parts.append(
                f"[Source {idx}]\n"
                f"URL: {result.url}\n"
                f"Title: {result.title}\n"
                f"Content: {result.content}\n"
            )

        return "\n".join(context_parts)

    def _extract_with_llm(self, company: str, context: str) -> List[FacilityCandidate]:
        """
        Use LLM to extract structured facility information

        Args:
            company: Company name
            context: Search results context

        Returns:
            List of FacilityCandidate objects
        """
        # Check cache (use first 1000 chars of context as key)
        cache_key = ResultCache.make_key("extract", company, context[:1000])
        cached = self.cache.get(cache_key)

        if cached:
            logger.info(f"Using cached extraction for: {company}")
            return [FacilityCandidate(**f) for f in cached]

        # Format prompt
        prompt = EXTRACTION_PROMPT.format(
            company=company,
            context=context
        )

        # Call LLM
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content

            # Track token usage with MetricsCollector
            if hasattr(response, 'response_metadata'):
                token_usage = response.response_metadata.get('token_usage', {})
                prompt_tokens = token_usage.get('prompt_tokens', 0)
                completion_tokens = token_usage.get('completion_tokens', 0)
                total_tokens = token_usage.get('total_tokens', 0)

                logger.info(f"📊 Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

                # Track with metrics collector
                if self.metrics:
                    self.metrics.track_llm_call(
                        node_name="extraction",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        model=self.model_name
                    )

            logger.debug(f"LLM response: {response_text[:500]}...")

            # Parse JSON response
            facilities = self._parse_llm_response(response_text)

            # Save to cache
            serializable = [f.model_dump() for f in facilities]
            self.cache.set(cache_key, serializable)
            logger.debug(f"Saved extraction to cache")

            return facilities

        except Exception as e:
            logger.error(f"LLM invocation error: {e}")
            raise

    def _parse_llm_response(self, response_text: str) -> List[FacilityCandidate]:
        """
        Parse LLM JSON response into FacilityCandidate objects

        Args:
            response_text: Raw LLM response

        Returns:
            List of validated FacilityCandidate objects
        """
        facilities = []

        try:
            # Extract JSON from response (handle markdown code blocks)
            json_text = response_text.strip()

            # Remove markdown code blocks if present
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            elif json_text.startswith("```"):
                json_text = json_text[3:]

            if json_text.endswith("```"):
                json_text = json_text[:-3]

            json_text = json_text.strip()

            # Parse JSON
            data = json.loads(json_text)

            # Extract facilities array
            facilities_data = data.get("facilities", [])

            if not isinstance(facilities_data, list):
                logger.error("LLM response does not contain 'facilities' array")
                return []

            # Validate each facility with Pydantic
            for facility_dict in facilities_data:
                try:
                    # Ensure evidence_urls is present and non-empty
                    if not facility_dict.get("evidence_urls"):
                        logger.warning(f"Skipping facility without evidence_urls: {facility_dict.get('facility_name')}")
                        continue

                    # Classify facility type (post-processing for better specificity)
                    original_type = facility_dict.get('facility_type', '')
                    classified_type = FacilityTypeClassifier.classify(
                        facility_name=facility_dict.get('facility_name', ''),
                        facility_type=original_type,
                        full_address=facility_dict.get('full_address', '')
                    )

                    # Update facility type if classification found something better
                    if classified_type != original_type:
                        logger.info(
                            f"Reclassified facility type: "
                            f"'{original_type}' → '{classified_type}' "
                            f"({facility_dict.get('facility_name')})"
                        )
                        facility_dict['facility_type'] = classified_type

                    # Create FacilityCandidate object (Pydantic will validate)
                    facility = FacilityCandidate(**facility_dict)
                    facilities.append(facility)

                except Exception as e:
                    logger.warning(f"Failed to parse facility: {e}")
                    logger.debug(f"Facility data: {facility_dict}")
                    continue

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response text: {response_text}")
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")

        return facilities
