"""
ValidationNode: Validates that facilities are real manufacturing sites
CRITICAL: Must filter out headquarters, offices, and R&D centers
"""
import os
import re
import logging
from typing import List, Dict, Any, Set
from langchain_openai import AzureChatOpenAI
from src.models import FacilityCandidate, ValidatedFacility
from src.prompts import VALIDATION_PROMPT

logger = logging.getLogger(__name__)


class ValidationNode:
    """
    Node for validating facilities using 2-layer approach:
    1. Keyword filtering (fast, no cost)
    2. LLM validation (slower, but accurate)
    """

    # CRITICAL: Keywords that indicate NON-manufacturing facilities
    # If any of these appear in facility name or address, REJECT it
    EXCLUDE_KEYWORDS: Set[str] = {
        # Headquarters
        'headquarters', 'hq', 'head office', 'corporate office',
        'corporate headquarters', 'global headquarters', 'regional headquarters',
        'main office', 'central office',

        # R&D (CRITICAL - automatic fail per rubric)
        'r&d', 'r & d', 'research', 'development', 'innovation center',
        'innovation centre', 'technology center', 'technology centre',
        'research center', 'research centre', 'development center',
        'research facility', 'development facility', 'tech center',
        'technical center', 'technical centre', 'laboratory',

        # Offices
        'sales office', 'regional office', 'branch office',
        'administrative', 'administration', 'admin building',
        'office building', 'office complex', 'business center',

        # Distribution/Logistics (not manufacturing)
        'distribution center', 'distribution centre', 'warehouse',
        'logistics center', 'logistics centre', 'fulfillment center',
        'service center', 'service centre', 'depot',

        # Other non-manufacturing
        'showroom', 'retail', 'store', 'shop', 'dealership',
        'training center', 'training centre', 'call center'
    }

    # Short keywords that need word-boundary matching to avoid false positives
    # (e.g., 'hq' should not match 'chq' in an address)
    WORD_BOUNDARY_KEYWORDS: Set[str] = {'hq'}

    # Keywords that indicate REAL manufacturing facilities
    REQUIRED_KEYWORDS: Set[str] = {
        'factory', 'plant', 'manufacturing', 'production',
        'assembly', 'refinery', 'mill', 'works', 'foundry',
        'fabrication', 'processing', 'smelter', 'forge'
    }

    def __init__(self, metrics_collector=None):
        """
        Initialize OpenAI client for LLM validation

        Args:
            metrics_collector: Optional MetricsCollector for tracking API usage
        """
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                temperature=0,
                max_tokens=100  # Validation only needs YES/NO
            )
            self.metrics = metrics_collector
            self.model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            logger.info("ValidationNode initialized with Azure OpenAI")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI for validation: {e}")
            raise

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted facilities using 2-layer approach

        Args:
            state: Current agent state with extracted_facilities

        Returns:
            Updated state with validated_facilities
        """
        extracted = state.get("extracted_facilities", [])
        logger.info(f"Validating {len(extracted)} extracted facilities")

        validated = []

        for facility in extracted:
            # Layer 1: Keyword filtering (fast, no cost)
            keyword_result, keyword_reason = self._keyword_filter(facility)

            if not keyword_result:
                # Failed keyword filter - create ValidatedFacility with failure
                validated_facility = ValidatedFacility(
                    **facility.model_dump(),
                    validation_passed=False,
                    validation_reason=keyword_reason
                )
                logger.info(f"REJECTED (keyword): {facility.facility_name} - {keyword_reason}")
                # Note: We don't add rejected facilities to the output
                continue

            # Layer 2: LLM validation (slower, but accurate)
            llm_result, llm_reason = self._llm_validation(facility)

            if llm_result:
                # PASSED all validation
                validated_facility = ValidatedFacility(
                    **facility.model_dump(),
                    validation_passed=True,
                    validation_reason="Passed keyword and LLM validation"
                )
                validated.append(validated_facility)
                logger.info(f"APPROVED: {facility.facility_name}")
            else:
                # Failed LLM validation
                logger.info(f"REJECTED (LLM): {facility.facility_name} - {llm_reason}")

        state["validated_facilities"] = validated
        logger.info(f"Validation complete: {len(validated)} facilities approved")

        return state

    def _keyword_filter(self, facility: FacilityCandidate) -> tuple[bool, str]:
        """
        Layer 1: Keyword-based filtering

        Args:
            facility: Facility candidate to check

        Returns:
            (passed: bool, reason: str)
        """
        # Combine name and address for checking
        text = f"{facility.facility_name} {facility.full_address} {facility.facility_type}".lower()

        # Check for EXCLUDE keywords (any match = reject immediately)
        for keyword in self.EXCLUDE_KEYWORDS:
            # Use word boundary matching for short keywords to avoid false positives
            if keyword in self.WORD_BOUNDARY_KEYWORDS:
                # Use regex word boundary (\b) for precise matching
                if re.search(rf'\b{re.escape(keyword)}\b', text):
                    return False, f"Contains excluded keyword: '{keyword}'"
            else:
                # Regular substring matching for longer keywords
                if keyword in text:
                    return False, f"Contains excluded keyword: '{keyword}'"

        # Check for REQUIRED keywords (enhanced signal, not hard requirement)
        # If facility has manufacturing keywords, it's a strong positive signal
        # If not, we'll let the LLM make the final judgment
        has_required = any(keyword in text for keyword in self.REQUIRED_KEYWORDS)

        if has_required:
            logger.debug(f"Facility '{facility.facility_name}' has manufacturing keywords - strong positive signal")
        else:
            logger.debug(f"Facility '{facility.facility_name}' lacks manufacturing keywords - LLM will decide")

        # PASS keyword filter - let LLM do final validation
        # This improves recall for real facilities that lack obvious keywords
        return True, "Passed keyword filter (no exclusions found)"

    def _llm_validation(self, facility: FacilityCandidate) -> tuple[bool, str]:
        """
        Layer 2: LLM-based validation for more nuanced cases

        Args:
            facility: Facility candidate to validate

        Returns:
            (passed: bool, reason: str)
        """
        try:
            # Format validation prompt
            evidence = facility.evidence_urls[0] if facility.evidence_urls else "No evidence"

            prompt = VALIDATION_PROMPT.format(
                facility_name=facility.facility_name,
                facility_type=facility.facility_type,
                address=facility.full_address,
                evidence=evidence
            )

            # Call LLM
            response = self.llm.invoke(prompt)
            answer = response.content.strip().upper()

            # Track token usage
            if hasattr(response, 'response_metadata') and self.metrics:
                token_usage = response.response_metadata.get('token_usage', {})
                prompt_tokens = token_usage.get('prompt_tokens', 0)
                completion_tokens = token_usage.get('completion_tokens', 0)

                self.metrics.track_llm_call(
                    node_name="validation",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model=self.model_name
                )

            logger.debug(f"LLM validation for '{facility.facility_name}': {answer}")

            # Parse response
            if "YES" in answer:
                return True, "LLM confirmed manufacturing facility"
            else:
                return False, "LLM rejected as non-manufacturing facility"

        except Exception as e:
            logger.error(f"LLM validation error: {e}")
            # Conservative: if validation fails, reject the facility
            return False, f"Validation error: {str(e)}"
