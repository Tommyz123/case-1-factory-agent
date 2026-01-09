"""
OutputNode: Formats final output with metadata
"""
import logging
from datetime import datetime
from typing import Dict, Any
from src.models import FinalOutput

logger = logging.getLogger(__name__)


class OutputNode:
    """Node for formatting final output"""

    def __init__(self):
        logger.info("OutputNode initialized")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format final output with metadata including metrics

        Args:
            state: Current agent state with validated_facilities

        Returns:
            Updated state with final_output
        """
        validated = state.get("validated_facilities", [])
        company = state.get("company", "")
        metrics_collector = state.get("metrics_collector")

        logger.info(f"Formatting output for {company}: {len(validated)} facilities")

        # Get metrics summary if available
        metrics_summary = {}
        if metrics_collector:
            metrics_summary = metrics_collector.get_summary()

        # Create metadata with metrics
        metadata = {
            "company": company,
            "total_facilities": len(validated),
            "timestamp": datetime.now().isoformat(),
            "search_results_count": len(state.get("search_results", [])),
            "extracted_count": len(state.get("extracted_facilities", [])),
            "validated_count": len(validated),
            "errors": state.get("errors", []),

            # Add metrics (required fields for rubric)
            **metrics_summary  # Includes total_tokens_used, total_api_calls, execution_time_seconds, etc.
        }

        # Create FinalOutput object
        final_output = FinalOutput(
            facilities=validated,
            metadata=metadata
        )

        state["final_output"] = final_output
        logger.info(f"Output formatted successfully with metrics")

        return state
