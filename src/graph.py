"""
LangGraph workflow for Factory Search Agent
"""
import logging
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END

from src.models import SearchResult, FacilityCandidate, ValidatedFacility, FinalOutput
from src.nodes.search import SearchNode
from src.nodes.extraction import ExtractionNode
from src.nodes.validation import ValidationNode
from src.nodes.geocoding import GeocodingNode
from src.nodes.deduplication import DeduplicationNode
from src.nodes.output import OutputNode
from src.utils.metrics import MetricsCollector, NodeTimer

logger = logging.getLogger(__name__)


# Define the state structure for LangGraph
class GraphState(TypedDict):
    """State that flows through the graph"""
    company: str
    search_results: List[SearchResult]
    extracted_facilities: List[FacilityCandidate]
    validated_facilities: List[ValidatedFacility]
    final_output: Optional[FinalOutput]
    errors: List[str]
    metrics_collector: MetricsCollector


def create_factory_search_graph(metrics_collector: MetricsCollector):
    """
    Create and compile the LangGraph workflow for factory search

    Workflow:
    Input → Search → Extract → Validate → Geocode → Deduplicate → Output → END

    Args:
        metrics_collector: MetricsCollector instance for tracking metrics

    Returns:
        Compiled LangGraph workflow
    """
    logger.info("Creating factory search graph")

    # Initialize nodes with metrics collector
    search_node = SearchNode()
    extraction_node = ExtractionNode(metrics_collector=metrics_collector)
    validation_node = ValidationNode(metrics_collector=metrics_collector)
    geocoding_node = GeocodingNode()
    deduplication_node = DeduplicationNode(metrics_collector=metrics_collector)
    output_node = OutputNode()

    # Create wrapper function for timing node execution
    def timed_node(node_func, node_name: str):
        """Wrap node execution with timing"""
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            with NodeTimer(metrics_collector, node_name):
                return node_func(state)
        return wrapper

    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes to the graph with timing wrappers
    workflow.add_node("search", timed_node(search_node.run, "search"))
    workflow.add_node("extract", timed_node(extraction_node.run, "extraction"))
    workflow.add_node("validate", timed_node(validation_node.run, "validation"))
    workflow.add_node("geocode", timed_node(geocoding_node.run, "geocoding"))
    workflow.add_node("deduplicate", timed_node(deduplication_node.run, "deduplication"))
    workflow.add_node("output", timed_node(output_node.run, "output"))

    # Define the workflow edges (linear flow)
    workflow.add_edge("search", "extract")
    workflow.add_edge("extract", "validate")
    workflow.add_edge("validate", "geocode")
    workflow.add_edge("geocode", "deduplicate")
    workflow.add_edge("deduplicate", "output")
    workflow.add_edge("output", END)

    # Set entry point
    workflow.set_entry_point("search")

    # Compile the graph
    compiled_graph = workflow.compile()
    logger.info("Factory search graph compiled successfully")

    return compiled_graph


def run_factory_search(company: str) -> FinalOutput:
    """
    High-level function to run factory search for a single company

    Args:
        company: Company name to search for

    Returns:
        FinalOutput containing validated facilities with metrics
    """
    logger.info(f"Starting factory search for: {company}")

    # Create metrics collector
    metrics_collector = MetricsCollector()

    # Create the graph with metrics collector
    graph = create_factory_search_graph(metrics_collector)

    # Initialize state
    initial_state = {
        "company": company,
        "search_results": [],
        "extracted_facilities": [],
        "validated_facilities": [],
        "final_output": None,
        "errors": [],
        "metrics_collector": metrics_collector
    }

    # Run the graph
    try:
        final_state = graph.invoke(initial_state)
        result = final_state.get("final_output")

        # Log metrics summary
        metrics_collector.log_summary()

        if result:
            logger.info(f"Factory search completed for {company}: {len(result.facilities)} facilities found")
            return result
        else:
            logger.error(f"No output generated for {company}")
            # Include metrics even in failure case
            metrics_summary = metrics_collector.get_summary()
            return FinalOutput(
                facilities=[],
                metadata={
                    "company": company,
                    "error": "No output generated",
                    **metrics_summary
                }
            )

    except Exception as e:
        logger.error(f"Error running factory search for {company}: {e}")
        # Include metrics even in error case
        metrics_summary = metrics_collector.get_summary()
        return FinalOutput(
            facilities=[],
            metadata={
                "company": company,
                "error": str(e),
                **metrics_summary
            }
        )
