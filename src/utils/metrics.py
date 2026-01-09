"""
Metrics collection utility for tracking API usage and costs
"""
import time
import logging
from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class APIMetrics:
    """Track API call metrics"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    estimated_cost: float = 0.0


@dataclass
class NodeMetrics:
    """Track metrics for a specific node"""
    node_name: str
    api_metrics: APIMetrics = field(default_factory=APIMetrics)
    execution_time: float = 0.0
    start_time: float = 0.0


class MetricsCollector:
    """
    Central metrics collector for tracking API usage, costs, and execution time

    Usage:
        collector = MetricsCollector()

        # Track LLM call
        collector.track_llm_call(
            node_name="extraction",
            prompt_tokens=1000,
            completion_tokens=500,
            model="gpt-4-turbo-preview"
        )

        # Track embedding call
        collector.track_embedding_call(
            node_name="deduplication",
            total_tokens=2000,
            model="text-embedding-3-small"
        )

        # Track execution time
        with NodeTimer(collector, "geocoding"):
            # ... node execution ...
            pass

        # Get summary
        summary = collector.get_summary()
    """

    # Pricing (as of 2024)
    PRICING = {
        'gpt-4-turbo-preview': {
            'prompt': 0.01,      # per 1K tokens
            'completion': 0.03   # per 1K tokens
        },
        'gpt-4': {
            'prompt': 0.03,
            'completion': 0.06
        },
        'gpt-3.5-turbo': {
            'prompt': 0.0005,
            'completion': 0.0015
        },
        'text-embedding-3-small': {
            'usage': 0.00002     # per 1K tokens
        },
        'text-embedding-3-large': {
            'usage': 0.00013
        }
    }

    def __init__(self):
        """Initialize metrics collector"""
        self.start_time = time.time()
        self.node_metrics: Dict[str, NodeMetrics] = {}
        self.total_api_calls = 0
        self.total_cost = 0.0

    def track_llm_call(
        self,
        node_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "gpt-4-turbo-preview"
    ):
        """
        Track LLM API call metrics

        Args:
            node_name: Name of the node making the call
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name for pricing
        """
        if node_name not in self.node_metrics:
            self.node_metrics[node_name] = NodeMetrics(node_name=node_name)

        metrics = self.node_metrics[node_name].api_metrics
        metrics.prompt_tokens += prompt_tokens
        metrics.completion_tokens += completion_tokens
        metrics.total_tokens += (prompt_tokens + completion_tokens)
        metrics.api_calls += 1

        # Calculate cost
        pricing = self.PRICING.get(model, self.PRICING['gpt-4-turbo-preview'])
        cost = (prompt_tokens / 1000 * pricing['prompt']) + \
               (completion_tokens / 1000 * pricing['completion'])
        metrics.estimated_cost += cost

        self.total_api_calls += 1
        self.total_cost += cost

        logger.debug(
            f"[{node_name}] LLM call tracked: "
            f"{prompt_tokens}+{completion_tokens} tokens, ${cost:.4f}"
        )

    def track_embedding_call(
        self,
        node_name: str,
        total_tokens: int,
        model: str = "text-embedding-3-small"
    ):
        """
        Track embedding API call metrics

        Args:
            node_name: Name of the node making the call
            total_tokens: Total tokens used
            model: Embedding model name
        """
        if node_name not in self.node_metrics:
            self.node_metrics[node_name] = NodeMetrics(node_name=node_name)

        metrics = self.node_metrics[node_name].api_metrics
        metrics.total_tokens += total_tokens
        metrics.api_calls += 1

        # Calculate cost
        pricing = self.PRICING.get(model, {}).get('usage', 0.00002)
        cost = total_tokens / 1000 * pricing
        metrics.estimated_cost += cost

        self.total_api_calls += 1
        self.total_cost += cost

        logger.debug(
            f"[{node_name}] Embedding call tracked: "
            f"{total_tokens} tokens, ${cost:.4f}"
        )

    def start_node_timer(self, node_name: str):
        """Start timer for a node"""
        if node_name not in self.node_metrics:
            self.node_metrics[node_name] = NodeMetrics(node_name=node_name)
        self.node_metrics[node_name].start_time = time.time()

    def stop_node_timer(self, node_name: str):
        """Stop timer for a node"""
        if node_name in self.node_metrics and self.node_metrics[node_name].start_time > 0:
            elapsed = time.time() - self.node_metrics[node_name].start_time
            self.node_metrics[node_name].execution_time += elapsed
            logger.debug(f"[{node_name}] Execution time: {elapsed:.2f}s")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get complete metrics summary

        Returns:
            Dictionary with all metrics suitable for metadata
        """
        total_execution_time = time.time() - self.start_time

        # Aggregate totals
        total_prompt_tokens = sum(
            m.api_metrics.prompt_tokens for m in self.node_metrics.values()
        )
        total_completion_tokens = sum(
            m.api_metrics.completion_tokens for m in self.node_metrics.values()
        )
        total_tokens = sum(
            m.api_metrics.total_tokens for m in self.node_metrics.values()
        )

        # Build per-node breakdown
        node_breakdown = {}
        for node_name, metrics in self.node_metrics.items():
            node_breakdown[node_name] = {
                'api_calls': metrics.api_metrics.api_calls,
                'tokens': metrics.api_metrics.total_tokens,
                'cost': round(metrics.api_metrics.estimated_cost, 4),
                'execution_time_seconds': round(metrics.execution_time, 2)
            }

        return {
            # Required fields for rubric
            'total_tokens_used': total_tokens,
            'total_api_calls': self.total_api_calls,
            'execution_time_seconds': round(total_execution_time, 2),

            # Detailed breakdown
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens,
            'estimated_cost_usd': round(self.total_cost, 4),
            'node_breakdown': node_breakdown,
            'timestamp': datetime.now().isoformat()
        }

    def log_summary(self):
        """Log metrics summary"""
        summary = self.get_summary()

        logger.info("")
        logger.info("=" * 60)
        logger.info("API USAGE & COST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total API calls: {summary['total_api_calls']}")
        logger.info(f"Total tokens: {summary['total_tokens_used']:,}")
        logger.info(f"  - Prompt: {summary['prompt_tokens']:,}")
        logger.info(f"  - Completion: {summary['completion_tokens']:,}")
        logger.info(f"Estimated cost: ${summary['estimated_cost_usd']:.4f}")
        logger.info(f"Execution time: {summary['execution_time_seconds']:.2f}s")
        logger.info("")
        logger.info("Per-node breakdown:")
        for node, metrics in summary['node_breakdown'].items():
            logger.info(
                f"  {node}: {metrics['api_calls']} calls, "
                f"{metrics['tokens']:,} tokens, "
                f"${metrics['cost']:.4f}, "
                f"{metrics['execution_time_seconds']:.2f}s"
            )
        logger.info("=" * 60)
        logger.info("")


class NodeTimer:
    """Context manager for timing node execution"""

    def __init__(self, collector: MetricsCollector, node_name: str):
        self.collector = collector
        self.node_name = node_name

    def __enter__(self):
        self.collector.start_node_timer(self.node_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.collector.stop_node_timer(self.node_name)
        return False
