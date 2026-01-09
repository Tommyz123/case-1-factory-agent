"""
Facility type classifier using keyword-based mapping
Fast, zero-cost classification
"""
import re
import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)


class FacilityTypeClassifier:
    """
    Classify facility types using keyword-based heuristics

    Maps generic types like "complex", "plant", "facility" to specific types
    like "Oil Refinery", "Chemical Plant", "Automotive Assembly Plant"
    """

    # Specific facility type mapping
    # Format: {specific_type: [keywords]}
    TYPE_MAPPINGS: Dict[str, Set[str]] = {
        'Oil Refinery': {
            'refinery', 'petroleum', 'crude oil', 'oil processing',
            'oil refining', 'distillation', 'oil and gas'
        },
        'Chemical Plant': {
            'chemical', 'chemicals', 'petrochemical', 'polymer',
            'plastics', 'resin', 'fertilizer', 'pharmaceutical manufacturing',
            'specialty chemicals', 'industrial chemicals', 'catalyst', 'catalysts'
        },
        'Automotive Assembly Plant': {
            'automotive', 'automobile', 'car assembly', 'vehicle assembly',
            'auto manufacturing', 'car manufacturing', 'vehicle manufacturing',
            'automotive production', 'car plant', 'vehicle plant', 'motor', 'motors'
        },
        'Electronics Factory': {
            'electronics', 'electronic', 'semiconductor', 'chip', 'microchip',
            'circuit', 'pcb', 'printed circuit', 'consumer electronics',
            'semiconductor fab', 'wafer fab'
        },
        'Steel Mill': {
            'steel', 'steel mill', 'steel works', 'steel plant',
            'steel production', 'steelmaking', 'integrated steel',
            'mini mill', 'rolling mill'
        },
        'Pharmaceutical Plant': {
            'pharmaceutical', 'pharma', 'drug manufacturing', 'medicine',
            'biopharmaceutical', 'biotech manufacturing', 'drug production',
            'active pharmaceutical ingredient', 'api manufacturing'
        },
        'Food Processing Plant': {
            'food processing', 'food production', 'food manufacturing',
            'beverage', 'brewery', 'dairy', 'meat processing',
            'food plant', 'bakery', 'cannery', 'food factory'
        },
        'Textile Mill': {
            'textile', 'fabric', 'garment', 'apparel', 'clothing',
            'spinning', 'weaving', 'dyeing', 'textile manufacturing'
        },
        'Semiconductor Fab': {
            'semiconductor fabrication', 'fab', 'wafer fabrication',
            'chip fabrication', 'semiconductor factory', 'foundry'
        },
        'Aircraft Assembly Plant': {
            'aircraft', 'airplane', 'aerospace', 'aviation',
            'aircraft assembly', 'aircraft manufacturing', 'aircraft production'
        },
        'Power Plant': {
            'power plant', 'power station', 'generating station',
            'power generation', 'nuclear plant', 'coal plant',
            'gas plant', 'hydroelectric', 'solar farm', 'wind farm'
        },
        'Mining Operation': {
            'mine', 'mining', 'quarry', 'extraction', 'ore processing',
            'mineral processing', 'coal mine', 'gold mine', 'copper mine'
        },
        'Cement Plant': {
            'cement', 'concrete', 'cement manufacturing', 'cement production',
            'clinker', 'cement works'
        },
        'Paper Mill': {
            'paper', 'pulp', 'paperboard', 'paper manufacturing',
            'paper production', 'pulp mill', 'paper mill'
        },
        'Metalworking Plant': {
            'metalworking', 'metal fabrication', 'foundry', 'forge',
            'casting', 'machining', 'metal stamping', 'metal forming'
        }
    }

    @classmethod
    def classify(
        cls,
        facility_name: str,
        facility_type: str,
        full_address: str = ""
    ) -> str:
        """
        Classify facility type using keyword matching

        Args:
            facility_name: Name of the facility
            facility_type: Current (possibly generic) type
            full_address: Address (may contain hints about type)

        Returns:
            Specific facility type (or enhanced generic if no match)
        """
        # Combine all text for matching
        combined_text = f"{facility_name} {facility_type} {full_address}".lower()

        # Try to find specific type
        matches = []
        for specific_type, keywords in cls.TYPE_MAPPINGS.items():
            for keyword in keywords:
                # Use word boundary matching to avoid false positives
                if re.search(r'\b' + re.escape(keyword) + r'\b', combined_text):
                    matches.append((specific_type, keyword))
                    break

        if matches:
            # Return first match (most specific)
            matched_type = matches[0][0]
            logger.debug(
                f"Classified '{facility_name}' as '{matched_type}' "
                f"(matched: {matches[0][1]})"
            )
            return matched_type

        # No match - return enhanced generic type
        return cls._enhance_generic_type(facility_type)

    @classmethod
    def _enhance_generic_type(cls, facility_type: str) -> str:
        """
        Enhance generic types to be slightly more descriptive

        Args:
            facility_type: Generic type like "complex", "plant", "facility"

        Returns:
            Enhanced type
        """
        type_lower = facility_type.lower()

        # Map generic terms to better alternatives
        generic_mappings = {
            'complex': 'Manufacturing Complex',
            'plant': 'Manufacturing Plant',
            'facility': 'Manufacturing Facility',
            'factory': 'Manufacturing Factory',
            'site': 'Manufacturing Site',
            'center': 'Production Center',
            'works': 'Manufacturing Works'
        }

        for generic, enhanced in generic_mappings.items():
            if generic in type_lower:
                return enhanced

        # If no match, capitalize first letter of each word
        return facility_type.title()
