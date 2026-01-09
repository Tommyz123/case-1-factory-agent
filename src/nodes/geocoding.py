"""
GeocodingNode: Enriches facilities with geocoding data
Adds latitude/longitude and optionally improves addresses
"""
import logging
from typing import Dict, Any
from src.utils.geocoding_service import GeocodingService
from src.models import ValidatedFacility

logger = logging.getLogger(__name__)


class GeocodingNode:
    """
    Node for enriching facilities with geocoding data

    Features:
    - Adds latitude/longitude coordinates (forward geocoding)
    - Improves addresses with street-level details (reverse geocoding)
    - Two-step process: forward geocode → reverse geocode
    - Handles failures gracefully (never breaks pipeline)
    - Tracks statistics including address improvements
    """

    def __init__(self):
        """Initialize geocoding service"""
        try:
            self.geocoding_service = GeocodingService()
            self.stats = {
                'total': 0,
                'geocoded': 0,
                'cached': 0,
                'failed': 0,
                'skipped': 0,  # Already had coordinates
                'address_improved': 0,  # Improved via reverse geocoding
                'precision': {
                    'street': 0,
                    'city': 0,
                    'country': 0
                }
            }
            logger.info("GeocodingNode initialized")
        except Exception as e:
            logger.error(f"Failed to initialize GeocodingNode: {e}")
            raise

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Geocode validated facilities

        Args:
            state: Contains validated_facilities list

        Returns:
            Updated state with enriched facilities
        """
        facilities = state.get("validated_facilities", [])

        if not facilities:
            logger.warning("No validated facilities to geocode")
            return state

        logger.info(f"Starting geocoding for {len(facilities)} facilities")

        # Reset stats
        self.stats = {
            'total': len(facilities),
            'geocoded': 0,
            'cached': 0,
            'failed': 0,
            'skipped': 0,
            'address_improved': 0,
            'precision': {'street': 0, 'city': 0, 'country': 0}
        }

        # Geocode each facility
        enriched_facilities = []

        for facility in facilities:
            try:
                enriched = self._geocode_facility(facility)
                enriched_facilities.append(enriched)
            except Exception as e:
                # Should never happen due to internal error handling,
                # but catch just in case
                logger.error(f"Unexpected error geocoding {facility.facility_name}: {e}")
                enriched_facilities.append(facility)  # Keep original
                self.stats['failed'] += 1

        # Update state
        state["validated_facilities"] = enriched_facilities

        # Log summary
        self._log_stats()

        return state

    def _geocode_facility(self, facility: ValidatedFacility) -> ValidatedFacility:
        """
        Geocode a single facility using two-step process:
        1. Forward geocode to get coordinates (if missing)
        2. Reverse geocode to get complete street address

        Args:
            facility: Facility to geocode

        Returns:
            Enriched facility with coordinates and improved address
        """
        facility_data = facility.model_dump()
        coordinates_added = False

        # Step 1: Forward geocoding (if coordinates missing)
        if facility.latitude is None or facility.longitude is None:
            try:
                result = self.geocoding_service.geocode(
                    address=facility.full_address,
                    city=facility.city,
                    country=facility.country
                )

                # Update coordinates
                facility_data['latitude'] = result['latitude']
                facility_data['longitude'] = result['longitude']

                # Optionally improve address from forward geocoding
                # (if it's street-level precision and passes validation)
                if (result['latitude'] is not None and
                    result.get('standardized_address') and
                    result['precision'] == 'street'):
                    if self._is_better_address(
                        current=facility.full_address,
                        new=result['standardized_address'],
                        precision=result['precision']
                    ):
                        logger.info(f"Improved address from forward geocoding: {facility.facility_name}")
                        logger.info(f"  Before: {facility.full_address}")
                        logger.info(f"  After: {result['standardized_address']}")
                        facility_data['full_address'] = result['standardized_address']
                        self.stats['address_improved'] += 1

                # Update statistics
                if result['latitude'] is not None:
                    self.stats['geocoded'] += 1
                    coordinates_added = True
                    if result['precision'] in self.stats['precision']:
                        self.stats['precision'][result['precision']] += 1
                else:
                    self.stats['failed'] += 1
                    # No coordinates, can't do reverse geocoding
                    return ValidatedFacility(**facility_data)

            except Exception as e:
                logger.warning(f"Forward geocoding failed for {facility.facility_name}: {e}")
                self.stats['failed'] += 1
                return facility
        else:
            # Already has coordinates
            if not coordinates_added:
                self.stats['skipped'] += 1

        # Step 2: Reverse geocoding (to get street address)
        # Only if we have coordinates
        if facility_data['latitude'] is not None and facility_data['longitude'] is not None:
            try:
                reverse_result = self.geocoding_service.reverse_geocode(
                    latitude=facility_data['latitude'],
                    longitude=facility_data['longitude']
                )

                if reverse_result['street_address']:
                    # Check if reverse geocoded address is better
                    if self._is_better_address(
                        current=facility.full_address,
                        new=reverse_result['street_address'],
                        precision=reverse_result['precision']
                    ):
                        logger.info(f"Improved address for {facility.facility_name}")
                        logger.info(f"  Before: {facility.full_address}")
                        logger.info(f"  After: {reverse_result['street_address']}")

                        facility_data['full_address'] = reverse_result['street_address']
                        self.stats['address_improved'] += 1

            except Exception as e:
                # Reverse geocoding failed - keep existing address
                logger.debug(f"Reverse geocoding failed for {facility.facility_name}: {e}")

        return ValidatedFacility(**facility_data)

    def _is_better_address(self, current: str, new: str, precision: str) -> bool:
        """
        Determine if new address is better than current

        CRITICAL: Conservative strategy - only replace when CERTAIN it's better

        Rules:
        1. If current already has street address, keep it (LLM-extracted is usually accurate)
        2. Verify city name matches (globally applicable)
        3. Only use new if current is city-only AND new adds street info

        Args:
            current: Current address
            new: New address from reverse geocoding
            precision: Precision level from reverse geocoding

        Returns:
            True if new address is better
        """
        current_lower = current.lower()
        new_lower = new.lower()

        # Rule 1: If current already has street address, keep it
        current_has_street = self._has_street_address(current)
        if current_has_street:
            logger.debug(f"Current address already has street details - keeping original")
            return False

        # Rule 2: Verify city name matches (globally applicable)
        current_city = self._extract_city_from_address(current)
        new_city = self._extract_city_from_address(new)

        if current_city and new_city:
            # City names must match (allow partial match, e.g., "Norco" and "City of Norco")
            if current_city not in new_lower and new_city not in current_lower:
                logger.warning(
                    f"City mismatch - rejecting reverse geocode:\n"
                    f"  Current: {current} (city: {current_city})\n"
                    f"  New: {new} (city: {new_city})"
                )
                return False

        # Rule 3: Only use new if it adds street info AND city matches
        new_has_street = self._has_street_address(new)
        if new_has_street and precision in ['building', 'street']:
            logger.info(f"Using reverse geocode - adds street details to city-only address")
            return True

        return False

    def _has_street_address(self, address: str) -> bool:
        """
        Check if address contains street-level information

        Logic: has number + has street keyword = has street address

        Args:
            address: Address string

        Returns:
            True if address has street-level info
        """
        has_number = any(char.isdigit() for char in address[:30])  # First 30 chars
        has_street_keyword = any(keyword in address.lower() for keyword in
                                 ['street', 'road', 'avenue', 'drive', 'boulevard',
                                  'way', 'lane', 'blvd', 'ave', 'rd', 'dr', 'st'])
        return has_number and has_street_keyword

    def _extract_city_from_address(self, address: str) -> str:
        """
        Extract city name from address (globally applicable)

        Strategy: Extract first or second comma-separated part as city name

        Examples:
            "15536 River Rd, Norco, LA 70079, USA" → "norco"
            "Deer Park, Texas, USA" → "deer park"
            "Shanghai, China" → "shanghai"
            "Tokyo, Japan" → "tokyo"

        Args:
            address: Address string

        Returns:
            City name (lowercase, cleaned)
        """
        # Clean address
        address_cleaned = address.lower()

        # Split by commas
        parts = [p.strip() for p in address_cleaned.split(',')]

        if len(parts) >= 2:
            # Usually city is after first comma
            # If first part has street keywords, city is in second part
            first_part = parts[0]
            has_street_keywords = any(keyword in first_part for keyword in
                                      ['street', 'road', 'avenue', 'drive', 'boulevard',
                                       'way', 'lane', 'blvd', 'ave', 'rd', 'dr', 'st'])

            if has_street_keywords and len(parts) >= 2:
                # First part is street, second part is city
                city = parts[1]
            else:
                # First part might be city
                city = parts[0]

            # Clean city name (remove digits and zip codes)
            city = ''.join(char for char in city if not char.isdigit()).strip()

            # Remove common non-city suffixes
            for suffix in [' usa', ' united states', ' china', ' japan', ' uk', ' canada']:
                city = city.replace(suffix, '')

            return city.strip()

        return ""

    def _enrich_facility(self, facility: ValidatedFacility, geocode_result: Dict) -> ValidatedFacility:
        """
        Create enriched facility with geocoding data

        Args:
            facility: Original facility
            geocode_result: Geocoding result dictionary

        Returns:
            New ValidatedFacility with updated coordinates and address
        """
        # Create updated facility data
        facility_data = facility.model_dump()

        # Update coordinates
        facility_data['latitude'] = geocode_result['latitude']
        facility_data['longitude'] = geocode_result['longitude']

        # Optionally improve address (only if street-level precision and address is better)
        if geocode_result['precision'] == 'street' and geocode_result['standardized_address']:
            standardized = geocode_result['standardized_address']

            # Only use standardized address if it seems more complete
            # (e.g., has more words/characters than original)
            if len(standardized) > len(facility.full_address):
                logger.debug(f"Improved address for {facility.facility_name}")
                logger.debug(f"  Original: {facility.full_address}")
                logger.debug(f"  Standardized: {standardized}")
                facility_data['full_address'] = standardized

        # Create new ValidatedFacility with updated data
        return ValidatedFacility(**facility_data)

    def _log_stats(self):
        """Log geocoding statistics"""
        total = self.stats['total']
        geocoded = self.stats['geocoded']
        failed = self.stats['failed']
        skipped = self.stats['skipped']
        improved = self.stats['address_improved']

        logger.info("")
        logger.info("=" * 60)
        logger.info("Geocoding Summary")
        logger.info("=" * 60)
        logger.info(f"Total facilities: {total}")
        logger.info(f"Already had coordinates: {skipped}")
        logger.info(f"Successfully geocoded: {geocoded} ({geocoded/total*100 if total > 0 else 0:.1f}%)")
        logger.info(f"Addresses improved (via reverse geocoding): {improved}")
        logger.info(f"Failed: {failed}")

        if geocoded > 0:
            logger.info("Precision breakdown:")
            logger.info(f"  - Street-level: {self.stats['precision']['street']}")
            logger.info(f"  - City-level: {self.stats['precision']['city']}")
            logger.info(f"  - Country-level: {self.stats['precision']['country']}")

        logger.info("=" * 60)
        logger.info("")
