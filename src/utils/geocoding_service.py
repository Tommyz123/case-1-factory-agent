"""
Geocoding Service using Nominatim (OpenStreetMap)
Provides free geocoding with caching and fallback strategies
"""
import logging
from typing import Dict, Optional, Tuple
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable
from geopy.extra.rate_limiter import RateLimiter
from src.utils.cache import ResultCache
from src.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)


class GeocodingService:
    """
    Free geocoding service using Nominatim (OpenStreetMap)

    Features:
    - Forward geocoding: address → coordinates
    - Reverse geocoding: coordinates → street address
    - Automatic rate limiting (1 req/sec for Nominatim compliance)
    - Result caching (7-day TTL)
    - Progressive fallback strategy (full address → city → country)
    - Timeout handling and retry logic
    - Address standardization
    """

    def __init__(self, cache_dir: str = ".cache/geocoding", ttl_days: int = 7, timeout: int = 10):
        """
        Initialize geocoding service

        Args:
            cache_dir: Directory for geocoding cache
            ttl_days: Cache time-to-live in days
            timeout: Timeout for geocoding requests in seconds
        """
        # Initialize Nominatim (requires User-Agent)
        self.geocoder = Nominatim(
            user_agent="factory-search-agent/1.0",
            timeout=timeout
        )

        # Apply rate limiting (1 request per second for Nominatim compliance)
        self.geocode_with_rate_limit = RateLimiter(
            self.geocoder.geocode,
            min_delay_seconds=1.0
        )

        # Apply rate limiting to reverse geocoding
        self.reverse_with_rate_limit = RateLimiter(
            self.geocoder.reverse,
            min_delay_seconds=1.0
        )

        # Initialize cache
        self.cache = ResultCache(cache_dir=cache_dir, ttl_days=ttl_days)

        logger.info(f"GeocodingService initialized (cache: {cache_dir}, TTL: {ttl_days} days)")

    def geocode(self, address: str, city: str, country: str) -> Dict:
        """
        Geocode an address with fallback strategies

        Args:
            address: Full street address
            city: City name
            country: Country name

        Returns:
            Dictionary with keys:
            - latitude: float or None
            - longitude: float or None
            - standardized_address: str or None
            - precision: str ('street', 'city', 'country', or 'failed')
        """
        # Check cache first
        cache_key = ResultCache.make_key("geocode", address, city, country)
        cached = self.cache.get(cache_key)

        if cached:
            logger.debug(f"Geocoding cache hit for: {city}, {country}")
            return cached

        # Try geocoding with fallback strategies
        result = self._geocode_with_fallback(address, city, country)

        # Cache the result
        self.cache.set(cache_key, result)

        return result

    def reverse_geocode(self, latitude: float, longitude: float) -> Dict:
        """
        Reverse geocode coordinates to get street address

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate

        Returns:
            Dictionary with keys:
            - street_address: str or None (complete street address)
            - raw_data: dict (raw OSM data)
            - precision: str ('building', 'street', 'city', 'region', or 'failed')
        """
        # Check cache first
        cache_key = ResultCache.make_key("reverse", latitude, longitude)
        cached = self.cache.get(cache_key)

        if cached:
            logger.debug(f"Reverse geocoding cache hit for: ({latitude}, {longitude})")
            return cached

        try:
            # Reverse geocode with rate limiting
            location = self.reverse_with_rate_limit(
                (latitude, longitude),
                exactly_one=True,
                language='en'
            )

            if location and location.raw:
                # Extract address from response
                address_parts = location.raw.get('address', {})

                # Build street-level address
                street_address = self._build_street_address(address_parts)

                result = {
                    'street_address': street_address,
                    'raw_data': location.raw,
                    'precision': self._determine_precision(address_parts)
                }

                # Cache the result
                self.cache.set(cache_key, result)

                logger.info(f"Reverse geocoded: ({latitude}, {longitude}) -> {street_address}")
                return result

        except Exception as e:
            logger.warning(f"Reverse geocoding failed for ({latitude}, {longitude}): {e}")

        # Return failure result
        return {
            'street_address': None,
            'raw_data': None,
            'precision': 'failed'
        }

    def _build_street_address(self, address_parts: Dict) -> Optional[str]:
        """
        Build complete street address from OSM address parts

        Args:
            address_parts: Dictionary of address components from OSM

        Returns:
            Formatted street address or None if no components available
        """
        components = []

        # Street number + street name
        if 'house_number' in address_parts:
            components.append(address_parts['house_number'])
        if 'road' in address_parts:
            components.append(address_parts['road'])
        elif 'street' in address_parts:
            components.append(address_parts['street'])

        # City - CRITICAL FIX: Add 'hamlet' to fallback chain
        # OSM uses different keys for settlements: city > town > village > hamlet
        # Without 'hamlet', addresses in small settlements lose their city name
        city = (address_parts.get('city') or
                address_parts.get('town') or
                address_parts.get('village') or
                address_parts.get('hamlet'))
        if city:
            components.append(city)

        # State/Province
        state = address_parts.get('state')
        if state:
            components.append(state)

        # Postal code
        postcode = address_parts.get('postcode')
        if postcode:
            components.append(postcode)

        # Country
        country = address_parts.get('country')
        if country:
            components.append(country)

        return ', '.join(components) if components else None

    def _determine_precision(self, address_parts: Dict) -> str:
        """
        Determine address precision level based on available components

        Args:
            address_parts: Dictionary of address components from OSM

        Returns:
            Precision level: 'building', 'street', 'city', or 'region'
        """
        if 'house_number' in address_parts and 'road' in address_parts:
            return 'building'
        elif 'road' in address_parts or 'street' in address_parts:
            return 'street'
        elif ('city' in address_parts or 'town' in address_parts or
              'village' in address_parts or 'hamlet' in address_parts):
            return 'city'
        else:
            return 'region'

    def _geocode_with_fallback(self, address: str, city: str, country: str) -> Dict:
        """
        Attempt geocoding with progressive fallback

        CRITICAL FIX: Use full address in fallback to preserve state/province information
        This prevents same-name city confusion (e.g., Deer Park TX vs Deer Park IL)

        Strategy:
        1. Try full address with country
        2. Fallback to FULL ADDRESS (preserves state/province info)
        3. Fallback to city + country (last resort, may be ambiguous)
        4. Fallback to just country
        5. Return None values if all fail
        """
        # Strategy 1: Try full address with country
        result = self._try_geocode(f"{address}, {country}", precision="street")
        if result['latitude'] is not None:
            logger.info(f"Geocoded (street-level): {city}, {country}")
            return result

        # Strategy 2: Fallback to FULL ADDRESS (not just city!)
        # CRITICAL: Use full address to preserve state/province information
        # Example: "Deer Park, Texas, USA" instead of just "Deer Park, USA"
        # This prevents OSM from returning wrong state coordinates
        result = self._try_geocode(f"{address}", precision="city")
        if result['latitude'] is not None:
            logger.info(f"Geocoded (full address, city-level): {city}, {country}")
            return result

        # Strategy 3: Fallback to city + country (last resort - may be ambiguous!)
        result = self._try_geocode(f"{city}, {country}", precision="city")
        if result['latitude'] is not None:
            logger.warning(f"Geocoded (city-only, may be ambiguous): {city}, {country}")
            return result

        # Strategy 4: Fallback to country only
        result = self._try_geocode(country, precision="country")
        if result['latitude'] is not None:
            logger.warning(f"Geocoded (country-level only): {country}")
            return result

        # All strategies failed
        logger.warning(f"Geocoding failed for: {address}, {city}, {country}")
        return {
            'latitude': None,
            'longitude': None,
            'standardized_address': None,
            'precision': 'failed'
        }

    @retry_with_backoff(retries=3, backoff_factor=2, exceptions=(GeocoderTimedOut, GeocoderServiceError))
    def _try_geocode(self, query: str, precision: str) -> Dict:
        """
        Attempt to geocode a single query with retry logic

        Args:
            query: Address query string
            precision: Precision level ('street', 'city', 'country')

        Returns:
            Geocoding result dictionary
        """
        try:
            location = self.geocode_with_rate_limit(query)

            if location:
                return {
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'standardized_address': location.address,
                    'precision': precision
                }
            else:
                return {
                    'latitude': None,
                    'longitude': None,
                    'standardized_address': None,
                    'precision': 'failed'
                }

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            # Retry will be handled by decorator
            logger.debug(f"Geocoding error (will retry): {e}")
            raise

        except GeocoderUnavailable as e:
            # Service unavailable - don't retry
            logger.warning(f"Geocoding service unavailable: {e}")
            return {
                'latitude': None,
                'longitude': None,
                'standardized_address': None,
                'precision': 'failed'
            }

        except Exception as e:
            # Unexpected error - don't retry
            logger.error(f"Unexpected geocoding error: {e}")
            return {
                'latitude': None,
                'longitude': None,
                'standardized_address': None,
                'precision': 'failed'
            }

    def test_connection(self) -> bool:
        """
        Test if geocoding service is available

        Returns:
            True if service is reachable, False otherwise
        """
        try:
            # Try a simple query
            result = self.geocode_with_rate_limit("United States")
            return result is not None
        except Exception as e:
            logger.error(f"Geocoding service connection test failed: {e}")
            return False
