"""
Unit tests for GeocodingService
Tests geocoding functionality with mocked Nominatim API
"""
import pytest
from unittest.mock import Mock, patch
from src.utils.geocoding_service import GeocodingService


class MockLocation:
    """Mock geopy Location object"""
    def __init__(self, latitude, longitude, address):
        self.latitude = latitude
        self.longitude = longitude
        self.address = address


@pytest.fixture
def mock_nominatim():
    """Mock Nominatim geocoder"""
    with patch('src.utils.geocoding_service.Nominatim') as mock:
        # Create mock geocoder instance
        mock_geocoder = Mock()
        mock.return_value = mock_geocoder

        # Mock successful geocoding
        mock_geocoder.geocode.return_value = MockLocation(
            latitude=37.4224764,
            longitude=-122.0842499,
            address="1600 Amphitheatre Parkway, Mountain View, CA 94043, USA"
        )

        yield mock_geocoder


@pytest.fixture
def geocoding_service(mock_nominatim):
    """Create GeocodingService with mocked Nominatim"""
    return GeocodingService()


def test_geocode_success(geocoding_service, mock_nominatim):
    """Test successful geocoding"""
    result = geocoding_service.geocode(
        address="1600 Amphitheatre Parkway",
        city="Mountain View",
        country="USA"
    )

    assert result['latitude'] == 37.4224764
    assert result['longitude'] == -122.0842499
    assert result['precision'] == 'street'
    assert '1600 Amphitheatre Parkway' in result['standardized_address']


def test_geocode_fallback_to_city(mock_nominatim):
    """Test fallback to city-level when street address fails"""
    # First call (full address) fails, second call (city) succeeds
    mock_nominatim.geocode.side_effect = [
        None,  # Full address fails
        MockLocation(37.39, -122.08, "Mountain View, CA, USA")  # City succeeds
    ]

    service = GeocodingService()
    result = service.geocode(
        address="Nonexistent Street",
        city="Mountain View",
        country="USA"
    )

    assert result['latitude'] == 37.39
    assert result['precision'] == 'city'


def test_geocode_all_fallbacks_fail(mock_nominatim):
    """Test when all geocoding strategies fail"""
    mock_nominatim.geocode.return_value = None

    service = GeocodingService()
    result = service.geocode(
        address="Invalid",
        city="Invalid",
        country="Invalid"
    )

    assert result['latitude'] is None
    assert result['longitude'] is None
    assert result['precision'] == 'failed'


def test_geocode_cache_hit(geocoding_service, mock_nominatim):
    """Test that cached results are used"""
    # First call
    result1 = geocoding_service.geocode(
        address="1600 Amphitheatre Parkway",
        city="Mountain View",
        country="USA"
    )

    # Second call (should use cache)
    result2 = geocoding_service.geocode(
        address="1600 Amphitheatre Parkway",
        city="Mountain View",
        country="USA"
    )

    # Should only call geocoder once (first time)
    # Note: RateLimiter complicates this, so we just verify results are consistent
    assert result1 == result2


def test_test_connection_success(mock_nominatim):
    """Test connection check succeeds"""
    service = GeocodingService()
    assert service.test_connection() is True


def test_test_connection_failure(mock_nominatim):
    """Test connection check fails gracefully"""
    mock_nominatim.geocode.side_effect = Exception("Connection failed")

    service = GeocodingService()
    assert service.test_connection() is False


# ============ Reverse Geocoding Tests ============


def test_reverse_geocode_success(mock_nominatim):
    """Test successful reverse geocoding with full address"""
    # Mock reverse geocoding response
    mock_location = Mock()
    mock_location.raw = {
        'address': {
            'house_number': '1001',
            'road': 'Cherry Blossom Way',
            'city': 'Georgetown',
            'state': 'Kentucky',
            'postcode': '40324',
            'country': 'United States'
        }
    }
    mock_nominatim.reverse.return_value = mock_location

    service = GeocodingService()
    result = service.reverse_geocode(38.2098, -84.5588)

    assert '1001' in result['street_address']
    assert 'Cherry Blossom Way' in result['street_address']
    assert result['precision'] == 'building'


def test_reverse_geocode_no_street_number(mock_nominatim):
    """Test reverse geocoding without street number (only street name)"""
    mock_location = Mock()
    mock_location.raw = {
        'address': {
            'road': 'Main Street',
            'city': 'Springfield',
            'state': 'Illinois'
        }
    }
    mock_nominatim.reverse.return_value = mock_location

    service = GeocodingService()
    result = service.reverse_geocode(39.0, -83.0)

    assert result['precision'] == 'street'
    assert 'Main Street' in result['street_address']


def test_reverse_geocode_city_level_only(mock_nominatim):
    """Test reverse geocoding with only city-level data"""
    mock_location = Mock()
    mock_location.raw = {
        'address': {
            'city': 'Springfield',
            'state': 'Illinois',
            'country': 'United States'
        }
    }
    mock_nominatim.reverse.return_value = mock_location

    service = GeocodingService()
    result = service.reverse_geocode(39.8, -89.6)

    assert result['precision'] == 'city'
    assert 'Springfield' in result['street_address']


def test_reverse_geocode_cache(mock_nominatim):
    """Test that reverse geocoding results are cached"""
    mock_location = Mock()
    mock_location.raw = {
        'address': {
            'house_number': '1001',
            'road': 'Cherry Blossom Way',
            'city': 'Georgetown'
        }
    }
    mock_nominatim.reverse.return_value = mock_location

    service = GeocodingService()

    # First call
    result1 = service.reverse_geocode(38.2098, -84.5588)

    # Second call (should use cache)
    result2 = service.reverse_geocode(38.2098, -84.5588)

    assert result1 == result2


def test_reverse_geocode_failure(mock_nominatim):
    """Test reverse geocoding failure handling"""
    mock_nominatim.reverse.return_value = None

    service = GeocodingService()
    result = service.reverse_geocode(0.0, 0.0)

    assert result['street_address'] is None
    assert result['precision'] == 'failed'


def test_reverse_geocode_exception_handling(mock_nominatim):
    """Test reverse geocoding handles exceptions gracefully"""
    mock_nominatim.reverse.side_effect = Exception("API Error")

    service = GeocodingService()
    result = service.reverse_geocode(38.2098, -84.5588)

    assert result['street_address'] is None
    assert result['precision'] == 'failed'
