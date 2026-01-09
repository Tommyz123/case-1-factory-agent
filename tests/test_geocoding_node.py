"""
Unit tests for GeocodingNode
Tests node integration with workflow
"""
import pytest
from unittest.mock import Mock, patch
from src.nodes.geocoding import GeocodingNode
from src.models import ValidatedFacility


@pytest.fixture
def mock_geocoding_service():
    """Mock GeocodingService"""
    with patch('src.nodes.geocoding.GeocodingService') as mock:
        # Create mock service instance
        mock_service = Mock()
        mock.return_value = mock_service

        # Mock successful geocoding
        mock_service.geocode.return_value = {
            'latitude': 38.2098,
            'longitude': -84.5588,
            'standardized_address': '1001 Cherry Blossom Way, Georgetown, KY 40324, USA',
            'precision': 'street'
        }

        yield mock_service


@pytest.fixture
def sample_facility():
    """Create sample facility without coordinates"""
    return ValidatedFacility(
        company_name="Toyota",
        facility_name="Georgetown Plant",
        facility_type="assembly plant",
        full_address="Georgetown, KY",
        country="USA",
        city="Georgetown",
        latitude=None,
        longitude=None,
        evidence_urls=["https://example.com"],
        confidence_score=0.9,
        validation_passed=True,
        validation_reason="Test facility"
    )


def test_geocoding_node_enriches_facility(mock_geocoding_service, sample_facility):
    """Test that node adds coordinates to facility"""
    node = GeocodingNode()

    state = {
        "validated_facilities": [sample_facility]
    }

    result_state = node.run(state)

    # Check that facility was enriched
    enriched = result_state["validated_facilities"][0]
    assert enriched.latitude == 38.2098
    assert enriched.longitude == -84.5588


def test_geocoding_node_skips_already_geocoded(mock_geocoding_service):
    """Test that facilities with coordinates are skipped"""
    facility_with_coords = ValidatedFacility(
        company_name="Toyota",
        facility_name="Georgetown Plant",
        facility_type="assembly plant",
        full_address="Georgetown, KY",
        country="USA",
        city="Georgetown",
        latitude=38.0,
        longitude=-84.0,
        evidence_urls=["https://example.com"],
        confidence_score=0.9,
        validation_passed=True,
        validation_reason="Test facility"
    )

    node = GeocodingNode()
    state = {"validated_facilities": [facility_with_coords]}

    result_state = node.run(state)

    # Geocoding service should not be called
    mock_geocoding_service.geocode.assert_not_called()

    # Coordinates should remain unchanged
    enriched = result_state["validated_facilities"][0]
    assert enriched.latitude == 38.0
    assert enriched.longitude == -84.0


def test_geocoding_node_handles_failure_gracefully(mock_geocoding_service, sample_facility):
    """Test that geocoding failures don't break pipeline"""
    # Mock geocoding failure
    mock_geocoding_service.geocode.return_value = {
        'latitude': None,
        'longitude': None,
        'standardized_address': None,
        'precision': 'failed'
    }

    node = GeocodingNode()
    state = {"validated_facilities": [sample_facility]}

    # Should not raise exception
    result_state = node.run(state)

    # Facility should be in result with null coordinates
    enriched = result_state["validated_facilities"][0]
    assert enriched.latitude is None
    assert enriched.longitude is None


def test_geocoding_node_empty_facilities():
    """Test handling of empty facilities list"""
    node = GeocodingNode()
    state = {"validated_facilities": []}

    result_state = node.run(state)

    assert result_state["validated_facilities"] == []


def test_geocoding_node_improves_address(mock_geocoding_service, sample_facility):
    """Test that street-level geocoding improves address"""
    # Mock returns better address
    mock_geocoding_service.geocode.return_value = {
        'latitude': 38.2098,
        'longitude': -84.5588,
        'standardized_address': '1001 Cherry Blossom Way, Georgetown, KY 40324, USA',
        'precision': 'street'
    }

    node = GeocodingNode()
    state = {"validated_facilities": [sample_facility]}

    result_state = node.run(state)

    enriched = result_state["validated_facilities"][0]

    # Address should be improved (standardized is longer than original)
    assert '1001 Cherry Blossom Way' in enriched.full_address


# ============ Reverse Geocoding Integration Tests ============


def test_geocoding_node_uses_reverse_geocoding(mock_geocoding_service, sample_facility):
    """Test that node uses reverse geocoding to improve addresses"""
    # Mock forward geocoding (adds coordinates)
    mock_geocoding_service.geocode.return_value = {
        'latitude': 38.2098,
        'longitude': -84.5588,
        'standardized_address': 'Georgetown, KY',
        'precision': 'city'
    }

    # Mock reverse geocoding (adds street details)
    mock_geocoding_service.reverse_geocode.return_value = {
        'street_address': '1001 Cherry Blossom Way, Georgetown, KY 40324, USA',
        'precision': 'building'
    }

    node = GeocodingNode()
    state = {"validated_facilities": [sample_facility]}
    result = node.run(state)

    enriched = result["validated_facilities"][0]

    # Address should be improved via reverse geocoding
    assert '1001 Cherry Blossom Way' in enriched.full_address
    assert enriched.latitude == 38.2098
    assert enriched.longitude == -84.5588


def test_geocoding_node_skips_reverse_if_address_not_better(mock_geocoding_service):
    """Test that reverse geocoding doesn't downgrade good addresses"""
    # Facility with already good address
    facility_with_good_address = ValidatedFacility(
        company_name="Toyota",
        facility_name="Georgetown Plant",
        facility_type="assembly plant",
        full_address="1001 Cherry Blossom Way, Georgetown, KY 40324, USA",
        country="USA",
        city="Georgetown",
        latitude=None,
        longitude=None,
        evidence_urls=["https://example.com"],
        confidence_score=0.9,
        validation_passed=True,
        validation_reason="Test facility"
    )

    # Mock forward geocoding
    mock_geocoding_service.geocode.return_value = {
        'latitude': 38.2098,
        'longitude': -84.5588,
        'standardized_address': 'Georgetown, KY',
        'precision': 'city'
    }

    # Mock reverse geocoding returns less precise address
    mock_geocoding_service.reverse_geocode.return_value = {
        'street_address': 'Georgetown, Kentucky, United States',
        'precision': 'city'
    }

    node = GeocodingNode()
    state = {"validated_facilities": [facility_with_good_address]}
    result = node.run(state)

    enriched = result["validated_facilities"][0]

    # Address should NOT be changed (original is better)
    assert enriched.full_address == "1001 Cherry Blossom Way, Georgetown, KY 40324, USA"


def test_geocoding_node_handles_reverse_geocoding_failure(mock_geocoding_service, sample_facility):
    """Test that reverse geocoding failures don't break pipeline"""
    # Mock forward geocoding success
    mock_geocoding_service.geocode.return_value = {
        'latitude': 38.2098,
        'longitude': -84.5588,
        'standardized_address': 'Georgetown, KY',
        'precision': 'city'
    }

    # Mock reverse geocoding failure
    mock_geocoding_service.reverse_geocode.return_value = {
        'street_address': None,
        'precision': 'failed'
    }

    node = GeocodingNode()
    state = {"validated_facilities": [sample_facility]}

    # Should not raise exception
    result_state = node.run(state)

    # Facility should have coordinates but original address
    enriched = result_state["validated_facilities"][0]
    assert enriched.latitude == 38.2098
    assert enriched.longitude == -84.5588
    assert enriched.full_address == sample_facility.full_address


def test_geocoding_node_statistics_include_improvements(mock_geocoding_service, sample_facility):
    """Test that statistics track address improvements"""
    # Mock forward geocoding
    mock_geocoding_service.geocode.return_value = {
        'latitude': 38.2098,
        'longitude': -84.5588,
        'standardized_address': 'Georgetown, KY',
        'precision': 'city'
    }

    # Mock reverse geocoding with better address
    mock_geocoding_service.reverse_geocode.return_value = {
        'street_address': '1001 Cherry Blossom Way, Georgetown, KY 40324, USA',
        'precision': 'building'
    }

    node = GeocodingNode()
    state = {"validated_facilities": [sample_facility]}
    node.run(state)

    # Check that address_improved counter was incremented
    assert node.stats['address_improved'] == 1
