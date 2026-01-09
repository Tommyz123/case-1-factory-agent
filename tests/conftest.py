"""
Pytest配置和共享fixtures
"""
import pytest
import os
from unittest.mock import Mock, MagicMock, patch
from src.models import SearchResult, FacilityCandidate, ValidatedFacility

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = "YES"
    mock_client.invoke.return_value = mock_response
    return mock_client

@pytest.fixture
def sample_search_result():
    """Sample SearchResult for testing"""
    return SearchResult(
        url="https://example.com/toyota-kentucky",
        title="Toyota Kentucky Manufacturing Plant",
        content="The Georgetown assembly plant produces Camry and Avalon vehicles with 8000 employees"
    )

@pytest.fixture
def sample_facility_candidate():
    """Sample FacilityCandidate for testing"""
    return FacilityCandidate(
        company_name="Toyota Motor Corporation",
        facility_name="Toyota Kentucky Plant",
        facility_type="assembly plant",
        full_address="1001 Cherry Blossom Way, Georgetown, KY 40324, USA",
        country="United States",
        city="Georgetown",
        evidence_urls=["https://example.com/toyota-kentucky"],
        confidence_score=0.9
    )

@pytest.fixture
def sample_hq_candidate():
    """Sample HQ (should be rejected)"""
    return FacilityCandidate(
        company_name="Toyota Motor Corporation",
        facility_name="Toyota Global Headquarters",
        facility_type="corporate office",
        full_address="1 Toyota-Cho, Toyota City, Aichi, Japan",
        country="Japan",
        city="Toyota City",
        evidence_urls=["https://example.com/toyota-hq"],
        confidence_score=0.8
    )

@pytest.fixture
def sample_rd_candidate():
    """Sample R&D center (should be rejected)"""
    return FacilityCandidate(
        company_name="Toyota Motor Corporation",
        facility_name="Toyota Technical Center",
        facility_type="research and development center",
        full_address="York Township, Michigan, USA",
        country="United States",
        city="Ann Arbor",
        evidence_urls=["https://example.com/toyota-rd"],
        confidence_score=0.7
    )


# =============== Mock Fixtures for SearchNode ===============

@pytest.fixture
def mock_tavily_client():
    """Mock Tavily搜索API"""
    mock_instance = Mock()
    mock_instance.search.return_value = {
        'results': [
            {
                'url': 'https://example.com/factory',
                'title': 'Toyota Factory',
                'content': 'Manufacturing plant producing vehicles...',
                'score': 0.9
            },
            {
                'url': 'https://example.com/plant',
                'title': 'Toyota Assembly Plant',
                'content': 'Assembly plant with production facilities...',
                'score': 0.85
            }
        ]
    }
    yield mock_instance


@pytest.fixture
def mock_serpapi():
    """Mock SerpAPI Google搜索"""
    mock_instance = Mock()
    mock_instance.get_dict.return_value = {
        'organic_results': [
            {
                'link': 'https://shell.com/refinery',
                'title': 'Shell Deer Park Refinery',
                'snippet': 'Oil refining facility processing crude oil...'
            },
            {
                'link': 'https://shell.com/chemical',
                'title': 'Shell Chemical Plant',
                'snippet': 'Chemical manufacturing plant...'
            }
        ]
    }
    yield mock_instance


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for web scraping"""
    with patch('src.nodes.search.requests.get') as mock:
        mock_response = Mock()
        mock_response.text = '''
            <div class="result">
                <a class="result__a" href="https://example.com/factory1">Toyota Factory</a>
                <a class="result__snippet">Manufacturing site producing cars</a>
            </div>
            <div class="result">
                <a class="result__a" href="https://example.com/plant1">Toyota Plant</a>
                <a class="result__snippet">Assembly plant for vehicles</a>
            </div>
        '''
        mock_response.status_code = 200
        mock.return_value = mock_response
        yield mock


@pytest.fixture
def mock_cache():
    """Mock ResultCache"""
    with patch('src.utils.cache.ResultCache') as mock:
        mock_instance = Mock()
        mock_instance.get.return_value = None  # 默认缓存未命中
        mock_instance.set.return_value = None
        mock_instance.make_key = Mock(return_value="test_cache_key")
        yield mock_instance


# =============== Mock Fixtures for ExtractionNode ===============

@pytest.fixture
def mock_chatgpt():
    """Mock AzureChatOpenAI用于提取"""
    with patch('src.nodes.extraction.AzureChatOpenAI') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = '''```json
{
    "facilities": [
        {
            "company_name": "Toyota Motor Corporation",
            "facility_name": "Kentucky Plant",
            "facility_type": "assembly plant",
            "full_address": "1001 Cherry Blossom Way, Georgetown, KY 40324, USA",
            "country": "USA",
            "city": "Georgetown",
            "evidence_urls": ["https://toyota.com/kentucky"]
        }
    ]
}
```'''
        mock_response.response_metadata = {
            'token_usage': {
                'prompt_tokens': 1500,
                'completion_tokens': 300,
                'total_tokens': 1800
            }
        }
        mock_instance.invoke.return_value = mock_response
        yield mock_instance


@pytest.fixture
def mock_facility_classifier():
    """Mock FacilityTypeClassifier"""
    with patch('src.nodes.extraction.FacilityTypeClassifier.classify') as mock:
        mock.return_value = "Automotive Assembly Plant"
        yield mock


# =============== Mock Fixtures for DeduplicationNode ===============

@pytest.fixture
def mock_openai_embeddings():
    """Mock AzureOpenAI嵌入API"""
    with patch('src.nodes.deduplication.AzureOpenAI') as mock:
        mock_client = Mock()
        mock.return_value = mock_client

        # Mock embeddings response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5]),  # Facility 1
            Mock(embedding=[0.11, 0.21, 0.31, 0.41, 0.51]),  # Similar to 1 (>0.95)
            Mock(embedding=[0.9, 0.8, 0.7, 0.6, 0.5])   # Different (<0.95)
        ]
        mock_response.usage = Mock(total_tokens=150)
        mock_client.embeddings.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def sample_duplicate_facilities():
    """创建测试用重复设施"""
    facility1 = ValidatedFacility(
        company_name="Toyota Motor Corporation",
        facility_name="Kentucky Plant",
        facility_type="Automotive Assembly Plant",
        full_address="1001 Cherry Blossom Way, Georgetown, KY 40324",
        country="USA",
        city="Georgetown",
        latitude=38.2098,
        longitude=-84.5588,
        evidence_urls=["https://toyota.com/kentucky"],
        confidence_score=0.9,
        validation_passed=True,
        validation_reason="Passed keyword and LLM validation"
    )

    facility2 = ValidatedFacility(
        company_name="Toyota Motor Corporation",
        facility_name="Georgetown Assembly",  # Different name
        facility_type="Automotive Assembly Plant",
        full_address="1001 cherry blossom way, georgetown, ky 40324",  # Same address (different case)
        country="USA",
        city="Georgetown",
        latitude=38.2099,  # Slightly different (within 1km)
        longitude=-84.5589,
        evidence_urls=["https://wikipedia.org/toyota-kentucky"],  # Different evidence
        confidence_score=0.85,
        validation_passed=True,
        validation_reason="Passed keyword and LLM validation"
    )

    facility3 = ValidatedFacility(
        company_name="Shell",
        facility_name="Deer Park Refinery",
        facility_type="Oil Refinery",
        full_address="Deer Park, Texas, USA",
        country="USA",
        city="Deer Park",
        latitude=29.7052,
        longitude=-95.1238,
        evidence_urls=["https://shell.com/deerpark"],
        confidence_score=0.95,
        validation_passed=True,
        validation_reason="Passed keyword and LLM validation"
    )

    return [facility1, facility2, facility3]


@pytest.fixture
def sample_validated_facility():
    """单个验证通过的设施（用于测试）"""
    return ValidatedFacility(
        company_name="Shell",
        facility_name="Pernis Refinery",
        facility_type="Oil Refinery",
        full_address="Pernis, Rotterdam, Netherlands",
        country="Netherlands",
        city="Rotterdam",
        latitude=51.8897,
        longitude=4.3697,
        evidence_urls=["https://shell.com/pernis"],
        confidence_score=0.92,
        validation_passed=True,
        validation_reason="Passed keyword and LLM validation"
    )
