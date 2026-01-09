"""
Tests for SearchNode
搜索节点的单元测试，包括多种搜索方法（Tavily, SerpAPI, Web Scraping）
"""
import pytest
import os
import sys
import importlib.util
from unittest.mock import Mock, patch, MagicMock
from src.nodes.search import SearchNode
from src.models import SearchResult

# Check if optional dependencies are available
SERPAPI_AVAILABLE = importlib.util.find_spec('serpapi') is not None


class TestSearchNodeInitialization:
    """测试SearchNode初始化"""

    def test_default_initialization(self):
        """测试默认初始化（无环境变量）"""
        with patch.dict(os.environ, {}, clear=True):
            node = SearchNode()
            assert node.search_method == "web_scraping"  # 默认方法

    def test_initialization_with_tavily_env(self):
        """测试Tavily环境变量配置"""
        with patch.dict(os.environ, {
            "SEARCH_METHOD": "tavily",
            "TAVILY_API_KEY": "test_api_key"
        }):
            with patch('tavily.TavilyClient'):
                node = SearchNode()
                assert node.search_method == "tavily"

    def test_initialization_fallback_when_tavily_not_installed(self):
        """测试Tavily未安装时回退到web scraping"""
        with patch.dict(os.environ, {
            "SEARCH_METHOD": "tavily",
            "TAVILY_API_KEY": "test_key"
        }):
            # Mock the import to raise ImportError
            import sys
            with patch.dict(sys.modules, {'tavily': None}):
                node = SearchNode()
                assert node.search_method == "web_scraping"

    def test_cache_initialization(self):
        """测试缓存正确初始化"""
        node = SearchNode()
        assert node.cache is not None


class TestSearchNodeQueries:
    """测试查询构建逻辑"""

    def test_query_building_in_run(self, mock_cache):
        """测试run方法生成的查询格式"""
        with patch('src.nodes.search.ResultCache', return_value=mock_cache):
            node = SearchNode()
            node.search_method = "web_scraping"

            # Mock the web scraping method to capture queries
            queries_used = []

            def mock_web_scraping(query):
                queries_used.append(query)
                return []

            node._search_web_scraping = mock_web_scraping

            state = {
                "company": "Toyota Motor Corporation",
                "errors": [],
                "search_results": []
            }

            node.run(state)

            # Verify 3 queries were generated
            assert len(queries_used) == 3

            # Verify queries contain company name
            assert all("Toyota Motor Corporation" in q for q in queries_used)

            # Verify exclusion keywords in at least one query
            assert any("-headquarters" in q or "-office" in q or "-HQ" in q for q in queries_used)


class TestTavilySearch:
    """测试Tavily搜索方法"""

    def test_tavily_search_success(self, mock_tavily_client):
        """测试Tavily搜索成功返回结果"""
        node = SearchNode()
        node.tavily = mock_tavily_client

        results = node._search_tavily("Toyota factories")

        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/factory"
        assert results[0]["title"] == "Toyota Factory"
        assert "Manufacturing" in results[0]["content"]

    def test_tavily_search_empty_results(self):
        """测试Tavily返回空结果"""
        mock_client = Mock()
        mock_client.search.return_value = {"results": []}

        node = SearchNode()
        node.tavily = mock_client

        results = node._search_tavily("NonexistentCompany")
        assert results == []

    def test_tavily_search_handles_exception(self):
        """测试Tavily搜索异常处理"""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("API Error")

        node = SearchNode()
        node.tavily = mock_client

        results = node._search_tavily("Test query")
        assert results == []  # 返回空列表而不是抛出异常


@pytest.mark.skipif(not SERPAPI_AVAILABLE, reason="serpapi not installed")
class TestSerpAPISearch:
    """测试SerpAPI搜索方法"""

    def test_serpapi_search_success(self, mock_serpapi):
        """测试SerpAPI搜索成功"""
        # Patch GoogleSearch at the import point in _search_serpapi method
        with patch('src.nodes.search.GoogleSearch', return_value=mock_serpapi):
            node = SearchNode()
            node.serpapi_key = "test_key"

            results = node._search_serpapi("Shell refineries")

            assert len(results) == 2
            assert results[0]["url"] == "https://shell.com/refinery"
            assert results[0]["title"] == "Shell Deer Park Refinery"

    def test_serpapi_search_empty_results(self):
        """测试SerpAPI返回空结果"""
        mock_search = Mock()
        mock_search.get_dict.return_value = {"organic_results": []}

        with patch('src.nodes.search.GoogleSearch', return_value=mock_search):
            node = SearchNode()
            node.serpapi_key = "test_key"

            results = node._search_serpapi("NoResults")
            assert results == []

    def test_serpapi_search_handles_exception(self):
        """测试SerpAPI异常处理"""
        with patch('src.nodes.search.GoogleSearch', side_effect=Exception("API Error")):
            node = SearchNode()
            node.serpapi_key = "test_key"

            results = node._search_serpapi("Error query")
            assert results == []


class TestWebScraping:
    """测试网页抓取方法"""

    def test_web_scraping_success(self, mock_requests_get):
        """测试网页抓取成功"""
        node = SearchNode()
        results = node._search_web_scraping("Toyota factory")

        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/factory1"
        assert results[0]["title"] == "Toyota Factory"
        assert "Manufacturing" in results[0]["content"]

    def test_web_scraping_handles_request_exception(self):
        """测试网页抓取处理requests异常"""
        with patch('src.nodes.search.requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")

            node = SearchNode()
            results = node._search_web_scraping("Test query")

            assert results == []

    def test_web_scraping_malformed_html(self):
        """测试网页抓取处理畸形HTML"""
        with patch('src.nodes.search.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.text = "<html><body>Invalid HTML</body></html>"
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            node = SearchNode()
            results = node._search_web_scraping("Query")

            # 应该返回空列表而不是崩溃
            assert results == []


class TestSearchNodeRun:
    """测试主运行方法"""

    def test_run_with_no_company_name(self):
        """测试无公司名称时早期返回"""
        node = SearchNode()
        state = {"company": "", "errors": []}

        result = node.run(state)

        assert "No company name provided" in result["errors"][0]

    def test_run_cache_hit(self, mock_cache):
        """测试缓存命中时直接返回"""
        cached_data = [{
            "url": "https://cached.com/result",
            "title": "Cached Result",
            "content": "Cached content"
        }]

        mock_cache.get.return_value = cached_data

        with patch('src.nodes.search.ResultCache', return_value=mock_cache):
            node = SearchNode()
            state = {
                "company": "Toyota",
                "errors": [],
                "search_results": []
            }

            result = node.run(state)

            # 验证返回了缓存结果
            assert len(result["search_results"]) == 1
            assert result["search_results"][0].url == "https://cached.com/result"

            # 验证没有执行新搜索（cache.set未被调用）
            mock_cache.set.assert_not_called()

    def test_run_cache_miss_executes_search(self, mock_cache):
        """测试缓存未命中时执行搜索"""
        mock_cache.get.return_value = None  # 缓存未命中

        with patch('src.nodes.search.ResultCache', return_value=mock_cache):
            node = SearchNode()
            node.search_method = "web_scraping"

            # Mock搜索方法返回结果
            node._search_web_scraping = Mock(return_value=[{
                "url": "https://new.com/result",
                "title": "New Result",
                "content": "New content"
            }])

            state = {
                "company": "Shell",
                "errors": [],
                "search_results": []
            }

            result = node.run(state)

            # 验证执行了搜索
            assert len(result["search_results"]) == 1

            # 验证结果被缓存
            mock_cache.set.assert_called_once()

    def test_run_url_deduplication(self, mock_cache):
        """测试URL去重逻辑"""
        mock_cache.get.return_value = None

        with patch('src.nodes.search.ResultCache', return_value=mock_cache):
            node = SearchNode()

            # Mock返回重复URL的结果
            node._search_web_scraping = Mock(return_value=[
                {"url": "https://same.com", "title": "Result 1", "content": "Content 1"},
                {"url": "https://same.com", "title": "Result 2", "content": "Content 2"},
                {"url": "https://different.com", "title": "Result 3", "content": "Content 3"}
            ])

            state = {
                "company": "Toyota",
                "errors": [],
                "search_results": []
            }

            result = node.run(state)

            # 验证只有2个唯一URL（重复的被移除）
            assert len(result["search_results"]) == 2
            urls = [r.url for r in result["search_results"]]
            assert "https://same.com" in urls
            assert "https://different.com" in urls

    def test_run_handles_search_result_validation_errors(self, mock_cache):
        """测试SearchResult验证失败时的处理"""
        mock_cache.get.return_value = None

        with patch('src.nodes.search.ResultCache', return_value=mock_cache):
            node = SearchNode()

            # Mock返回包含缺少必需字段的搜索结果（将抛出ValidationError）
            node._search_web_scraping = Mock(return_value=[
                {"url": "https://valid.com", "title": "Valid", "content": "Content"},
                {"title": "Missing URL"},  # 缺少必需字段url
            ])

            state = {
                "company": "Toyota",
                "errors": [],
                "search_results": []
            }

            result = node.run(state)

            # 验证只有有效结果被包含（缺少字段的被跳过）
            assert len(result["search_results"]) == 1
            assert result["search_results"][0].url == "https://valid.com"

    def test_run_handles_individual_query_exceptions(self, mock_cache):
        """测试单个查询失败时继续其他查询"""
        mock_cache.get.return_value = None

        with patch('src.nodes.search.ResultCache', return_value=mock_cache):
            node = SearchNode()

            # Mock第一个查询失败，后续成功
            call_count = [0]

            def mock_search(query):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise Exception("First query failed")
                return [{"url": f"https://result{call_count[0]}.com", "title": f"Result {call_count[0]}", "content": "Content"}]

            node._search_web_scraping = mock_search

            state = {
                "company": "Toyota",
                "errors": [],
                "search_results": []
            }

            result = node.run(state)

            # 验证有错误记录
            assert len(result["errors"]) == 1
            assert "First query failed" in result["errors"][0]

            # 验证其他查询继续执行并返回结果
            assert len(result["search_results"]) > 0


class TestSearchNodeIntegration:
    """集成测试"""

    def test_full_search_flow_with_tavily(self, mock_tavily_client, mock_cache):
        """测试完整Tavily搜索流程"""
        mock_cache.get.return_value = None

        with patch('src.nodes.search.ResultCache', return_value=mock_cache):
            with patch.dict(os.environ, {"SEARCH_METHOD": "tavily", "TAVILY_API_KEY": "test_key"}):
                node = SearchNode()
                # Manually set the tavily client (since we can't mock the import easily)
                node.search_method = "tavily"
                node.tavily = mock_tavily_client

                state = {
                    "company": "Toyota Motor Corporation",
                    "errors": [],
                    "search_results": []
                }

                result = node.run(state)

                # 验证搜索结果
                assert len(result["search_results"]) > 0
                assert all(isinstance(r, SearchResult) for r in result["search_results"])

                # 验证结果被缓存
                assert mock_cache.set.called

    def test_full_search_flow_with_web_scraping(self, mock_requests_get, mock_cache):
        """测试完整web scraping流程"""
        mock_cache.get.return_value = None

        with patch('src.nodes.search.ResultCache', return_value=mock_cache):
            node = SearchNode()

            state = {
                "company": "Shell",
                "errors": [],
                "search_results": []
            }

            result = node.run(state)

            # 验证搜索执行
            assert "search_results" in result

            # 验证状态更新
            assert isinstance(result, dict)
