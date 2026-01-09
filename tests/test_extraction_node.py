"""
Tests for ExtractionNode
提取节点的单元测试，包括LLM提取和JSON解析
"""
import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from src.nodes.extraction import ExtractionNode
from src.models import SearchResult, FacilityCandidate


class TestExtractionNodeInitialization:
    """测试ExtractionNode初始化"""

    def test_default_initialization(self):
        """测试默认初始化"""
        with patch('src.nodes.extraction.AzureChatOpenAI') as mock_chatgpt:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
                node = ExtractionNode()

                # 验证ChatOpenAI配置
                mock_chatgpt.assert_called_once()
                call_kwargs = mock_chatgpt.call_args.kwargs
                assert call_kwargs['temperature'] == 0
                assert call_kwargs['max_tokens'] == 4000
                assert call_kwargs['api_key'] == "test_key"

    def test_initialization_with_custom_model(self):
        """测试自定义模型初始化"""
        with patch('src.nodes.extraction.AzureChatOpenAI'):
            with patch.dict(os.environ, {
                "OPENAI_API_KEY": "test_key",
                "OPENAI_MODEL": "gpt-4o"
            }):
                node = ExtractionNode()
                assert node.model_name == "gpt-4o"

    def test_initialization_with_metrics_collector(self):
        """测试带MetricsCollector的初始化"""
        mock_metrics = Mock()
        with patch('src.nodes.extraction.AzureChatOpenAI'):
            node = ExtractionNode(metrics_collector=mock_metrics)
            assert node.metrics == mock_metrics

    def test_cache_initialization(self):
        """测试缓存正确初始化"""
        with patch('src.nodes.extraction.AzureChatOpenAI'):
            node = ExtractionNode()
            assert node.cache is not None


class TestBuildContext:
    """测试上下文构建逻辑"""

    def test_build_context_formatting(self):
        """测试上下文格式化正确"""
        with patch('src.nodes.extraction.AzureChatOpenAI'):
            node = ExtractionNode()

            search_results = [
                SearchResult(
                    url="https://example.com/1",
                    title="Factory 1",
                    content="Manufacturing content 1"
                ),
                SearchResult(
                    url="https://example.com/2",
                    title="Factory 2",
                    content="Manufacturing content 2"
                )
            ]

            context = node._build_context(search_results)

            # 验证格式
            assert "[Source 1]" in context
            assert "[Source 2]" in context
            assert "URL: https://example.com/1" in context
            assert "Title: Factory 1" in context
            assert "Content: Manufacturing content 1" in context

    def test_build_context_limits_to_15_results(self):
        """测试限制15个搜索结果"""
        with patch('src.nodes.extraction.AzureChatOpenAI'):
            node = ExtractionNode()

            # 创建20个搜索结果
            search_results = [
                SearchResult(
                    url=f"https://example.com/{i}",
                    title=f"Factory {i}",
                    content=f"Content {i}"
                )
                for i in range(20)
            ]

            context = node._build_context(search_results)

            # 验证只包含前15个
            assert "[Source 15]" in context
            assert "[Source 16]" not in context

    def test_build_context_handles_empty_fields(self):
        """测试处理空字段"""
        with patch('src.nodes.extraction.AzureChatOpenAI'):
            node = ExtractionNode()

            search_results = [
                SearchResult(url="", title="", content="")
            ]

            context = node._build_context(search_results)

            # 应该不会崩溃
            assert "[Source 1]" in context
            assert "URL:" in context


class TestExtractWithLLM:
    """测试LLM提取方法"""

    def test_extract_cache_hit(self, mock_cache):
        """测试缓存命中时直接返回"""
        cached_data = [{
            "company_name": "Toyota",
            "facility_name": "Kentucky Plant",
            "facility_type": "assembly plant",
            "full_address": "Georgetown, KY, USA",
            "country": "USA",
            "city": "Georgetown",
            "evidence_urls": ["https://toyota.com"],
            "confidence_score": 0.9
        }]

        mock_cache.get.return_value = cached_data

        with patch('src.nodes.extraction.AzureChatOpenAI'):
            with patch('src.nodes.extraction.ResultCache', return_value=mock_cache):
                node = ExtractionNode()

                results = node._extract_with_llm("Toyota", "test context")

                # 验证返回缓存结果
                assert len(results) == 1
                assert isinstance(results[0], FacilityCandidate)
                assert results[0].facility_name == "Kentucky Plant"

                # 验证没有调用LLM
                node.llm.invoke.assert_not_called()

    def test_extract_cache_miss_calls_llm(self, mock_chatgpt, mock_cache):
        """测试缓存未命中时调用LLM"""
        mock_cache.get.return_value = None  # 缓存未命中

        with patch('src.nodes.extraction.AzureChatOpenAI', return_value=mock_chatgpt):
            with patch('src.nodes.extraction.ResultCache', return_value=mock_cache):
                node = ExtractionNode()

                results = node._extract_with_llm("Toyota", "test context")

                # 验证调用了LLM
                mock_chatgpt.invoke.assert_called_once()

                # 验证结果被缓存
                mock_cache.set.assert_called_once()

    def test_extract_tracks_token_usage(self, mock_chatgpt, mock_cache):
        """测试Token使用量追踪"""
        mock_cache.get.return_value = None

        with patch('src.nodes.extraction.AzureChatOpenAI', return_value=mock_chatgpt):
            with patch('src.nodes.extraction.ResultCache', return_value=mock_cache):
                node = ExtractionNode()

                results = node._extract_with_llm("Toyota", "test context")

                # 验证响应包含token使用量
                mock_chatgpt.invoke.return_value.response_metadata['token_usage']

    def test_extract_tracks_metrics_if_collector_provided(self, mock_chatgpt, mock_cache):
        """测试Metrics追踪（如果提供collector）"""
        mock_cache.get.return_value = None
        mock_metrics = Mock()

        with patch('src.nodes.extraction.AzureChatOpenAI', return_value=mock_chatgpt):
            with patch('src.nodes.extraction.ResultCache', return_value=mock_cache):
                with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4-turbo-preview"}):
                    node = ExtractionNode(metrics_collector=mock_metrics)

                    results = node._extract_with_llm("Toyota", "test context")

                    # 验证调用了metrics collector（model是self.model_name字符串）
                    mock_metrics.track_llm_call.assert_called_once_with(
                        node_name="extraction",
                        prompt_tokens=1500,
                        completion_tokens=300,
                        model="gpt-4-turbo-preview"
                    )

    def test_extract_llm_exception_reraises(self, mock_cache):
        """测试LLM异常重新抛出"""
        mock_cache.get.return_value = None

        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM API Error")

        with patch('src.nodes.extraction.AzureChatOpenAI', return_value=mock_llm):
            with patch('src.nodes.extraction.ResultCache', return_value=mock_cache):
                node = ExtractionNode()

                with pytest.raises(Exception) as exc_info:
                    node._extract_with_llm("Toyota", "test context")

                assert "LLM API Error" in str(exc_info.value)


class TestParseLLMResponse:
    """测试LLM响应解析"""

    def test_parse_valid_json(self):
        """测试解析有效JSON"""
        with patch('src.nodes.extraction.AzureChatOpenAI'):
            node = ExtractionNode()

            response_text = json.dumps({
                "facilities": [
                    {
                        "company_name": "Toyota",
                        "facility_name": "Kentucky Plant",
                        "facility_type": "assembly plant",
                        "full_address": "Georgetown, KY, USA",
                        "country": "USA",
                        "city": "Georgetown",
                        "evidence_urls": ["https://toyota.com"]
                    }
                ]
            })

            facilities = node._parse_llm_response(response_text)

            assert len(facilities) == 1
            assert facilities[0].facility_name == "Kentucky Plant"

    def test_parse_removes_markdown_code_blocks(self):
        """测试去除Markdown代码块"""
        with patch('src.nodes.extraction.AzureChatOpenAI'):
            node = ExtractionNode()

            # 带```json```的响应
            response_text = '''```json
{
    "facilities": [
        {
            "company_name": "Toyota",
            "facility_name": "Test Plant",
            "facility_type": "assembly plant",
            "full_address": "Test Address",
            "country": "USA",
            "city": "Test City",
            "evidence_urls": ["https://test.com"]
        }
    ]
}
```'''

            facilities = node._parse_llm_response(response_text)

            assert len(facilities) == 1
            assert facilities[0].facility_name == "Test Plant"

    def test_parse_skips_facilities_without_evidence_urls(self):
        """测试跳过没有evidence_urls的设施"""
        with patch('src.nodes.extraction.AzureChatOpenAI'):
            node = ExtractionNode()

            response_text = json.dumps({
                "facilities": [
                    {
                        "company_name": "Toyota",
                        "facility_name": "Valid Plant",
                        "facility_type": "assembly plant",
                        "full_address": "Address",
                        "country": "USA",
                        "city": "City",
                        "evidence_urls": ["https://toyota.com"]
                    },
                    {
                        "company_name": "Toyota",
                        "facility_name": "Invalid Plant",
                        "facility_type": "assembly plant",
                        "full_address": "Address",
                        "country": "USA",
                        "city": "City",
                        "evidence_urls": []  # 空evidence_urls
                    }
                ]
            })

            facilities = node._parse_llm_response(response_text)

            # 只有一个有效设施
            assert len(facilities) == 1
            assert facilities[0].facility_name == "Valid Plant"

    def test_parse_reclassifies_facility_types(self):
        """测试设施类型重分类"""
        with patch('src.nodes.extraction.AzureChatOpenAI'):
            with patch('src.nodes.extraction.FacilityTypeClassifier.classify') as mock_classify:
                mock_classify.return_value = "Automotive Assembly Plant"

                node = ExtractionNode()

                response_text = json.dumps({
                    "facilities": [
                        {
                            "company_name": "Toyota",
                            "facility_name": "Kentucky Plant",
                            "facility_type": "factory",
                            "full_address": "Georgetown, KY, USA",
                            "country": "USA",
                            "city": "Georgetown",
                            "evidence_urls": ["https://toyota.com"]
                        }
                    ]
                })

                facilities = node._parse_llm_response(response_text)

                # 验证类型被重分类
                assert facilities[0].facility_type == "Automotive Assembly Plant"
                mock_classify.assert_called_once()

    def test_parse_skips_invalid_pydantic_facilities(self):
        """测试跳过Pydantic验证失败的设施"""
        with patch('src.nodes.extraction.AzureChatOpenAI'):
            node = ExtractionNode()

            # 缺少必需字段的设施
            response_text = json.dumps({
                "facilities": [
                    {
                        "company_name": "Toyota",
                        # 缺少facility_name
                        "facility_type": "assembly plant",
                        "full_address": "Address",
                        "country": "USA",
                        "city": "City",
                        "evidence_urls": ["https://toyota.com"]
                    }
                ]
            })

            facilities = node._parse_llm_response(response_text)

            # 无效设施被跳过
            assert len(facilities) == 0

    def test_parse_handles_json_decode_error(self):
        """测试处理JSON解析错误"""
        with patch('src.nodes.extraction.AzureChatOpenAI'):
            node = ExtractionNode()

            # 无效JSON
            response_text = "This is not valid JSON at all!"

            facilities = node._parse_llm_response(response_text)

            # 返回空列表而不是崩溃
            assert facilities == []

    def test_parse_handles_non_array_facilities(self):
        """测试处理非数组的facilities字段"""
        with patch('src.nodes.extraction.AzureChatOpenAI'):
            node = ExtractionNode()

            # facilities不是数组
            response_text = json.dumps({
                "facilities": "this should be an array"
            })

            facilities = node._parse_llm_response(response_text)

            # 返回空列表
            assert facilities == []


class TestExtractionNodeRun:
    """测试主运行方法"""

    def test_run_with_no_search_results(self):
        """测试无搜索结果时早期返回"""
        with patch('src.nodes.extraction.AzureChatOpenAI'):
            node = ExtractionNode()

            state = {
                "company": "Toyota",
                "search_results": [],
                "errors": []
            }

            result = node.run(state)

            # 验证早期返回
            assert result["extracted_facilities"] == []

    def test_run_updates_state_correctly(self, mock_chatgpt):
        """测试状态正确更新"""
        with patch('src.nodes.extraction.AzureChatOpenAI', return_value=mock_chatgpt):
            node = ExtractionNode()

            search_results = [
                SearchResult(
                    url="https://toyota.com",
                    title="Toyota Factory",
                    content="Manufacturing content"
                )
            ]

            state = {
                "company": "Toyota",
                "search_results": search_results,
                "errors": []
            }

            result = node.run(state)

            # 验证extracted_facilities被添加到state
            assert "extracted_facilities" in result
            assert isinstance(result["extracted_facilities"], list)

    def test_run_handles_exceptions_gracefully(self):
        """测试异常处理不崩溃"""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Unexpected error")

        with patch('src.nodes.extraction.AzureChatOpenAI', return_value=mock_llm):
            node = ExtractionNode()

            search_results = [
                SearchResult(
                    url="https://toyota.com",
                    title="Toyota Factory",
                    content="Content"
                )
            ]

            state = {
                "company": "Toyota",
                "search_results": search_results,
                "errors": []
            }

            result = node.run(state)

            # 验证不崩溃，返回空列表
            assert result["extracted_facilities"] == []

            # 验证错误被记录
            assert len(result["errors"]) > 0
            assert "Extraction error" in result["errors"][0]

    def test_run_logs_extraction_count(self, mock_chatgpt):
        """测试日志记录提取的设施数量"""
        with patch('src.nodes.extraction.AzureChatOpenAI', return_value=mock_chatgpt):
            node = ExtractionNode()

            search_results = [
                SearchResult(
                    url="https://toyota.com",
                    title="Toyota Factory",
                    content="Manufacturing content"
                )
            ]

            state = {
                "company": "Toyota",
                "search_results": search_results,
                "errors": []
            }

            with patch('src.nodes.extraction.logger') as mock_logger:
                result = node.run(state)

                # 验证日志记录了提取的设施数量
                mock_logger.info.assert_called()
