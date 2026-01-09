"""
Tests for DeduplicationNode
去重节点的单元测试，包括地址、Embedding和坐标三种去重策略
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.nodes.deduplication import DeduplicationNode
from src.models import ValidatedFacility


class TestDeduplicationNodeInitialization:
    """测试DeduplicationNode初始化"""

    def test_default_initialization(self):
        """测试默认初始化"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            assert node.embedding_model == "text-embedding-3-small"
            assert node.similarity_threshold == 0.95

    def test_initialization_with_metrics_collector(self):
        """测试带MetricsCollector的初始化"""
        mock_metrics = Mock()
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode(metrics_collector=mock_metrics)
            assert node.metrics == mock_metrics


class TestDeduplicateByAddress:
    """测试地址去重"""

    def test_exact_address_match_merges(self):
        """测试完全匹配的地址被合并"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            facility1 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Kentucky Plant",
                facility_type="Assembly Plant",
                full_address="1001 Cherry Blossom Way, Georgetown, KY",
                country="USA",
                city="Georgetown",
                evidence_urls=["https://toyota.com/1"],
                confidence_score=0.9,
                validation_passed=True,
                validation_reason="Passed"
            )

            facility2 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Georgetown Assembly",
                facility_type="Assembly Plant",
                full_address="1001 cherry blossom way, georgetown, ky",  # 不同大小写
                country="USA",
                city="Georgetown",
                evidence_urls=["https://toyota.com/2"],
                confidence_score=0.85,
                validation_passed=True,
                validation_reason="Passed"
            )

            result = node._deduplicate_by_address([facility1, facility2])

            # 验证合并为1个设施
            assert len(result) == 1
            assert len(result[0].evidence_urls) == 2  # 合并了evidence_urls
            assert result[0].confidence_score == 0.9  # 取最大值

    def test_different_addresses_not_merged(self):
        """测试不同地址不会被合并"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            facility1 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Kentucky Plant",
                facility_type="Assembly Plant",
                full_address="Georgetown, KY",
                country="USA",
                city="Georgetown",
                evidence_urls=["https://toyota.com/1"],
                confidence_score=0.9,
                validation_passed=True,
                validation_reason="Passed"
            )

            facility2 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Texas Plant",
                facility_type="Assembly Plant",
                full_address="San Antonio, TX",
                country="USA",
                city="San Antonio",
                evidence_urls=["https://toyota.com/2"],
                confidence_score=0.85,
                validation_passed=True,
                validation_reason="Passed"
            )

            result = node._deduplicate_by_address([facility1, facility2])

            # 验证未合并
            assert len(result) == 2

    def test_multiple_duplicates_merged(self):
        """测试多重复地址的处理"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            facility1 = ValidatedFacility(
                company_name="Shell",
                facility_name="Refinery 1",
                facility_type="Oil Refinery",
                full_address="Deer Park, TX",
                country="USA",
                city="Deer Park",
                evidence_urls=["https://shell.com/1"],
                confidence_score=0.9,
                validation_passed=True,
                validation_reason="Passed"
            )

            facility2 = ValidatedFacility(
                company_name="Shell",
                facility_name="Refinery 2",
                facility_type="Oil Refinery",
                full_address="deer park, tx",
                country="USA",
                city="Deer Park",
                evidence_urls=["https://shell.com/2"],
                confidence_score=0.85,
                validation_passed=True,
                validation_reason="Passed"
            )

            facility3 = ValidatedFacility(
                company_name="Shell",
                facility_name="Refinery 3",
                facility_type="Oil Refinery",
                full_address="DEER PARK, TX",
                country="USA",
                city="Deer Park",
                evidence_urls=["https://shell.com/3"],
                confidence_score=0.95,
                validation_passed=True,
                validation_reason="Passed"
            )

            result = node._deduplicate_by_address([facility1, facility2, facility3])

            # 验证全部合并为1个
            assert len(result) == 1
            assert len(result[0].evidence_urls) == 3
            assert result[0].confidence_score == 0.95


class TestDeduplicateByEmbedding:
    """测试Embedding去重"""

    def test_high_similarity_detected_as_duplicate(self, mock_openai_embeddings):
        """测试高相似度被识别为重复"""
        with patch('src.nodes.deduplication.AzureOpenAI', return_value=mock_openai_embeddings):
            node = DeduplicationNode()
            node.client = mock_openai_embeddings

            # 创建2个相似的设施
            facility1 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Kentucky Plant",
                facility_type="Assembly Plant",
                full_address="Georgetown, KY",
                country="USA",
                city="Georgetown",
                evidence_urls=["https://toyota.com/1"],
                confidence_score=0.9,
                validation_passed=True,
                validation_reason="Passed"
            )

            facility2 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Georgetown Assembly",  # 相似的名称
                facility_type="Assembly Plant",
                full_address="Georgetown, Kentucky",  # 相似的地址
                country="USA",
                city="Georgetown",
                evidence_urls=["https://toyota.com/2"],
                confidence_score=0.85,
                validation_passed=True,
                validation_reason="Passed"
            )

            result = node._deduplicate_by_embedding([facility1, facility2])

            # 验证相似的被合并（mock embeddings返回[0.1,0.2,0.3]和[0.11,0.21,0.31]，相似度>0.95）
            assert len(result) == 1  # 被合并为1个
            assert len(result[0].evidence_urls) == 2  # 合并了evidence_urls

    def test_embedding_api_called_correctly(self, mock_openai_embeddings):
        """测试Embedding API调用正确"""
        with patch('src.nodes.deduplication.AzureOpenAI', return_value=mock_openai_embeddings):
            node = DeduplicationNode()
            node.client = mock_openai_embeddings

            facilities = [
                ValidatedFacility(
                    company_name="Toyota",
                    facility_name="Plant 1",
                    facility_type="Assembly Plant",
                    full_address="Address 1",
                    country="USA",
                    city="City 1",
                    evidence_urls=["https://toyota.com/1"],
                    confidence_score=0.9,
                    validation_passed=True,
                    validation_reason="Passed"
                ),
                ValidatedFacility(
                    company_name="Toyota",
                    facility_name="Plant 2",
                    facility_type="Assembly Plant",
                    full_address="Address 2",
                    country="USA",
                    city="City 2",
                    evidence_urls=["https://toyota.com/2"],
                    confidence_score=0.85,
                    validation_passed=True,
                    validation_reason="Passed"
                )
            ]

            node._deduplicate_by_embedding(facilities)

            # 验证embeddings API被调用
            mock_openai_embeddings.embeddings.create.assert_called_once()

    def test_embedding_exception_returns_original_list(self):
        """测试Embedding异常时返回原列表"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")

        with patch('src.nodes.deduplication.AzureOpenAI', return_value=mock_client):
            node = DeduplicationNode()
            node.client = mock_client

            facilities = [
                ValidatedFacility(
                    company_name="Toyota",
                    facility_name="Plant 1",
                    facility_type="Assembly Plant",
                    full_address="Address 1",
                    country="USA",
                    city="City 1",
                    evidence_urls=["https://toyota.com"],
                    confidence_score=0.9,
                    validation_passed=True,
                    validation_reason="Passed"
                )
            ]

            result = node._deduplicate_by_embedding(facilities)

            # 验证返回原列表
            assert result == facilities


class TestDeduplicateByCoordinates:
    """测试坐标去重"""

    def test_close_coordinates_detected_as_duplicate(self):
        """测试近距离坐标被识别为重复"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            facility1 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Plant A",
                facility_type="Assembly Plant",
                full_address="Georgetown, KY",
                country="USA",
                city="Georgetown",
                latitude=38.2098,
                longitude=-84.5588,
                evidence_urls=["https://toyota.com/1"],
                confidence_score=0.9,
                validation_passed=True,
                validation_reason="Passed"
            )

            facility2 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Plant B",
                facility_type="Assembly Plant",
                full_address="Georgetown, KY",
                country="USA",
                city="Georgetown",
                latitude=38.2099,  # 非常接近（<1km）
                longitude=-84.5589,
                evidence_urls=["https://toyota.com/2"],
                confidence_score=0.85,
                validation_passed=True,
                validation_reason="Passed"
            )

            result = node._deduplicate_by_coordinates([facility1, facility2])

            # 验证被合并
            assert len(result) == 1

    def test_far_coordinates_not_merged(self):
        """测试远距离坐标不会合并"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            facility1 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Kentucky Plant",
                facility_type="Assembly Plant",
                full_address="Georgetown, KY",
                country="USA",
                city="Georgetown",
                latitude=38.2098,
                longitude=-84.5588,
                evidence_urls=["https://toyota.com/1"],
                confidence_score=0.9,
                validation_passed=True,
                validation_reason="Passed"
            )

            facility2 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Texas Plant",
                facility_type="Assembly Plant",
                full_address="San Antonio, TX",
                country="USA",
                city="San Antonio",
                latitude=29.4241,  # 非常远
                longitude=-98.4936,
                evidence_urls=["https://toyota.com/2"],
                confidence_score=0.85,
                validation_passed=True,
                validation_reason="Passed"
            )

            result = node._deduplicate_by_coordinates([facility1, facility2])

            # 验证未合并
            assert len(result) == 2

    def test_skips_facilities_without_coordinates(self):
        """测试跳过没有坐标的设施"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            facility1 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Plant with coords",
                facility_type="Assembly Plant",
                full_address="Georgetown, KY",
                country="USA",
                city="Georgetown",
                latitude=38.2098,
                longitude=-84.5588,
                evidence_urls=["https://toyota.com/1"],
                confidence_score=0.9,
                validation_passed=True,
                validation_reason="Passed"
            )

            facility2 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Plant without coords",
                facility_type="Assembly Plant",
                full_address="Unknown Address",
                country="USA",
                city="Unknown",
                latitude=None,  # 无坐标
                longitude=None,
                evidence_urls=["https://toyota.com/2"],
                confidence_score=0.85,
                validation_passed=True,
                validation_reason="Passed"
            )

            result = node._deduplicate_by_coordinates([facility1, facility2])

            # 验证都保留（无坐标的不参与去重）
            assert len(result) == 2


class TestHaversineDistance:
    """测试Haversine距离计算"""

    def test_same_location_distance_zero(self):
        """测试相同位置距离为0"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            distance = node._haversine_distance(38.2098, -84.5588, 38.2098, -84.5588)

            assert distance == pytest.approx(0.0, abs=0.01)

    def test_known_distance_calculation(self):
        """测试已知距离的计算"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            # 赤道上1度经度约111km
            distance = node._haversine_distance(0, 0, 0, 1)

            assert distance == pytest.approx(111.0, rel=0.01)  # 约111km

    def test_diagonal_distance(self):
        """测试对角线距离"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            # Georgetown, KY to Louisville, KY 约110km
            distance = node._haversine_distance(38.2098, -84.5588, 38.2527, -85.7585)

            assert distance > 100  # 应该大于100km
            assert distance < 150  # 应该小于150km


class TestMergeFacilities:
    """测试设施合并逻辑"""

    def test_merge_preserves_facility1_info(self):
        """测试合并保留facility1的基本信息"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            facility1 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Kentucky Plant",
                facility_type="Assembly Plant",
                full_address="Georgetown, KY",
                country="USA",
                city="Georgetown",
                evidence_urls=["https://toyota.com/1"],
                confidence_score=0.9,
                validation_passed=True,
                validation_reason="Passed"
            )

            facility2 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Different Name",
                facility_type="Different Type",
                full_address="Different Address",
                country="USA",
                city="Different City",
                evidence_urls=["https://toyota.com/2"],
                confidence_score=0.85,
                validation_passed=True,
                validation_reason="Passed"
            )

            merged = node._merge_facilities(facility1, facility2)

            # 验证保留facility1的信息
            assert merged.facility_name == "Kentucky Plant"
            assert merged.facility_type == "Assembly Plant"
            assert merged.full_address == "Georgetown, KY"

    def test_merge_combines_evidence_urls(self):
        """测试合并合并evidence_urls"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            facility1 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Plant",
                facility_type="Assembly Plant",
                full_address="Address",
                country="USA",
                city="City",
                evidence_urls=["https://toyota.com/1", "https://toyota.com/shared"],
                confidence_score=0.9,
                validation_passed=True,
                validation_reason="Passed"
            )

            facility2 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Plant",
                facility_type="Assembly Plant",
                full_address="Address",
                country="USA",
                city="City",
                evidence_urls=["https://toyota.com/2", "https://toyota.com/shared"],
                confidence_score=0.85,
                validation_passed=True,
                validation_reason="Passed"
            )

            merged = node._merge_facilities(facility1, facility2)

            # 验证evidence_urls合并且去重
            assert len(merged.evidence_urls) == 3
            assert set(merged.evidence_urls) == {
                "https://toyota.com/1",
                "https://toyota.com/2",
                "https://toyota.com/shared"
            }

    def test_merge_takes_max_confidence(self):
        """测试合并取最大confidence_score"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            facility1 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Plant",
                facility_type="Assembly Plant",
                full_address="Address",
                country="USA",
                city="City",
                evidence_urls=["https://toyota.com/1"],
                confidence_score=0.85,
                validation_passed=True,
                validation_reason="Passed"
            )

            facility2 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Plant",
                facility_type="Assembly Plant",
                full_address="Address",
                country="USA",
                city="City",
                evidence_urls=["https://toyota.com/2"],
                confidence_score=0.95,
                validation_passed=True,
                validation_reason="Passed"
            )

            merged = node._merge_facilities(facility1, facility2)

            # 验证取最大值
            assert merged.confidence_score == 0.95

    def test_merge_coordinates_fallback(self):
        """测试坐标回退逻辑"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            facility1 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Plant",
                facility_type="Assembly Plant",
                full_address="Address",
                country="USA",
                city="City",
                latitude=None,  # facility1缺少坐标
                longitude=None,
                evidence_urls=["https://toyota.com/1"],
                confidence_score=0.9,
                validation_passed=True,
                validation_reason="Passed"
            )

            facility2 = ValidatedFacility(
                company_name="Toyota",
                facility_name="Plant",
                facility_type="Assembly Plant",
                full_address="Address",
                country="USA",
                city="City",
                latitude=38.2098,  # facility2有坐标
                longitude=-84.5588,
                evidence_urls=["https://toyota.com/2"],
                confidence_score=0.85,
                validation_passed=True,
                validation_reason="Passed"
            )

            merged = node._merge_facilities(facility1, facility2)

            # 验证使用facility2的坐标
            assert merged.latitude == 38.2098
            assert merged.longitude == -84.5588


class TestComputeSimilarityMatrix:
    """测试相似度矩阵计算"""

    def test_similarity_matrix_symmetric(self):
        """测试相似度矩阵是对称的"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            embeddings = np.array([
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9]
            ])

            matrix = node._compute_similarity_matrix(embeddings)

            # 验证对称
            assert np.allclose(matrix, matrix.T)

    def test_similarity_matrix_diagonal_is_one(self):
        """测试相似度矩阵对角线为1"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            embeddings = np.array([
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ])

            matrix = node._compute_similarity_matrix(embeddings)

            # 验证对角线为1（自相似度）
            assert matrix[0, 0] == pytest.approx(1.0, abs=0.001)
            assert matrix[1, 1] == pytest.approx(1.0, abs=0.001)

    def test_similarity_matrix_range(self):
        """测试相似度矩阵值范围[0, 1]"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            embeddings = np.array([
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9]
            ])

            matrix = node._compute_similarity_matrix(embeddings)

            # 验证范围
            assert np.all(matrix >= 0)
            assert np.all(matrix <= 1)


class TestDeduplicationNodeRun:
    """测试主运行方法"""

    def test_run_with_zero_facilities(self):
        """测试0个设施时早期返回"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            state = {
                "validated_facilities": []
            }

            result = node.run(state)

            # 验证早期返回
            assert result["validated_facilities"] == []

    def test_run_with_one_facility(self):
        """测试1个设施时早期返回"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            facility = ValidatedFacility(
                company_name="Toyota",
                facility_name="Plant",
                facility_type="Assembly Plant",
                full_address="Address",
                country="USA",
                city="City",
                evidence_urls=["https://toyota.com"],
                confidence_score=0.9,
                validation_passed=True,
                validation_reason="Passed"
            )

            state = {
                "validated_facilities": [facility]
            }

            result = node.run(state)

            # 验证早期返回
            assert len(result["validated_facilities"]) == 1

    def test_run_executes_three_phases(self, sample_duplicate_facilities):
        """测试运行执行三个阶段的去重"""
        with patch('src.nodes.deduplication.AzureOpenAI'):
            node = DeduplicationNode()

            state = {
                "validated_facilities": sample_duplicate_facilities
            }

            # Mock embedding API to avoid actual calls
            with patch.object(node, '_deduplicate_by_embedding', return_value=sample_duplicate_facilities):
                result = node.run(state)

                # 验证状态更新
                assert "validated_facilities" in result
                assert isinstance(result["validated_facilities"], list)
