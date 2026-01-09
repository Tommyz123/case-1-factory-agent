"""
ValidationNode单元测试 - 最关键的测试
"""
import pytest
from unittest.mock import patch, Mock
from src.nodes.validation import ValidationNode
from src.models import FacilityCandidate

class TestValidationNode:
    """ValidationNode测试套件"""

    @pytest.fixture
    def validation_node(self):
        """创建ValidationNode实例"""
        with patch('src.nodes.validation.AzureChatOpenAI') as mock_azure:
            mock_instance = Mock()
            mock_instance.invoke.return_value = Mock(content="YES")
            mock_azure.return_value = mock_instance
            node = ValidationNode()
            yield node

    # ============ 关键词过滤测试 ============

    def test_keyword_filter_rejects_headquarters(self, validation_node, sample_hq_candidate):
        """测试：拒绝总部"""
        passed, reason = validation_node._keyword_filter(sample_hq_candidate)

        assert not passed
        assert "headquarters" in reason.lower() or "excluded keyword" in reason.lower()

    def test_keyword_filter_rejects_hq(self, validation_node):
        """测试：拒绝HQ缩写"""
        candidate = FacilityCandidate(
            company_name="Test Company",
            facility_name="Company HQ Building",
            facility_type="office",
            full_address="123 Main St",
            country="USA",
            city="New York",
            evidence_urls=["https://example.com"]
        )

        passed, reason = validation_node._keyword_filter(candidate)
        assert not passed

    def test_keyword_filter_rejects_research_center(self, validation_node, sample_rd_candidate):
        """测试：拒绝R&D中心"""
        passed, reason = validation_node._keyword_filter(sample_rd_candidate)

        assert not passed
        # 可能匹配多个排除关键词（research/development/technical center等）
        assert "excluded keyword" in reason.lower()

    def test_keyword_filter_rejects_sales_office(self, validation_node):
        """测试：拒绝销售办公室"""
        candidate = FacilityCandidate(
            company_name="Test Company",
            facility_name="Regional Sales Office",
            facility_type="sales office",
            full_address="456 Commerce Ave",
            country="USA",
            city="Chicago",
            evidence_urls=["https://example.com"]
        )

        passed, reason = validation_node._keyword_filter(candidate)
        assert not passed

    def test_keyword_filter_accepts_factory(self, validation_node, sample_facility_candidate):
        """测试：接受工厂"""
        passed, reason = validation_node._keyword_filter(sample_facility_candidate)

        assert passed
        assert "passed" in reason.lower()

    def test_keyword_filter_accepts_refinery(self, validation_node):
        """测试：接受炼油厂"""
        candidate = FacilityCandidate(
            company_name="Shell",
            facility_name="Pernis Refinery",
            facility_type="oil refinery",
            full_address="Pernis, Rotterdam, Netherlands",
            country="Netherlands",
            city="Rotterdam",
            evidence_urls=["https://example.com"]
        )

        passed, reason = validation_node._keyword_filter(candidate)
        assert passed

    def test_keyword_filter_passes_without_manufacturing_keywords(self, validation_node):
        """测试：缺少制造关键词的设施通过关键词过滤（由LLM最终判断）"""
        candidate = FacilityCandidate(
            company_name="Test Company",
            facility_name="Test Facility",  # 没有factory/plant等关键词
            facility_type="building",
            full_address="789 Test St",
            country="USA",
            city="Boston",
            evidence_urls=["https://example.com"]
        )

        passed, reason = validation_node._keyword_filter(candidate)
        # 新策略：关键词过滤只检查排除词，不强制要求制造关键词
        # 让LLM做最终判断以提高召回率
        assert passed
        assert "passed" in reason.lower()

    # ============ LLM验证测试 ============

    def test_llm_validation_accepts_when_llm_says_yes(self, validation_node, sample_facility_candidate):
        """测试：LLM返回YES时接受"""
        validation_node.llm.invoke.return_value.content = "YES"

        passed, reason = validation_node._llm_validation(sample_facility_candidate)

        assert passed
        assert "manufacturing facility" in reason.lower()

    def test_llm_validation_rejects_when_llm_says_no(self, validation_node, sample_facility_candidate):
        """测试：LLM返回NO时拒绝"""
        validation_node.llm.invoke.return_value.content = "NO"

        passed, reason = validation_node._llm_validation(sample_facility_candidate)

        assert not passed

    def test_llm_validation_handles_exception(self, validation_node, sample_facility_candidate):
        """测试：LLM异常时保守拒绝"""
        validation_node.llm.invoke.side_effect = Exception("API Error")

        passed, reason = validation_node._llm_validation(sample_facility_candidate)

        assert not passed
        assert "error" in reason.lower()

    # ============ 完整流程测试 ============

    def test_run_filters_hq_correctly(self, validation_node, sample_hq_candidate):
        """测试：完整流程拒绝HQ"""
        state = {
            "extracted_facilities": [sample_hq_candidate],
            "validated_facilities": [],
            "errors": []
        }

        result_state = validation_node.run(state)

        # 应该被拒绝，validated_facilities应该为空
        assert len(result_state["validated_facilities"]) == 0

    def test_run_accepts_valid_factory(self, validation_node, sample_facility_candidate):
        """测试：完整流程接受工厂"""
        validation_node.llm.invoke.return_value.content = "YES"

        state = {
            "extracted_facilities": [sample_facility_candidate],
            "validated_facilities": [],
            "errors": []
        }

        result_state = validation_node.run(state)

        # 应该被接受
        assert len(result_state["validated_facilities"]) == 1
        assert result_state["validated_facilities"][0].validation_passed == True

    def test_run_filters_multiple_facilities(self, validation_node, sample_facility_candidate, sample_hq_candidate):
        """测试：混合设施正确过滤"""
        validation_node.llm.invoke.return_value.content = "YES"

        state = {
            "extracted_facilities": [
                sample_facility_candidate,  # 应该通过
                sample_hq_candidate         # 应该被拒绝
            ],
            "validated_facilities": [],
            "errors": []
        }

        result_state = validation_node.run(state)

        # 只有1个应该通过
        assert len(result_state["validated_facilities"]) == 1
        assert "Kentucky" in result_state["validated_facilities"][0].facility_name


# ============ 边缘案例测试 ============

class TestValidationEdgeCases:
    """边缘案例测试"""

    @pytest.fixture
    def validation_node(self):
        with patch('src.nodes.validation.AzureChatOpenAI') as mock_azure:
            mock_instance = Mock()
            mock_instance.invoke.return_value = Mock(content="YES")
            mock_azure.return_value = mock_instance
            node = ValidationNode()
            yield node

    def test_innovation_center_rejected(self, validation_node):
        """测试：拒绝创新中心"""
        candidate = FacilityCandidate(
            company_name="Apple",
            facility_name="Apple Innovation Center",
            facility_type="innovation center",
            full_address="Cupertino, CA",
            country="USA",
            city="Cupertino",
            evidence_urls=["https://example.com"]
        )

        passed, _ = validation_node._keyword_filter(candidate)
        assert not passed

    def test_distribution_center_rejected(self, validation_node):
        """测试：拒绝配送中心"""
        candidate = FacilityCandidate(
            company_name="Amazon",
            facility_name="Amazon Distribution Center",
            facility_type="distribution center",
            full_address="Seattle, WA",
            country="USA",
            city="Seattle",
            evidence_urls=["https://example.com"]
        )

        passed, _ = validation_node._keyword_filter(candidate)
        assert not passed

    def test_foundry_accepted(self, validation_node):
        """测试：接受铸造厂"""
        candidate = FacilityCandidate(
            company_name="TSMC",
            facility_name="TSMC Fab 18",
            facility_type="semiconductor foundry",
            full_address="Tainan, Taiwan",
            country="Taiwan",
            city="Tainan",
            evidence_urls=["https://example.com"]
        )

        passed, _ = validation_node._keyword_filter(candidate)
        assert passed
