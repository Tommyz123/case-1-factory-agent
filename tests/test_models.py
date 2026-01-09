"""
Pydantic模型验证测试
"""
import pytest
from pydantic import ValidationError
from src.models import FacilityCandidate, ValidatedFacility

class TestFacilityCandidate:
    """FacilityCandidate模型测试"""

    def test_valid_facility_candidate(self):
        """测试：有效的设施候选"""
        candidate = FacilityCandidate(
            company_name="Toyota",
            facility_name="Kentucky Plant",
            facility_type="assembly plant",
            full_address="Georgetown, KY, USA",
            country="USA",
            city="Georgetown",
            evidence_urls=["https://example.com"]
        )

        assert candidate.company_name == "Toyota"
        assert len(candidate.evidence_urls) > 0

    def test_evidence_urls_required(self):
        """测试：evidence_urls必填"""
        with pytest.raises(ValidationError) as exc_info:
            FacilityCandidate(
                company_name="Toyota",
                facility_name="Kentucky Plant",
                facility_type="assembly plant",
                full_address="Georgetown, KY, USA",
                country="USA",
                city="Georgetown",
                evidence_urls=[]  # 空列表应该失败
            )

        assert "evidence_urls" in str(exc_info.value)

    def test_evidence_urls_cannot_be_empty_list(self):
        """测试：evidence_urls不能为空列表"""
        with pytest.raises(ValidationError):
            FacilityCandidate(
                company_name="Toyota",
                facility_name="Test",
                facility_type="factory",
                full_address="Test",
                country="USA",
                city="Test",
                evidence_urls=[]
            )

    def test_latitude_validation(self):
        """测试：纬度范围验证"""
        with pytest.raises(ValidationError):
            FacilityCandidate(
                company_name="Toyota",
                facility_name="Test",
                facility_type="factory",
                full_address="Test",
                country="USA",
                city="Test",
                evidence_urls=["https://example.com"],
                latitude=100.0  # 超出范围
            )

    def test_longitude_validation(self):
        """测试：经度范围验证"""
        with pytest.raises(ValidationError):
            FacilityCandidate(
                company_name="Toyota",
                facility_name="Test",
                facility_type="factory",
                full_address="Test",
                country="USA",
                city="Test",
                evidence_urls=["https://example.com"],
                longitude=200.0  # 超出范围
            )

    def test_optional_coordinates(self):
        """测试：坐标是可选的"""
        candidate = FacilityCandidate(
            company_name="Toyota",
            facility_name="Test Plant",
            facility_type="factory",
            full_address="Test Address",
            country="USA",
            city="Test City",
            evidence_urls=["https://example.com"]
            # latitude和longitude不提供
        )

        assert candidate.latitude is None
        assert candidate.longitude is None

    def test_confidence_score_range(self):
        """测试：置信度范围验证"""
        # 有效范围
        candidate = FacilityCandidate(
            company_name="Toyota",
            facility_name="Test",
            facility_type="factory",
            full_address="Test",
            country="USA",
            city="Test",
            evidence_urls=["https://example.com"],
            confidence_score=0.85
        )
        assert candidate.confidence_score == 0.85

        # 超出范围
        with pytest.raises(ValidationError):
            FacilityCandidate(
                company_name="Toyota",
                facility_name="Test",
                facility_type="factory",
                full_address="Test",
                country="USA",
                city="Test",
                evidence_urls=["https://example.com"],
                confidence_score=1.5  # 超出[0,1]范围
            )


class TestValidatedFacility:
    """ValidatedFacility模型测试"""

    def test_valid_validated_facility(self):
        """测试：有效的验证后设施"""
        facility = ValidatedFacility(
            company_name="Toyota",
            facility_name="Kentucky Plant",
            facility_type="assembly plant",
            full_address="Georgetown, KY, USA",
            country="USA",
            city="Georgetown",
            evidence_urls=["https://example.com"],
            confidence_score=0.9,
            validation_passed=True,
            validation_reason="Passed all checks"
        )

        assert facility.validation_passed == True
        assert facility.validation_reason == "Passed all checks"

    def test_inherits_from_facility_candidate(self):
        """测试：继承自FacilityCandidate"""
        facility = ValidatedFacility(
            company_name="Toyota",
            facility_name="Test",
            facility_type="factory",
            full_address="Test",
            country="USA",
            city="Test",
            evidence_urls=["https://example.com"],
            validation_passed=True,
            validation_reason="Test"
        )

        # 应该有FacilityCandidate的所有字段
        assert hasattr(facility, 'company_name')
        assert hasattr(facility, 'facility_name')
        assert hasattr(facility, 'evidence_urls')

        # 加上额外字段
        assert hasattr(facility, 'validation_passed')
        assert hasattr(facility, 'validation_reason')
