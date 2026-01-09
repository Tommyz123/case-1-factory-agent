"""
DeduplicationNode: 使用Embedding进行语义去重
"""
import os
import logging
import numpy as np
from typing import List, Dict, Any
from openai import AzureOpenAI
from src.models import ValidatedFacility

logger = logging.getLogger(__name__)


class DeduplicationNode:
    """
    去重节点 - 使用text-embedding-3-small

    去重策略:
    1. 完全相同的地址 → 直接合并
    2. Embedding相似度 > 0.95 → 视为重复
    3. 经纬度距离 < 1km → 视为重复（如果有坐标）
    """

    def __init__(self, metrics_collector=None):
        """
        初始化OpenAI client for embeddings

        Args:
            metrics_collector: Optional MetricsCollector for tracking API usage
        """
        try:
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            self.similarity_threshold = 0.95
            self.metrics = metrics_collector
            logger.info("DeduplicationNode initialized with Azure OpenAI")
        except Exception as e:
            logger.error(f"Failed to initialize DeduplicationNode: {e}")
            raise

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行去重

        Args:
            state: 包含 validated_facilities 的状态

        Returns:
            更新后的状态，包含去重后的设施列表
        """
        facilities = state.get("validated_facilities", [])
        logger.info(f"Deduplicating {len(facilities)} facilities")

        if len(facilities) <= 1:
            logger.info("No deduplication needed (0 or 1 facility)")
            return state

        # Step 1: 精确地址去重
        facilities = self._deduplicate_by_address(facilities)
        logger.info(f"After address dedup: {len(facilities)} facilities")

        # Step 2: Embedding语义去重
        facilities = self._deduplicate_by_embedding(facilities)
        logger.info(f"After embedding dedup: {len(facilities)} facilities")

        # Step 3: 地理坐标去重（如果有）
        facilities = self._deduplicate_by_coordinates(facilities)
        logger.info(f"After coordinate dedup: {len(facilities)} facilities")

        state["validated_facilities"] = facilities
        return state

    def _deduplicate_by_address(self, facilities: List[ValidatedFacility]) -> List[ValidatedFacility]:
        """
        基于精确地址去重

        Args:
            facilities: 设施列表

        Returns:
            去重后的设施列表
        """
        seen_addresses = {}
        deduplicated = []

        for facility in facilities:
            # 标准化地址（小写、去空格）
            normalized_address = facility.full_address.lower().strip()

            if normalized_address in seen_addresses:
                # 地址重复，合并到已存在的设施
                existing_idx = seen_addresses[normalized_address]
                deduplicated[existing_idx] = self._merge_facilities(deduplicated[existing_idx], facility)
                logger.info(f"Merged duplicate address: {facility.facility_name}")
            else:
                seen_addresses[normalized_address] = len(deduplicated)
                deduplicated.append(facility)

        return deduplicated

    def _deduplicate_by_embedding(self, facilities: List[ValidatedFacility]) -> List[ValidatedFacility]:
        """
        基于Embedding语义相似度去重

        Args:
            facilities: 设施列表

        Returns:
            去重后的设施列表
        """
        if len(facilities) <= 1:
            return facilities

        try:
            # 生成embeddings
            embeddings = self._get_embeddings(facilities)

            # 计算相似度矩阵
            similarity_matrix = self._compute_similarity_matrix(embeddings)

            # 标记重复项
            to_remove = set()
            for i in range(len(facilities)):
                if i in to_remove:
                    continue

                for j in range(i + 1, len(facilities)):
                    if j in to_remove:
                        continue

                    if similarity_matrix[i][j] > self.similarity_threshold:
                        logger.info(
                            f"Found duplicate (similarity={similarity_matrix[i][j]:.3f}): "
                            f"{facilities[i].facility_name} ≈ {facilities[j].facility_name}"
                        )
                        # 合并j到i
                        facilities[i] = self._merge_facilities(facilities[i], facilities[j])
                        to_remove.add(j)

            # 移除重复项
            deduplicated = [f for idx, f in enumerate(facilities) if idx not in to_remove]
            return deduplicated

        except Exception as e:
            logger.error(f"Embedding deduplication failed: {e}")
            # 失败时返回原列表
            return facilities

    def _deduplicate_by_coordinates(self, facilities: List[ValidatedFacility]) -> List[ValidatedFacility]:
        """
        基于地理坐标去重（如果有坐标）

        距离 < 1km 视为重复

        Args:
            facilities: 设施列表

        Returns:
            去重后的设施列表
        """
        # 筛选有坐标的设施
        facilities_with_coords = [
            (idx, f) for idx, f in enumerate(facilities)
            if f.latitude is not None and f.longitude is not None
        ]

        if len(facilities_with_coords) <= 1:
            return facilities

        to_remove = set()

        for i, (idx_i, facility_i) in enumerate(facilities_with_coords):
            if idx_i in to_remove:
                continue

            for j in range(i + 1, len(facilities_with_coords)):
                idx_j, facility_j = facilities_with_coords[j]

                if idx_j in to_remove:
                    continue

                # 计算距离
                distance_km = self._haversine_distance(
                    facility_i.latitude, facility_i.longitude,
                    facility_j.latitude, facility_j.longitude
                )

                if distance_km < 1.0:  # < 1km
                    logger.info(
                        f"Found duplicate by coordinates ({distance_km:.2f}km): "
                        f"{facility_i.facility_name} ≈ {facility_j.facility_name}"
                    )
                    # 合并
                    facilities[idx_i] = self._merge_facilities(facilities[idx_i], facilities[idx_j])
                    to_remove.add(idx_j)

        # 移除重复项
        deduplicated = [f for idx, f in enumerate(facilities) if idx not in to_remove]
        return deduplicated

    def _get_embeddings(self, facilities: List[ValidatedFacility]) -> np.ndarray:
        """
        获取设施的embeddings

        使用 facility_name + full_address 作为文本

        Args:
            facilities: 设施列表

        Returns:
            embeddings矩阵 (n_facilities, embedding_dim)
        """
        texts = [
            f"{f.facility_name} {f.full_address}"
            for f in facilities
        ]

        response = self.client.embeddings.create(
            input=texts,
            model=self.embedding_model
        )

        # Track embedding token usage
        if hasattr(response, 'usage') and self.metrics:
            total_tokens = response.usage.total_tokens
            self.metrics.track_embedding_call(
                node_name="deduplication",
                total_tokens=total_tokens,
                model=self.embedding_model
            )
            logger.debug(f"Embedding API call: {total_tokens} tokens")

        embeddings = np.array([item.embedding for item in response.data])
        return embeddings

    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        计算余弦相似度矩阵

        Args:
            embeddings: embeddings矩阵

        Returns:
            相似度矩阵 (n x n)
        """
        # 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / norms

        # 计算余弦相似度
        similarity_matrix = embeddings_normalized @ embeddings_normalized.T

        return similarity_matrix

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        计算两点间的Haversine距离（km）

        Args:
            lat1, lon1: 点1的纬度和经度
            lat2, lon2: 点2的纬度和经度

        Returns:
            距离（公里）
        """
        from math import radians, sin, cos, sqrt, atan2

        R = 6371  # 地球半径（km）

        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)

        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c

        return distance

    def _merge_facilities(self, facility1: ValidatedFacility, facility2: ValidatedFacility) -> ValidatedFacility:
        """
        合并两个重复的设施

        策略:
        - 保留第一个设施的基本信息
        - 合并所有evidence_urls
        - 取最高的confidence_score

        Args:
            facility1: 设施1（保留）
            facility2: 设施2（合并到1）

        Returns:
            合并后的设施
        """
        # 合并evidence_urls（去重）
        merged_evidence = list(set(facility1.evidence_urls + facility2.evidence_urls))

        # 取最高confidence_score
        merged_confidence = max(
            facility1.confidence_score or 0,
            facility2.confidence_score or 0
        )

        # 创建新的ValidatedFacility
        merged = ValidatedFacility(
            company_name=facility1.company_name,
            facility_name=facility1.facility_name,
            facility_type=facility1.facility_type,
            full_address=facility1.full_address,
            country=facility1.country,
            city=facility1.city,
            latitude=facility1.latitude or facility2.latitude,
            longitude=facility1.longitude or facility2.longitude,
            evidence_urls=merged_evidence,
            confidence_score=merged_confidence,
            validation_passed=True,
            validation_reason=f"Merged duplicate: {facility1.validation_reason}"
        )

        return merged
