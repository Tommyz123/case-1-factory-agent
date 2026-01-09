"""
缓存机制 - 减少重复API调用
"""
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ResultCache:
    """
    结果缓存类

    支持:
    - 搜索结果缓存
    - LLM响应缓存
    - 自动过期（默认7天）
    """

    def __init__(self, cache_dir: str = ".cache", ttl_days: int = 7):
        """
        初始化缓存

        Args:
            cache_dir: 缓存目录
            ttl_days: 缓存有效期（天）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days

        logger.info(f"Cache initialized: {self.cache_dir} (TTL={ttl_days} days)")

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存

        Args:
            key: 缓存键

        Returns:
            缓存的值，如果不存在或过期则返回None
        """
        cache_file = self._get_cache_file(key)

        if not cache_file.exists():
            logger.debug(f"Cache miss: {key[:50]}...")
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检查是否过期
            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time > timedelta(days=self.ttl_days):
                logger.info(f"Cache expired: {key[:50]}...")
                cache_file.unlink()  # 删除过期缓存
                return None

            logger.info(f"Cache hit: {key[:50]}...")
            return data['value']

        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None

    def set(self, key: str, value: Any):
        """
        设置缓存

        Args:
            key: 缓存键
            value: 要缓存的值
        """
        cache_file = self._get_cache_file(key)

        try:
            data = {
                'key': key,
                'timestamp': datetime.now().isoformat(),
                'value': value
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Cache set: {key[:50]}...")

        except Exception as e:
            logger.error(f"Error writing cache: {e}")

    def invalidate(self, key: str):
        """
        清除缓存

        Args:
            key: 缓存键
        """
        cache_file = self._get_cache_file(key)

        if cache_file.exists():
            cache_file.unlink()
            logger.info(f"Cache invalidated: {key[:50]}...")

    def clear_all(self):
        """清除所有缓存"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("All cache cleared")

    def _get_cache_file(self, key: str) -> Path:
        """
        获取缓存文件路径

        Args:
            key: 缓存键

        Returns:
            缓存文件路径
        """
        # 使用MD5哈希作为文件名
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    @staticmethod
    def make_key(*args, **kwargs) -> str:
        """
        生成缓存键

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            缓存键字符串
        """
        content = json.dumps(
            {"args": args, "kwargs": kwargs},
            sort_keys=True,
            default=str
        )
        return content
