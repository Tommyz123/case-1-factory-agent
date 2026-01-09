"""
重试装饰器
"""
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def retry_with_backoff(retries=3, backoff_factor=2, exceptions=(Exception,)):
    """
    带指数退避的重试装饰器

    Args:
        retries: 重试次数
        backoff_factor: 退避因子
        exceptions: 需要重试的异常类型

    Example:
        @retry_with_backoff(retries=3, exceptions=(requests.RequestException,))
        def my_function():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == retries - 1:
                        logger.error(f"{func.__name__} failed after {retries} attempts: {e}")
                        raise

                    wait_time = backoff_factor ** attempt
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{retries}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)

        return wrapper
    return decorator
