"""型定義モジュール"""

from enum import StrEnum


class AppEnv(StrEnum):
    """環境名の列挙型"""

    DEV = 'dev'
    TEST = 'test'
    PROD = 'prod'
