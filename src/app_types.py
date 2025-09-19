"""型定義モジュール"""

from enum import StrEnum


class AppEnv(StrEnum):
    """環境名の列挙型"""

    DEV = 'dev'
    TEST = 'test'
    PROD = 'prod'


class SearchType(StrEnum):
    """検索タイプの列挙型"""

    SEMANTIC = 'semantic'  # セマンティック検索のみ
    KEYWORD = 'keyword'  # キーワード検索のみ
    HYBRID = 'hybrid'  # ハイブリッド検索
