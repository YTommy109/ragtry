"""RAGシステム固有の例外クラス定義"""

from pathlib import Path


class BaseError(Exception):
    """RAGシステム全般のベース例外"""

    def __init__(self, message: str | None = None) -> None:
        """例外を初期化する"""
        self.message = message or 'RAGシステムでエラーが発生しました'
        super().__init__(self.message)


class NotFoundFileError(BaseError):
    """ファイルが見つからない例外"""

    def __init__(self, path: Path, file_type: str | None = None) -> None:
        """ファイルが見つからない例外を初期化する"""
        self.path = str(path)
        self.file_type = file_type or 'ファイル'
        message = f'{self.file_type}が見つかりません: {self.path}'
        super().__init__(message)


class OperationFailedError(BaseError):
    """操作が失敗した例外"""

    def __init__(self, operation: str, error: str | None = None) -> None:
        """操作失敗例外を初期化する"""
        self.operation = operation
        self.error = error or '不明なエラー'
        message = f'{operation}に失敗しました: {self.error}'
        super().__init__(message)


class UndefinedVariableError(BaseError):
    """必須の変数が設定されていない例外"""

    def __init__(self, var_name: str) -> None:
        message = f'必須の変数が設定されていません: {var_name}'
        super().__init__(message)
