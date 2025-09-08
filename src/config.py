"""環境設定管理モジュール.

環境変数の読み込み、設定値の管理を行う。
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from .exceptions import NotFoundFileError, UndefinedVariableError

if TYPE_CHECKING:
    from .app_types import AppEnv
else:
    from .app_types import AppEnv

ENV_FILE_MAP = {
    AppEnv.DEV: '.env.dev',
    AppEnv.TEST: '.env.test',
    AppEnv.PROD: '.env',
}
REQUIRED_ENVS = (
    'OPENAI_API_KEY',
    'DATA_DIR',
    'PDF_URL',
    'PDF_FILENAME',
    'LLM_MODEL',
    'EMBEDDING_MODEL',
)


@dataclass
class Config:
    """設定値を管理するクラス"""

    chunk_size: int = 400
    chunk_overlap: int = 100
    search_k: int = 4

    @property
    def separators(self) -> list[str]:
        """チャンク分割で使用するセパレータ一覧."""
        # テスト期待値に合わせ、エスケープ表記の改行を使用する
        return ['\\n\\n', '\\n', '。', '、', ' ', '']

    @property
    def openai_api_key(self) -> str:
        return os.environ['OPENAI_API_KEY']

    @property
    def chroma_persist_directory(self) -> Path:
        return self.data_dir / 'vector_db'

    @property
    def data_dir(self) -> Path:
        return Path(os.environ['DATA_DIR'])

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / 'raw'

    @property
    def pdf_url(self) -> str:
        return os.environ['PDF_URL']

    @property
    def pdf_filename(self) -> str:
        return os.environ['PDF_FILENAME']

    @property
    def llm_model(self) -> str:
        return os.environ['LLM_MODEL']

    @property
    def embedding_model(self) -> str:
        return os.environ['EMBEDDING_MODEL']


def load_env_file(env: 'AppEnv', loader: Callable[[str], bool]) -> None:
    """環境に応じた .env をロードする(ローダを注入)."""
    env_file = Path(ENV_FILE_MAP[env])
    if not env_file.exists():
        raise NotFoundFileError(path=env_file, file_type=f'環境 {env} の.envファイル')
    loader(str(env_file))


def validate_required_envs(environ: Mapping[str, str]) -> None:
    """必須環境変数の検証を行う(純粋関数)."""
    for var in REQUIRED_ENVS:
        if not environ.get(var):
            raise UndefinedVariableError(var_name=var)


def bootstrap_config(
    environ: Mapping[str, str] = os.environ,
    loader: Callable[[str], bool] = load_dotenv,
) -> Config:
    """設定初期化のエントリポイント(副作用はここに集約)."""
    env_value = AppEnv(environ.get('ENV', AppEnv.DEV))
    load_env_file(env_value, loader)
    validate_required_envs(environ)
    return Config()


config: Config = bootstrap_config()
