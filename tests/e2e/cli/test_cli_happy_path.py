"""CLI E2Eテスト - 簡略版."""

import subprocess
import sys
import tempfile
from pathlib import Path


class TestCLIHappyPath:
    """CLI機能のハッピーパステスト - PoCレベル."""

    def test_version_command(self) -> None:
        """バージョンコマンドのテスト."""
        result = subprocess.run(
            [sys.executable, 'main.py', 'version'],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert 'スクラムガイド拡張パック RAGシステム' in result.stdout
        assert 'v1.0.0' in result.stdout

    def test_help_command(self) -> None:
        """ヘルプコマンドのテスト."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--help'],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert 'スクラムガイド拡張パック RAGシステム' in result.stdout
        assert 'build' in result.stdout
        assert 'query' in result.stdout
        assert 'info' in result.stdout

    def test_info_command_no_database(self) -> None:
        """データベースなしでのinfoコマンドテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 環境変数を一時ディレクトリに設定
            env = {'DATA_DIR': str(Path(temp_dir)), 'OPENAI_API_KEY': 'sk-test-mock-key-1234567890'}

            result = subprocess.run(
                [sys.executable, 'main.py', '--env', 'dev', 'info'],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
            )

            assert result.returncode == 0
            assert 'システム情報' in result.stdout
            assert 'コレクション存在: いいえ' in result.stdout

    def test_query_command_no_database(self) -> None:
        """データベースなしでのqueryコマンドテスト."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env = {'DATA_DIR': str(Path(temp_dir)), 'OPENAI_API_KEY': 'sk-test-mock-key-1234567890'}

            result = subprocess.run(
                [sys.executable, 'main.py', '--env', 'dev', 'query', 'テスト質問'],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
            )

            assert result.returncode == 1
            assert 'ベクトルデータベースが存在しません' in result.stderr
            assert 'build コマンドを実行してください' in result.stderr

    def test_invalid_environment(self) -> None:
        """無効な環境指定のテスト."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--env', 'invalid', 'info'],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 2
        assert 'invalid' in result.stderr
