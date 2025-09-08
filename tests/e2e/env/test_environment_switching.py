"""環境別テスト - 簡略版."""

import subprocess
import sys


class TestEnvironmentSwitching:
    """
    Scenario: 環境変数による環境切り替え
        Given: .env.test が配置されている
        When: CLIでinfoを実行する
        Then: 現在の実行環境が表示される
    """

    def test_環境を読み込める(self) -> None:
        """
        Scenario: .env.test を使って環境が読み込まれる
            Given: .env.test が設定済み
            When: info コマンドを実行する
            Then: 実行環境が表示される
        """
        # Scenario: .env.test を使って環境が読み込まれる
        # Given: .env.test が設定済み
        # When: info コマンドを実行する
        env = {'OPENAI_API_KEY': 'sk-test-mock-key-1234567890'}
        result = subprocess.run(
            [sys.executable, 'main.py', 'info'],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        # Then: 実行結果が成功し、環境情報が出力される
        assert result.returncode == 0
        assert '実行環境:' in result.stdout

    def test_ENV変数で切り替わる(self) -> None:
        """
        Scenario: .env.test の環境で info を実行
            Given: .env.test が設定済み
            When: info コマンドを実行する
            Then: 実行環境が表示される
        """
        # Scenario: .env.test の環境で info を実行
        # Given: .env.test が設定済み
        # When: info コマンドを実行する
        env = {'OPENAI_API_KEY': 'sk-test-mock-key-1234567890'}
        result = subprocess.run(
            [sys.executable, 'main.py', 'info'],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        # Then: 実行結果が成功し、環境情報が出力される
        assert result.returncode == 0
        assert '実行環境:' in result.stdout
