"""CLI モジュールのテスト."""

import pytest
from click.testing import CliRunner
from pytest_mock import MockerFixture

from src.cli import cli


class TestCLICommands:
    """CLI コマンドのテスト."""

    def test_version_command(self, runner: CliRunner) -> None:
        """バージョンコマンドのテスト."""
        # Arrange
        # なし

        # Act
        result = runner.invoke(cli, ['version'])

        # Assert
        assert result.exit_code == 0
        assert 'スクラムガイド拡張パック RAGシステム v1.0.0' in result.output

    def test_info_command_success(self, mocker: MockerFixture, runner: CliRunner) -> None:
        """info コマンド成功のテスト."""
        # Arrange（グローバルconfigをそのまま使用）
        mock_rag_service = mocker.patch('src.cli.RAGService')
        mock_service_instance = mocker.Mock()
        mock_service_instance.get_collection_info.return_value = {
            'collection_exists': True,
            'document_count': 100,
        }
        mock_rag_service.return_value = mock_service_instance

        # Act
        result = runner.invoke(cli, ['info'])

        # Assert（ENVの具体値には依存しない）
        assert result.exit_code == 0
        assert '実行環境:' in result.output
        assert 'コレクション存在: はい' in result.output
        assert '文書数: 100件' in result.output

    def test_build_command_success(self, mocker: MockerFixture, runner: CliRunner) -> None:
        """build コマンド成功のテスト."""
        # Arrange（グローバルconfigをそのまま使用）
        mock_rag_service = mocker.patch('src.cli.RAGService')
        mock_service_instance = mocker.Mock()
        mock_service_instance.get_collection_info.return_value = {
            'collection_exists': False,
            'document_count': 0,
        }
        mock_service_instance.build_knowledge_base.return_value = None
        mock_rag_service.return_value = mock_service_instance

        # Act
        result = runner.invoke(cli, ['build'])

        # Assert
        assert result.exit_code == 0
        assert 'ベクトルデータベースの構築を開始します' in result.output
        mock_service_instance.build_knowledge_base.assert_called_once()

    def test_build_command_with_existing_data(
        self, mocker: MockerFixture, runner: CliRunner
    ) -> None:
        """既存データありでのbuild コマンドのテスト."""
        # Arrange（グローバルconfigをそのまま使用）
        mock_rag_service = mocker.patch('src.cli.RAGService')
        mock_service_instance = mocker.Mock()
        mock_service_instance.get_collection_info.return_value = {
            'collection_exists': True,
            'document_count': 50,
        }
        mock_service_instance.build_knowledge_base.return_value = None
        mock_rag_service.return_value = mock_service_instance

        # Act ('y'で応答する入力)
        result = runner.invoke(cli, ['build'], input='y\n')

        # Assert
        assert result.exit_code == 0
        assert '既存のコレクションを発見しました(文書数: 50)' in result.output
        mock_service_instance.build_knowledge_base.assert_called_once()

    def test_build_command_user_cancels(self, mocker: MockerFixture, runner: CliRunner) -> None:
        """ユーザーがキャンセルした場合のbuild コマンドのテスト."""
        # Arrange（グローバルconfigをそのまま使用）
        mock_rag_service = mocker.patch('src.cli.RAGService')
        mock_service_instance = mocker.Mock()
        mock_service_instance.get_collection_info.return_value = {
            'collection_exists': True,
            'document_count': 50,
        }
        mock_rag_service.return_value = mock_service_instance

        # Act ('n'で応答する入力)
        result = runner.invoke(cli, ['build'], input='n\n')

        # Assert
        assert result.exit_code == 0
        assert '構築をキャンセルしました' in result.output
        mock_service_instance.build_knowledge_base.assert_not_called()

    def test_query_command_success(self, mocker: MockerFixture, runner: CliRunner) -> None:
        """query コマンド成功のテスト."""
        # Arrange（グローバルconfigをそのまま使用）
        mock_rag_service = mocker.patch('src.cli.RAGService')
        mock_service_instance = mocker.Mock()
        mock_service_instance.get_collection_info.return_value = {
            'collection_exists': True,
            'document_count': 100,
        }
        mock_service_instance.search_similar_documents.return_value = [
            mocker.Mock(page_content='テスト内容', metadata={'page': 1})
        ]
        mock_service_instance.generate_response.return_value = 'テスト回答'
        mock_rag_service.return_value = mock_service_instance

        # Act
        result = runner.invoke(cli, ['query', 'スクラムとは何ですか?'], input='n\n')

        # Assert
        assert result.exit_code == 0
        assert '質問: スクラムとは何ですか?' in result.output
        assert '【回答】' in result.output
        assert 'テスト回答' in result.output

    def test_query_command_no_database(self, mocker: MockerFixture, runner: CliRunner) -> None:
        """データベースなしでのquery コマンドのテスト."""
        # Arrange（グローバルconfigをそのまま使用）
        mock_rag_service = mocker.patch('src.cli.RAGService')
        mock_service_instance = mocker.Mock()
        mock_service_instance.get_collection_info.return_value = {
            'collection_exists': False,
            'document_count': 0,
        }
        mock_rag_service.return_value = mock_service_instance

        # Act
        result = runner.invoke(cli, ['query', 'テスト質問'])

        # Assert
        assert result.exit_code == 1
        assert 'ベクトルデータベースが存在しません' in result.output

    def test_invalid_environment(self, runner: CliRunner) -> None:
        """無効な環境指定のテスト."""
        # Arrange
        # なし

        # Act
        result = runner.invoke(cli, ['--env', 'invalid', 'info'])

        # Assert
        assert result.exit_code == 2

    def test_config_error_handling(self, mocker: MockerFixture, runner: CliRunner) -> None:
        """設定エラーのハンドリングテスト."""
        # Arrange (RAGServiceでエラーを発生させる)
        mock_rag_service = mocker.patch('src.cli.RAGService')
        mock_service_instance = mocker.Mock()
        mock_service_instance.get_collection_info.side_effect = Exception('設定エラー')
        mock_rag_service.return_value = mock_service_instance

        # Act
        result = runner.invoke(cli, ['--env', 'dev', 'info'])

        # Assert
        assert result.exit_code == 1
        assert 'エラー' in result.output


@pytest.fixture
def runner() -> CliRunner:
    """Click CLIランナーを提供する."""
    return CliRunner()
