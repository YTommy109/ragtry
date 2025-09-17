"""DocumentLoaderの統合テスト."""

from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from src.document_loader import DocumentLoader
from src.exceptions import OperationFailedError


class Testドキュメントローダー統合:
    """DocumentLoaderの統合テスト."""

    @pytest.mark.skip(
        reason='実際のHTTP通信を避けるため、モック化されたレスポンスでは空のPDFファイルが作成される'
    )
    def test_PDF処理でチャンクが生成される(self, document_loader: 'DocumentLoader') -> None:
        try:
            chunks = document_loader.process_scrum_guide_pdf()

            # 基本的な検証
            assert len(chunks) > 0, 'チャンクが生成されていない'

            # 最初の3つのチャンクを詳細確認
            for i, chunk in enumerate(chunks[:3]):
                assert chunk.page_content, f'チャンク {i + 1} の内容が空'
                assert chunk.metadata, f'チャンク {i + 1} のメタデータが空'
                assert len(chunk.page_content) > 0, f'チャンク {i + 1} の長さが0'

                # デバッグ情報を出力（テスト失敗時のみ）
                if i == 0:  # 最初のチャンクのみ
                    print(f'処理されたチャンク数: {len(chunks)}')
                    print(f'--- チャンク {i + 1} ---')
                    print(f'長さ: {len(chunk.page_content)}')
                    print(f'メタデータ: {chunk.metadata}')
                    print(f'内容(最初の100文字): {chunk.page_content[:100]}...')

        except Exception as e:
            raise OperationFailedError(operation='PDF処理', error=str(e)) from e

    def test_モックを使ってPDF処理できる(
        self, document_loader: 'DocumentLoader', mocker: MockerFixture
    ) -> None:
        from langchain_core.documents import Document  # noqa: PLC0415

        # 実際のメソッド名を確認してモック
        mock_download = mocker.patch.object(document_loader, 'download_pdf')
        mock_extract = mocker.patch.object(document_loader, 'extract_text_from_pdf')

        # モックの戻り値を設定（Documentオブジェクトのリストを返す）
        mock_download.return_value = Path('/tmp/mock.pdf')
        mock_extract.return_value = [
            Document(
                page_content='これはテスト用のPDFコンテンツです。' * 100,
                metadata={'source': 'test', 'page': 1},
            )
        ]

        chunks = document_loader.process_scrum_guide_pdf()

        # モックが呼ばれたことを確認
        mock_download.assert_called_once()
        mock_extract.assert_called_once()

        # 結果を確認
        assert len(chunks) > 0
        assert all(chunk.page_content for chunk in chunks)
