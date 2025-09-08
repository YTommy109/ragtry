"""document_loader モジュールのテスト."""

import tempfile
from pathlib import Path

import pytest
import requests
from langchain.schema import Document
from pytest_mock import MockerFixture

from src.document_loader import DocumentLoader
from src.exceptions import NotFoundFileError, OperationFailedError


class Testドキュメントローダー:
    def setup_method(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.loader = DocumentLoader(data_dir=self.temp_dir)

    def test_初期化(self) -> None:
        # データディレクトリが作成されることを確認
        assert self.loader.data_dir == self.temp_dir
        assert self.loader.raw_dir == self.temp_dir / 'raw'
        assert self.loader.raw_dir.exists()

    def test_PDFをダウンロードできる(self, mocker: MockerFixture) -> None:
        # モックの設定
        mock_get = mocker.patch('requests.get')
        mock_response = mocker.Mock()
        mock_response.content = b'fake pdf content'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # テスト実行
        url = 'https://example.com/test.pdf'
        filename = 'test.pdf'
        result_path = self.loader.download_pdf(url, filename)

        # 検証
        assert result_path == self.loader.raw_dir / filename
        assert result_path.exists()
        mock_get.assert_called_once_with(url, timeout=30)

    def test_既存PDFはダウンロードしない(self, mocker: MockerFixture) -> None:
        # モックの設定
        mock_get = mocker.patch('requests.get')

        # 既存ファイルを作成
        filename = 'existing.pdf'
        existing_file = self.loader.raw_dir / filename
        existing_file.write_text('existing content')

        # テスト実行
        url = 'https://example.com/test.pdf'
        result_path = self.loader.download_pdf(url, filename)

        # 検証
        assert result_path == existing_file
        mock_get.assert_not_called()

    def test_PDFダウンロードが失敗する(self, mocker: MockerFixture) -> None:
        # モックの設定
        mock_get = mocker.patch('requests.get')
        mock_get.side_effect = requests.RequestException('Connection failed')

        # テスト実行と検証
        with pytest.raises(OperationFailedError):
            self.loader.download_pdf('https://example.com/test.pdf', 'test.pdf')

    def test_存在する(self) -> None:
        # ファイルを作成
        filename = 'test_file.pdf'
        test_file = self.loader.raw_dir / filename
        test_file.write_text('test content')

        # テスト実行 - Path.exists()を直接使用
        result = (self.loader.raw_dir / filename).exists()

        # 検証
        assert result is True

    def test_存在しない(self) -> None:
        # テスト実行 - Path.exists()を直接使用
        result = (self.loader.raw_dir / 'nonexistent.pdf').exists()

        # 検証
        assert result is False

    def test_PDFからテキストを抽出できる(self, mocker: MockerFixture) -> None:
        # テストファイルを作成
        pdf_path = self.loader.raw_dir / 'test.pdf'
        pdf_path.write_bytes(b'fake pdf content')

        # モックの設定
        mock_pdf_loader = mocker.patch('src.document_loader.PyPDFLoader')
        mock_loader_instance = mocker.Mock()
        mock_documents = [Document(page_content='Test content', metadata={'page': 1})]
        mock_loader_instance.load.return_value = mock_documents
        mock_pdf_loader.return_value = mock_loader_instance

        # テスト実行
        result = self.loader.extract_text_from_pdf(pdf_path)

        # 検証
        assert result == mock_documents
        mock_pdf_loader.assert_called_once_with(str(pdf_path))

    def test_PDF抽出がファイルなしで失敗する(self) -> None:
        nonexistent_path = self.loader.raw_dir / 'nonexistent.pdf'

        with pytest.raises(NotFoundFileError):
            self.loader.extract_text_from_pdf(nonexistent_path)

    def test_ドキュメントを分割できる(self, mocker: MockerFixture) -> None:
        # モックの設定
        mock_splitter = mocker.patch('src.document_loader.RecursiveCharacterTextSplitter')
        mock_splitter_instance = mocker.Mock()
        split_docs = [
            Document(page_content='chunk1', metadata={}),
            Document(page_content='chunk2', metadata={}),
        ]
        mock_splitter_instance.split_documents.return_value = split_docs
        mock_splitter.return_value = mock_splitter_instance

        # テスト用ドキュメント
        original_docs = [Document(page_content='Original content', metadata={'source': 'test.pdf'})]

        # テスト実行
        result = self.loader.split_documents(original_docs)

        # 検証
        assert len(result) == 2
        assert result[0].metadata['chunk_id'] == 0
        assert result[1].metadata['chunk_id'] == 1
        assert result[0].metadata['source'] == 'test.pdf'
        assert result[1].metadata['source'] == 'test.pdf'

        # splitter が正しく呼ばれたことを確認
        mock_splitter.assert_called_once_with(
            chunk_size=400,
            chunk_overlap=100,
            separators=['\\n\\n', '\\n', '。', '、', ' ', ''],
        )

    def test_PDFを処理できる(self, mocker: MockerFixture) -> None:
        # モックの設定
        mock_download = mocker.patch.object(DocumentLoader, 'download_pdf')
        mock_extract = mocker.patch.object(DocumentLoader, 'extract_text_from_pdf')
        mock_split = mocker.patch.object(DocumentLoader, 'split_documents')

        mock_path = Path('fake/path/test.pdf')
        mock_download.return_value = mock_path

        mock_documents = [Document(page_content='test content', metadata={})]
        mock_extract.return_value = mock_documents

        mock_chunks = [
            Document(page_content='chunk1', metadata={}),
            Document(page_content='chunk2', metadata={}),
        ]
        mock_split.return_value = mock_chunks

        # テスト実行
        result = self.loader.process_scrum_guide_pdf()

        # 検証
        assert result == mock_chunks
        expected_url = 'https://scrumexpansion.org/ja/scrum-guide-expansion-pack/2025.6/pdf/scrum-guide-expansion-pack.ja.pdf'
        expected_filename = 'scrum-guide-expansion-pack.ja.pdf'

        mock_download.assert_called_once_with(expected_url, expected_filename)
        mock_extract.assert_called_once_with(mock_path)
        mock_split.assert_called_once_with(mock_documents)

    def test_PDF処理が失敗する(self, mocker: MockerFixture) -> None:
        # モックの設定
        mock_download = mocker.patch.object(DocumentLoader, 'download_pdf')
        mock_download.side_effect = requests.RequestException('Download failed')

        # テスト実行と検証
        with pytest.raises(OperationFailedError):
            self.loader.process_scrum_guide_pdf()


class Testドキュメントローダー統合:
    def setup_method(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.loader = DocumentLoader(data_dir=self.temp_dir)

    def test_ファイルを保存できる(self) -> None:
        content = b'test file content'
        file_path = self.temp_dir / 'test_file.txt'

        # _save_fileメソッドを直接テスト
        self.loader._save_file(content, file_path)

        # ファイルが正しく保存されていることを確認
        assert file_path.exists()
        assert file_path.read_bytes() == content

    def test_ファイル保存がIOエラーで失敗する(self) -> None:
        # 存在しないディレクトリにファイル保存を試行
        invalid_path = Path('/invalid/path/test.txt')

        with pytest.raises(OperationFailedError):
            self.loader._save_file(b'content', invalid_path)
