"""PDF文書読み込み・処理モジュール.

スクラムガイド拡張パックPDFのダウンロード、テキスト抽出、チャンク分割を行う。
"""

from pathlib import Path

import requests
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from .config import config
from .exceptions import NotFoundFileError, OperationFailedError


class DocumentLoader:
    """PDF文書の読み込みと処理を行うクラス."""

    def __init__(self, data_dir: Path = Path('data')) -> None:
        """初期化.

        Args:
            data_dir: データディレクトリのパス
        """
        self.data_dir = data_dir
        self.raw_dir = data_dir / 'raw'
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_pdf(self, url: str, filename: str) -> Path:
        """PDFファイルをダウンロードする.

        Args:
            url: PDFファイルのURL
            filename: 保存するファイル名

        Returns:
            保存されたファイルのパス

        Raises:
            requests.RequestException: ダウンロードに失敗した場合
            IOError: ファイル保存に失敗した場合
        """
        file_path = self.raw_dir / filename

        # ファイルが既に存在する場合はスキップ
        if file_path.exists():
            return file_path

        return self._download_file(url, file_path)

    def _download_file(self, url: str, file_path: Path) -> Path:
        """ファイルをダウンロードする内部メソッド."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            self._save_file(response.content, file_path)
            return file_path
        except requests.RequestException as e:
            raise OperationFailedError(operation='PDFダウンロード', error=str(e)) from e

    def _save_file(self, content: bytes, file_path: Path) -> None:
        """ファイルを保存する内部メソッド."""
        try:
            with open(file_path, 'wb') as f:
                f.write(content)
        except OSError as e:
            # ダウンロードに失敗した場合は部分的に作成されたファイルを削除
            if file_path.exists():
                file_path.unlink()
            raise OperationFailedError(operation='ファイル保存', error=str(e)) from e

    def extract_text_from_pdf(self, pdf_path: Path) -> list[Document]:
        """PDFからテキストを抽出する.

        Args:
            pdf_path: PDFファイルのパス

        Returns:
            抽出されたドキュメントのリスト

        Raises:
            NotFoundFileError: PDFファイルが存在しない場合
            Exception: PDF読み込みに失敗した場合
        """
        if not pdf_path.exists():
            raise NotFoundFileError(path=pdf_path, file_type='PDFファイル')

        try:
            loader = PyPDFLoader(str(pdf_path))
            return loader.load()

        except FileNotFoundError as e:
            raise NotFoundFileError(path=pdf_path, file_type='PDFファイル') from e
        except Exception as e:
            raise OperationFailedError(operation='ドキュメント読み込み', error=str(e)) from e

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """ドキュメントをチャンクに分割する.

        Args:
            documents: 分割するドキュメントのリスト

        Returns:
            分割されたドキュメントのリスト
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=list(config.separators),
        )

        split_docs = text_splitter.split_documents(documents)

        # メタデータの補強
        for i, doc in enumerate(split_docs):
            doc.metadata['chunk_id'] = i
            # 元のドキュメントのsourceを保持（複数ドキュメントの場合も考慮）
            if documents and 'source' in documents[0].metadata:
                doc.metadata['source'] = str(documents[0].metadata.get('source', ''))

        return split_docs

    def process_scrum_guide_pdf(self) -> list[Document]:
        """スクラムガイド拡張パックPDFを処理する.

        Returns:
            処理されたドキュメントチャンクのリスト

        Raises:
            Exception: 処理に失敗した場合
        """
        try:
            # PDFダウンロード
            pdf_path = self.download_pdf(config.pdf_url, config.pdf_filename)

            # テキスト抽出
            documents = self.extract_text_from_pdf(pdf_path)

            # チャンク分割
            return self.split_documents(documents)

        except Exception as e:
            raise OperationFailedError(operation='スクラムガイドPDF処理', error=str(e)) from e
