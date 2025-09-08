"""ベクトルストア管理モジュール.

Chromaデータベースの初期化、埋め込み処理、データ保存を行う。
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from .exceptions import OperationFailedError


class VectorStore:
    """ベクトルストアの管理を行うクラス."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = 'scrum_guide_collection',
        embedding_model: str = 'text-embedding-3-small',
    ) -> None:
        """初期化.

        Args:
            persist_directory: Chromaデータベースの永続化ディレクトリ
            collection_name: コレクション名
            embedding_model: 埋め込みモデル名
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # ベースデータディレクトリも作成しておく
        self.persist_directory.parent.mkdir(parents=True, exist_ok=True)
        # ベクトルDBディレクトリも作成
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # 埋め込み関数の初期化
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # Chromaクライアントの初期化
        self._initialize_chroma_client()

    def _initialize_chroma_client(self) -> None:
        """Chromaクライアントを初期化する."""
        # Chroma設定
        settings = Settings(
            persist_directory=str(self.persist_directory),
            anonymized_telemetry=False,
            is_persistent=True,
        )

        self.client = chromadb.Client(settings)

    def reset_client(self) -> None:
        """Chromaクライアントをリセットする.

        Raises:
            OperationFailedError: クライアントのリセットに失敗した場合
        """
        try:
            if hasattr(self, 'client'):
                del self.client
            self._initialize_chroma_client()
        except Exception as e:
            raise OperationFailedError(operation='クライアントのリセット', error=str(e)) from e

    def initialize_vectorstore(self) -> Chroma:
        """ベクトルストアを初期化する.

        Returns:
            初期化されたChromaベクトルストア

        Raises:
            Exception: 初期化に失敗した場合
        """
        try:
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory),
                client_settings=Settings(
                    persist_directory=str(self.persist_directory),
                    anonymized_telemetry=False,
                    is_persistent=True,
                ),
            )

        except Exception as e:
            # 設定の競合エラーの場合、クライアントをリセットして再試行
            if 'different settings' in str(e):
                self.reset_client()
                try:
                    return Chroma(
                        collection_name=self.collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=str(self.persist_directory),
                        client_settings=Settings(
                            persist_directory=str(self.persist_directory),
                            anonymized_telemetry=False,
                            is_persistent=True,
                        ),
                    )
                except Exception as retry_e:
                    raise OperationFailedError(
                        operation='ベクトルストア初期化', error=str(retry_e)
                    ) from retry_e

            raise OperationFailedError(operation='ベクトルストア初期化', error=str(e)) from e

    def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 100,
    ) -> None:
        """ドキュメントをベクトルストアに追加する.

        Args:
            documents: 追加するドキュメントのリスト
            batch_size: バッチ処理のサイズ

        Raises:
            Exception: ドキュメント追加に失敗した場合
        """
        if not documents:
            return

        try:
            vectorstore = self.initialize_vectorstore()

            # バッチ処理でドキュメントを追加
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                vectorstore.add_documents(batch)

            # 新しいlangchain-chromaでは自動的に永続化される

        except Exception as e:
            raise OperationFailedError(operation='ドキュメント追加', error=str(e)) from e

    def collection_exists(self) -> bool:
        """コレクションが存在するかチェックする.

        Returns:
            コレクションが存在する場合True
        """
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            return self.collection_name in collection_names
        except Exception:
            return False

    def delete_collection(self) -> None:
        """コレクションを削除する.

        Raises:
            Exception: コレクション削除に失敗した場合
        """
        try:
            if self.collection_exists():
                self.client.delete_collection(self.collection_name)
        except Exception as e:
            raise OperationFailedError(operation='コレクション削除', error=str(e)) from e

    def get_collection_count(self) -> int:
        """コレクション内のドキュメント数を取得する.

        Returns:
            ドキュメント数

        Raises:
            Exception: カウント取得に失敗した場合
        """
        try:
            if not self.collection_exists():
                return 0

            collection = self.client.get_collection(self.collection_name)
            return collection.count()

        except Exception as e:
            raise OperationFailedError(operation='ドキュメント数取得', error=str(e)) from e

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[Document]:
        """類似度検索を実行する.

        Args:
            query: 検索クエリ
            k: 取得する文書数

        Returns:
            類似度の高い文書のリスト

        Raises:
            Exception: 検索に失敗した場合
        """
        try:
            vectorstore = self.initialize_vectorstore()
            return vectorstore.similarity_search(query, k=k)

        except Exception as e:
            raise OperationFailedError(operation='類似度検索', error=str(e)) from e

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        """スコア付き類似度検索を実行する.

        Args:
            query: 検索クエリ
            k: 取得する文書数

        Returns:
            (文書, スコア)のタプルのリスト

        Raises:
            Exception: 検索に失敗した場合
        """
        try:
            vectorstore = self.initialize_vectorstore()
            return vectorstore.similarity_search_with_score(query, k=k)

        except Exception as e:
            raise OperationFailedError(operation='スコア付き類似度検索', error=str(e)) from e
