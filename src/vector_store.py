"""ベクトルストア管理モジュール.

FAISSベクトルデータベースの初期化、埋め込み処理、データ保存を行う。
"""

from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from .exceptions import OperationFailedError


class VectorStore:
    """ベクトルストアの管理を行うクラス."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = 'scrum_guide_collection',
        embedding_model: str = 'text-embedding-3-small',
        openai_api_base: str | None = None,
    ) -> None:
        """初期化.

        Args:
            persist_directory: FAISSベクトルデータベースの永続化ディレクトリ
            collection_name: コレクション名(FAISSファイル名の一部として使用)
            embedding_model: 埋め込みモデル名
            openai_api_base: OpenAI互換APIのベースURL(オプション)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # ベースデータディレクトリも作成しておく
        self.persist_directory.parent.mkdir(parents=True, exist_ok=True)
        # ベクトルDBディレクトリも作成
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # 埋め込み関数の初期化
        if openai_api_base:
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                base_url=openai_api_base,
            )
        else:
            self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # FAISSインデックスファイルのパス
        self.index_path = self.persist_directory / f'{collection_name}.faiss'
        self.pkl_path = self.persist_directory / f'{collection_name}.pkl'

    def _load_vectorstore(self) -> FAISS | None:  # noqa: PLR0911
        """既存のFAISSベクトルストアを読み込む.

        Returns:
            読み込まれたFAISSベクトルストア、存在しない場合はNone
        """
        # ファイルが存在しない場合は None を返す
        if not (self.index_path.exists() and self.pkl_path.exists()):
            return None

        # ファイルが存在する場合は読み込みを試行、失敗時も None を返す
        try:
            return FAISS.load_local(
                str(self.persist_directory),
                self.embeddings,
                index_name=self.collection_name,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            # ファイルが破損している場合などは None を返して新規作成を促す
            return None

    def _create_new_vectorstore(self, documents: list[Document]) -> FAISS:
        """新規ベクトルストアを作成する.

        Args:
            documents: 初期ドキュメントのリスト

        Returns:
            作成されたFAISSベクトルストア
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)

    def _add_to_existing_vectorstore(
        self, vectorstore: FAISS, documents: list[Document], batch_size: int
    ) -> None:
        """既存のベクトルストアにドキュメントを追加する.

        Args:
            vectorstore: 既存のFAISSベクトルストア
            documents: 追加するドキュメントのリスト
            batch_size: バッチ処理のサイズ
        """
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            vectorstore.add_documents(batch)

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
            OperationFailedError: ドキュメント追加に失敗した場合
        """
        if not documents:
            return

        try:
            vectorstore = self._load_vectorstore()

            if vectorstore is None:
                vectorstore = self._create_new_vectorstore(documents)
            else:
                self._add_to_existing_vectorstore(vectorstore, documents, batch_size)

            # FAISSインデックスを保存
            vectorstore.save_local(str(self.persist_directory), index_name=self.collection_name)

        except Exception as e:
            raise OperationFailedError(operation='ドキュメント追加', error=str(e)) from e

    def collection_exists(self) -> bool:
        """コレクションが存在するかチェックする.

        Returns:
            コレクションが存在する場合True
        """
        return self.index_path.exists() and self.pkl_path.exists()

    def delete_collection(self) -> None:
        """コレクションを削除する.

        Raises:
            OperationFailedError: コレクション削除に失敗した場合
        """
        try:
            if self.index_path.exists():
                self.index_path.unlink()
            if self.pkl_path.exists():
                self.pkl_path.unlink()
        except Exception as e:
            raise OperationFailedError(operation='コレクション削除', error=str(e)) from e

    def get_collection_count(self) -> int:
        """コレクション内のドキュメント数を取得する.

        Returns:
            ドキュメント数

        Raises:
            OperationFailedError: カウント取得に失敗した場合
        """
        try:
            if not self.collection_exists():
                return 0

            vectorstore = self._load_vectorstore()
            return vectorstore.index.ntotal if vectorstore else 0

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
            OperationFailedError: 検索に失敗した場合
        """
        try:
            vectorstore = self._load_vectorstore()
            if vectorstore is None:
                return []  # ベクトルストアが存在しない場合は空のリストを返す
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
            OperationFailedError: 検索に失敗した場合
        """
        try:
            vectorstore = self._load_vectorstore()
            if vectorstore is None:
                return []  # ベクトルストアが存在しない場合は空のリストを返す
            return vectorstore.similarity_search_with_score(query, k=k)

        except Exception as e:
            raise OperationFailedError(operation='スコア付き類似度検索', error=str(e)) from e
