"""RAGサービスモジュール.

RAG (Retrieval-Augmented Generation) の実装。
"""

from typing import Any

from langchain.schema import Document
from langchain_openai import ChatOpenAI

from .config import config
from .document_loader import DocumentLoader
from .exceptions import OperationFailedError
from .vector_store import VectorStore


class RAGService:
    """RAGサービスクラス."""

    def __init__(self) -> None:
        """初期化."""
        self.config = config
        self.vector_store = VectorStore(
            persist_directory=self.config.chroma_persist_directory,
            embedding_model=self.config.embedding_model,
            openai_api_base=self.config.openai_api_base,
        )
        # ChatOpenAIの初期化
        if self.config.openai_api_base:
            self.llm = ChatOpenAI(
                model=self.config.llm_model,
                api_key=self.config.openai_api_key,  # type: ignore[arg-type]
                temperature=0.1,
                base_url=self.config.openai_api_base,
            )
        else:
            self.llm = ChatOpenAI(
                model=self.config.llm_model,
                api_key=self.config.openai_api_key,  # type: ignore[arg-type]
                temperature=0.1,
            )

    def search_similar_documents(self, query: str, k: int | None = None) -> list[Document]:
        """類似文書を検索する.

        Args:
            query: 検索クエリ
            k: 検索する文書数 (None の場合は設定値を使用)

        Returns:
            類似文書のリスト

        Raises:
            RAGServiceError: 検索に失敗した場合
        """
        try:
            search_k = k if k is not None else self.config.search_k
            result: list[Document] = self.vector_store.similarity_search(query, k=search_k)
            return result
        except Exception as e:
            raise OperationFailedError(operation='文書検索', error=str(e)) from e

    def build_context(self, documents: list[Document]) -> str:
        """文書リストからコンテキストを構築する.

        Args:
            documents: 文書リスト

        Returns:
            構築されたコンテキスト文字列
        """
        if not documents:
            return ''

        context_parts = []
        for i, doc in enumerate(documents, 1):
            page_info = f' (ページ{doc.metadata.get("page")})' if 'page' in doc.metadata else ''
            context_part = f'[文書{i}]{page_info}: {doc.page_content}'
            context_parts.append(context_part)

        return '\n\n'.join(context_parts)

    def create_prompt(self, query: str, context: str = '') -> str:
        """質問とコンテキストからプロンプトを作成する.

        Args:
            query: ユーザーの質問
            context: 検索で取得された関連文書

        Returns:
            作成されたプロンプト
        """
        if not context:
            return f"""スクラムガイド拡張パックに関する質問に答えてください。

【質問】
{query}

【回答】
申し訳ありませんが、関連する情報が見つかりませんでした。より具体的な質問をお試しください。
"""

        return f"""あなたはスクラムガイド拡張パック(2025.6版)の専門家です。
参考文書の内容に基づいて回答してください。

【参考文書】
{context}

【質問】
{query}

【回答】
"""

    def generate_response(self, query: str) -> str:
        """質問に対する回答を生成する.

        Args:
            query: ユーザーの質問

        Returns:
            生成された回答

        Raises:
            OperationFailedError: 回答生成に失敗した場合
        """
        try:
            # 関連文書を検索
            documents = self.search_similar_documents(query)

            # コンテキストを構築
            context = self.build_context(documents)

            # プロンプトを作成
            prompt = self.create_prompt(query, context)

            # LLMで回答を生成
            response = self.llm.invoke(prompt)

            # レスポンスからテキストを抽出
            if hasattr(response, 'content'):
                return str(response.content)
            return str(response)

        except Exception as e:
            raise OperationFailedError(operation='回答生成', error=str(e)) from e

    def get_collection_info(self) -> dict[str, Any]:
        """コレクション情報を取得する.

        Returns:
            コレクション情報の辞書
        """
        try:
            return {
                'document_count': self.vector_store.get_collection_count(),
                'collection_exists': self.vector_store.collection_exists(),
            }
        except Exception:
            return {'document_count': 0, 'collection_exists': False}

    def build_knowledge_base(self) -> None:
        """ナレッジベースを構築する.

        Raises:
            RAGServiceError: 構築に失敗した場合
        """
        try:
            # DocumentLoaderを初期化: DATA_DIR/raw 配下を使う
            data_dir = self.config.chroma_persist_directory.parent
            raw_dir = data_dir / 'raw'
            raw_dir.mkdir(parents=True, exist_ok=True)
            loader = DocumentLoader(data_dir=data_dir)

            # PDFを処理してチャンクを生成
            chunks = loader.process_scrum_guide_pdf()

            # 既存のコレクションを削除
            if self.vector_store.collection_exists():
                self.vector_store.delete_collection()

            # 新しいドキュメントを追加
            self.vector_store.add_documents(chunks)

        except Exception as e:
            raise OperationFailedError(operation='ナレッジベース構築', error=str(e)) from e
