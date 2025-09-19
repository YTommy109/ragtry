"""BM25キーワード検索モジュール.

BM25アルゴリズムを使用したキーワード検索機能を提供する。
"""

import pickle
import re
from pathlib import Path
from typing import Any

from langchain.schema import Document
from rank_bm25 import BM25Okapi

from .exceptions import OperationFailedError


class KeywordSearch:
    """BM25を使用したキーワード検索クラス."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = 'scrum_guide_collection',
    ) -> None:
        """初期化.

        Args:
            persist_directory: BM25インデックスの永続化ディレクトリ
            collection_name: コレクション名(ファイル名の一部として使用)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # BM25インデックスファイルのパス
        self.index_path = self.persist_directory / f'{collection_name}_bm25.pkl'
        self.documents_path = self.persist_directory / f'{collection_name}_docs.pkl'

        # BM25インデックスとドキュメントの初期化
        self.bm25: Any | None = None
        self.documents: list[Document] = []

    def _tokenize(self, text: str) -> list[str]:
        """テキストをトークン化する.

        Args:
            text: トークン化するテキスト

        Returns:
            トークンのリスト
        """
        # 簡単な日本語・英語対応のトークン化
        tokens = []
        min_token_length = 2

        # 英数字の単語を抽出
        english_tokens = re.findall(r'[a-zA-Z0-9]+', text)
        tokens.extend([token.lower() for token in english_tokens if len(token) >= min_token_length])

        # 日本語の文字列をn-gramで分割
        japanese_parts = re.findall(r'[ぁ-んァ-ヶ一-龯]+', text)
        for japanese_text in japanese_parts:
            # 2-gram, 3-gram, 4-gramを生成
            for n in range(2, min(len(japanese_text) + 1, 5)):
                for i in range(len(japanese_text) - n + 1):
                    token = japanese_text[i : i + n]
                    tokens.append(token)

        # 重複を除去して返す
        return list({token for token in tokens if len(token) >= min_token_length})

    def _files_exist(self) -> bool:
        """インデックスファイルが存在するかチェック."""
        return self.index_path.exists() and self.documents_path.exists()

    def _load_index(self) -> bool:
        """既存のBM25インデックスを読み込む.

        Returns:
            読み込みに成功した場合True
        """
        if not self._files_exist():
            return False

        try:
            with open(self.index_path, 'rb') as f:
                self.bm25 = pickle.load(f)  # noqa: S301

            with open(self.documents_path, 'rb') as f:
                self.documents = pickle.load(f)  # noqa: S301

        except Exception:
            # ファイルが破損している場合などは False を返す
            self.bm25 = None
            self.documents = []
            return False

        return True

    def _save_index(self) -> None:
        """BM25インデックスを保存する.

        Raises:
            OperationFailedError: 保存に失敗した場合
        """
        try:
            # ディレクトリが存在しない場合は作成
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            with open(self.index_path, 'wb') as f:
                pickle.dump(self.bm25, f)

            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)

        except Exception as e:
            raise OperationFailedError(operation='BM25インデックス保存', error=str(e)) from e

    def add_documents(self, documents: list[Document]) -> None:
        """ドキュメントをBM25インデックスに追加する.

        Args:
            documents: 追加するドキュメントのリスト

        Raises:
            OperationFailedError: ドキュメント追加に失敗した場合
        """
        if not documents:
            return

        try:
            # 既存のインデックスを読み込み
            self._load_index()

            # 新しいドキュメントを追加
            self.documents.extend(documents)

            # 全ドキュメントのテキストをトークン化
            tokenized_docs = [self._tokenize(doc.page_content) for doc in self.documents]

            # BM25インデックスを再構築
            self.bm25 = BM25Okapi(tokenized_docs)

            # インデックスを保存
            self._save_index()

        except Exception as e:
            raise OperationFailedError(operation='BM25ドキュメント追加', error=str(e)) from e

    def _prepare_search(self, query: str) -> tuple[list[str], bool]:
        """検索の準備を行う."""
        if not self._load_index() or self.bm25 is None:
            return [], False

        tokenized_query = self._tokenize(query)
        return tokenized_query, bool(tokenized_query)

    def search(self, query: str, k: int = 4) -> list[Document]:
        """BM25を使用してキーワード検索を実行する.

        Args:
            query: 検索クエリ
            k: 取得する文書数

        Returns:
            スコアの高い文書のリスト

        Raises:
            OperationFailedError: 検索に失敗した場合
        """
        try:
            tokenized_query, is_valid = self._prepare_search(query)
            if not is_valid:
                return []

            # BM25スコアを計算
            scores = self.bm25.get_scores(tokenized_query)  # type: ignore[union-attr]

            # スコアの高い順にソートしてトップkを取得
            top_indices = scores.argsort()[-k:][::-1]

            # 結果を返す
            return [self.documents[i] for i in top_indices if scores[i] > 0]

        except Exception as e:
            raise OperationFailedError(operation='BM25検索', error=str(e)) from e

    def collection_exists(self) -> bool:
        """コレクションが存在するかチェックする.

        Returns:
            コレクションが存在する場合True
        """
        return self.index_path.exists() and self.documents_path.exists()

    def delete_collection(self) -> None:
        """コレクションを削除する.

        Raises:
            OperationFailedError: コレクション削除に失敗した場合
        """
        try:
            if self.index_path.exists():
                self.index_path.unlink()
            if self.documents_path.exists():
                self.documents_path.unlink()

            # メモリ上のデータもクリア
            self.bm25 = None
            self.documents = []

        except Exception as e:
            raise OperationFailedError(operation='BM25コレクション削除', error=str(e)) from e

    def get_collection_count(self) -> int:
        """コレクション内のドキュメント数を取得する.

        Returns:
            ドキュメント数

        Raises:
            OperationFailedError: カウント取得に失敗した場合
        """
        try:
            if not self.collection_exists() or not self._load_index():
                return 0

            return len(self.documents)

        except Exception as e:
            raise OperationFailedError(operation='BM25ドキュメント数取得', error=str(e)) from e
