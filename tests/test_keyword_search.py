"""BM25キーワード検索のテストモジュール."""

import tempfile
from pathlib import Path

import pytest
from langchain.schema import Document

from src.keyword_search import KeywordSearch


class TestKeywordSearch:
    """BM25キーワード検索のテストクラス."""

    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリを作成するフィクスチャ."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def keyword_search(self, temp_dir):
        """KeywordSearchインスタンスを作成するフィクスチャ."""
        return KeywordSearch(persist_directory=temp_dir, collection_name='test')

    @pytest.fixture
    def sample_documents(self):
        """サンプルドキュメントを作成するフィクスチャ."""
        return [
            Document(
                page_content='スクラムは反復的で漸進的なソフトウェア開発手法です。',
                metadata={'page': 1},
            ),
            Document(
                page_content='スプリントは1-4週間の短い期間で実行される開発サイクルです。',
                metadata={'page': 2},
            ),
            Document(
                page_content='プロダクトオーナーは製品の価値を最大化する責任があります。',
                metadata={'page': 3},
            ),
        ]

    def test_add_documents(self, keyword_search, sample_documents):
        """ドキュメント追加のテスト."""
        # ドキュメントを追加
        keyword_search.add_documents(sample_documents)

        # コレクションが存在することを確認
        assert keyword_search.collection_exists()

        # ドキュメント数を確認
        assert keyword_search.get_collection_count() == 3

    def test_search(self, keyword_search, sample_documents):
        """検索機能のテスト."""
        # ドキュメントを追加
        keyword_search.add_documents(sample_documents)

        # スクラムで検索
        results = keyword_search.search('スクラム', k=2)
        assert len(results) >= 1
        assert 'スクラム' in results[0].page_content

        # 存在しない単語で検索
        results = keyword_search.search('存在しない単語', k=2)
        assert len(results) == 0

    def test_delete_collection(self, keyword_search, sample_documents):
        """コレクション削除のテスト."""
        # ドキュメントを追加
        keyword_search.add_documents(sample_documents)
        assert keyword_search.collection_exists()

        # コレクションを削除
        keyword_search.delete_collection()
        assert not keyword_search.collection_exists()
        assert keyword_search.get_collection_count() == 0

    def test_tokenize(self, keyword_search):
        """トークン化のテスト."""
        text = 'スクラムは反復的で漸進的なソフトウェア開発手法です。'
        tokens = keyword_search._tokenize(text)

        assert 'スクラム' in tokens
        assert '反復的' in tokens
        assert '開発' in tokens
        assert len(tokens) > 0
