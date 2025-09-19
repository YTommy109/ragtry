"""ハイブリッド検索の統合テストモジュール."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain.schema import Document

from src.app_types import SearchType
from src.config import Config
from src.rag_service import RAGService


class TestHybridSearch:
    """ハイブリッド検索の統合テストクラス."""

    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリを作成するフィクスチャ."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_config(self, temp_dir):
        """モック設定を作成するフィクスチャ."""
        config = MagicMock(spec=Config)
        config.faiss_persist_directory = temp_dir / 'vector_db'
        config.bm25_persist_directory = temp_dir / 'keyword_db'
        config.search_k = 4
        config.embedding_model = 'text-embedding-3-small'
        config.openai_api_base = None
        config.llm_model = 'gpt-3.5-turbo'
        config.openai_api_key = 'test-key'
        return config

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

    def test_search_type_enum(self):
        """SearchType enumのテスト."""
        assert SearchType.SEMANTIC.value == 'semantic'
        assert SearchType.KEYWORD.value == 'keyword'
        assert SearchType.HYBRID.value == 'hybrid'

    @pytest.mark.integration
    def test_rag_service_initialization_with_hybrid_support(self, mock_config):
        """ハイブリッド検索対応のRAGService初期化テスト."""
        # OpenAIのモックを設定
        with pytest.MonkeyPatch().context() as m:
            # OpenAI関連のモックを設定
            mock_openai = MagicMock()
            m.setattr('src.rag_service.ChatOpenAI', mock_openai)
            m.setattr('src.vector_store.OpenAIEmbeddings', MagicMock())

            # RAGServiceを初期化
            rag_service = RAGService(config_instance=mock_config)

            # VectorStoreが正しく初期化されていることを確認
            assert rag_service.vector_store is not None
            assert hasattr(rag_service.vector_store, 'keyword_search')

    def test_search_similar_documents_with_different_types(self, mock_config, sample_documents):
        """異なる検索タイプでの文書検索テスト."""
        with pytest.MonkeyPatch().context() as m:
            # 必要なモックを設定
            m.setattr('src.rag_service.ChatOpenAI', MagicMock())
            m.setattr('src.vector_store.OpenAIEmbeddings', MagicMock())

            # RAGServiceを初期化
            rag_service = RAGService(config_instance=mock_config)

            # VectorStoreのメソッドをモック
            similarity_mock = MagicMock(return_value=sample_documents[:2])
            bm25_mock = MagicMock(return_value=sample_documents[1:])
            hybrid_mock = MagicMock(return_value=sample_documents)

            m.setattr(rag_service.vector_store, 'similarity_search', similarity_mock)
            m.setattr(rag_service.vector_store, 'bm25_search', bm25_mock)
            m.setattr(rag_service.vector_store, 'hybrid_search', hybrid_mock)

            # セマンティック検索
            results = rag_service.search_similar_documents(
                'スクラム', k=2, search_type=SearchType.SEMANTIC
            )
            assert len(results) == 2
            similarity_mock.assert_called_once_with('スクラム', k=2)

            # キーワード検索
            results = rag_service.search_similar_documents(
                'スクラム', k=2, search_type=SearchType.KEYWORD
            )
            assert len(results) == 2
            bm25_mock.assert_called_once_with('スクラム', k=2)

            # ハイブリッド検索
            results = rag_service.search_similar_documents(
                'スクラム', k=2, search_type=SearchType.HYBRID
            )
            assert len(results) == 3
            hybrid_mock.assert_called_once_with('スクラム', k=2)

    def test_unsupported_search_type(self, mock_config):
        """サポートされていない検索タイプのテスト."""
        with pytest.MonkeyPatch().context() as m:
            # 必要なモックを設定
            m.setattr('src.rag_service.ChatOpenAI', MagicMock())
            m.setattr('src.vector_store.OpenAIEmbeddings', MagicMock())

            # RAGServiceを初期化
            rag_service = RAGService(config_instance=mock_config)

            # 無効な検索タイプでエラーが発生することを確認
            with pytest.raises(ValueError, match='Unsupported search type'):
                rag_service._execute_search('test', 4, 'invalid_type')  # type: ignore[arg-type]
