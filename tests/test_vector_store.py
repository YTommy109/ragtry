"""vector_store のテスト."""

import os
import tempfile
from pathlib import Path

import pytest
from langchain.schema import Document
from pytest_mock import MockerFixture

from src.exceptions import OperationFailedError
from src.vector_store import VectorStore


class TestVectorStore:
    """ベクターストアクラスのテスト."""

    def setup_method(self) -> None:
        """テストセットアップ."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def test_初期化できる(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings
        mock_embeddings = mocker.patch('src.vector_store.OpenAIEmbeddings')
        mock_embeddings.return_value = mocker.Mock()
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        assert vector_store.persist_directory == self.temp_dir
        assert vector_store.collection_name == 'scrum_guide_collection'
        assert vector_store.embedding_model == 'text-embedding-3-small'
        assert self.temp_dir.exists()

    def test_カスタム設定で初期化できる(self, mocker: MockerFixture) -> None:
        custom_dir = self.temp_dir / 'custom'

        mock_embeddings = mocker.patch('src.vector_store.OpenAIEmbeddings')
        mock_embeddings.return_value = mocker.Mock()
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(
            persist_directory=custom_dir,
            collection_name='custom_collection',
            embedding_model='text-embedding-ada-002',
        )

        assert vector_store.persist_directory == custom_dir
        assert vector_store.collection_name == 'custom_collection'
        assert vector_store.embedding_model == 'text-embedding-ada-002'
        assert custom_dir.exists()

    def test_ベクトルストアを初期化できる(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        mock_chroma = mocker.patch('src.vector_store.Chroma')
        mock_vectorstore = mocker.Mock()
        mock_chroma.return_value = mock_vectorstore

        result = vector_store.initialize_vectorstore()

        assert result == mock_vectorstore
        # Chromaの呼び出しを確認
        mock_chroma.assert_called_once()
        call_args = mock_chroma.call_args
        assert call_args.kwargs['collection_name'] == 'scrum_guide_collection'
        assert call_args.kwargs['embedding_function'] == vector_store.embeddings
        assert call_args.kwargs['persist_directory'] == str(self.temp_dir)

    def test_ベクトルストア初期化が失敗する(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        mock_chroma = mocker.patch('src.vector_store.Chroma')
        mock_chroma.side_effect = Exception('Chroma error')

        with pytest.raises(OperationFailedError) as exc_info:
            vector_store.initialize_vectorstore()

        assert 'ベクトルストア初期化' in str(exc_info.value)

    def test_ドキュメントを追加できる(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        # モックの設定
        mock_init = mocker.patch.object(VectorStore, 'initialize_vectorstore')
        mock_vectorstore = mocker.Mock()
        mock_init.return_value = mock_vectorstore

        documents = [
            Document(page_content='test1', metadata={}),
            Document(page_content='test2', metadata={}),
        ]

        # テスト実行
        vector_store.add_documents(documents)

        # 検証
        mock_vectorstore.add_documents.assert_called_once_with(documents)
        # Note: persist() is not called in newer langchain-chroma (auto-persists)

    def test_空リストは追加処理を行わない(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        # モックの設定
        mock_init = mocker.patch.object(VectorStore, 'initialize_vectorstore')

        # テスト実行
        vector_store.add_documents([])

        # 検証（初期化すら呼ばれない）
        mock_init.assert_not_called()

    def test_ドキュメントをバッチ追加できる(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        # モックの設定
        mock_init = mocker.patch.object(VectorStore, 'initialize_vectorstore')
        mock_vectorstore = mocker.Mock()
        mock_init.return_value = mock_vectorstore

        # 150個のドキュメントを作成（バッチサイズ100を超える）
        documents = [Document(page_content=f'test{i}', metadata={}) for i in range(150)]

        # テスト実行
        vector_store.add_documents(documents, batch_size=100)

        # 検証（2回に分けて呼ばれる）
        assert mock_vectorstore.add_documents.call_count == 2
        # 1回目: 100個
        first_call_args = mock_vectorstore.add_documents.call_args_list[0][0][0]
        assert len(first_call_args) == 100
        # 2回目: 50個
        second_call_args = mock_vectorstore.add_documents.call_args_list[1][0][0]
        assert len(second_call_args) == 50

    def test_コレクションが存在しない(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        result = vector_store.collection_exists()
        assert result is False

    def test_コレクションが存在すると削除される(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        mock_exists = mocker.patch.object(VectorStore, 'collection_exists')
        mock_exists.return_value = True

        mock_delete = mocker.patch.object(vector_store.client, 'delete_collection')
        vector_store.delete_collection()
        mock_delete.assert_called_once_with('scrum_guide_collection')

    def test_コレクションが無いと削除されない(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        mock_exists = mocker.patch.object(VectorStore, 'collection_exists')
        mock_exists.return_value = False

        mock_delete = mocker.patch.object(vector_store.client, 'delete_collection')
        vector_store.delete_collection()
        mock_delete.assert_not_called()

    def test_コレクションが無いとカウント0(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        result = vector_store.get_collection_count()
        assert result == 0

    def test_類似度検索で文書を取得できる(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        # モックの設定
        mock_init = mocker.patch.object(VectorStore, 'initialize_vectorstore')
        mock_vectorstore = mocker.Mock()
        mock_results = [Document(page_content='result1', metadata={})]
        mock_vectorstore.similarity_search.return_value = mock_results
        mock_init.return_value = mock_vectorstore

        # テスト実行
        result = vector_store.similarity_search('query', k=1)

        # 検証
        assert result == mock_results
        mock_vectorstore.similarity_search.assert_called_once_with('query', k=1)

    def test_スコア付き検索で文書を取得できる(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        # モックの設定
        mock_init = mocker.patch.object(VectorStore, 'initialize_vectorstore')
        mock_vectorstore = mocker.Mock()
        mock_doc = Document(page_content='result1', metadata={})
        mock_results = [(mock_doc, 0.9)]
        mock_vectorstore.similarity_search_with_score.return_value = mock_results
        mock_init.return_value = mock_vectorstore

        # テスト実行
        result = vector_store.similarity_search_with_score('query', k=1)

        # 検証
        assert result == mock_results
        mock_vectorstore.similarity_search_with_score.assert_called_once_with('query', k=1)


class TestVectorStoreIntegration:
    """ベクターストアの統合テスト."""

    def setup_method(self) -> None:
        """テストセットアップ."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def test_ディレクトリを作成できる(self, mocker: MockerFixture) -> None:
        """ディレクトリ作成のテスト."""
        # 新しいディレクトリパスを作成
        new_dir = self.temp_dir / 'new_subdir'
        assert not new_dir.exists()

        # VectorStoreを作成
        mock_embeddings = mocker.patch('src.vector_store.OpenAIEmbeddings')
        mock_embeddings.return_value = mocker.Mock()
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        VectorStore(persist_directory=new_dir)

        # ディレクトリが作成されていることを確認
        assert new_dir.exists()
