"""vector_store のテスト."""

import os
import tempfile
from pathlib import Path

from langchain.schema import Document
from pytest_mock import MockerFixture

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

    def test_ベクトルストアを読み込める(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        # FAISSの読み込みをモック（存在しないケース）
        mock_load = mocker.patch.object(vector_store, '_load_vectorstore')
        mock_load.return_value = None

        result = vector_store._load_vectorstore()

        assert result is None

    def test_ベクトルストア読み込みが失敗する(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        # ファイルが存在することをモック
        mock_path = mocker.Mock()
        mock_path.exists.return_value = True
        vector_store.index_path = mock_path
        vector_store.pkl_path = mock_path

        # FAISS.load_localでエラーを発生させる
        mock_faiss = mocker.patch('src.vector_store.FAISS')
        mock_faiss.load_local.side_effect = Exception('FAISS error')

        result = vector_store._load_vectorstore()

        assert result is None

    def test_ドキュメントを追加できる(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        # 新規作成のケース
        mock_load = mocker.patch.object(vector_store, '_load_vectorstore')
        mock_load.return_value = None
        mock_create = mocker.patch.object(vector_store, '_create_new_vectorstore')
        mock_vectorstore = mocker.Mock()
        mock_create.return_value = mock_vectorstore

        documents = [
            Document(page_content='test1', metadata={}),
            Document(page_content='test2', metadata={}),
        ]

        # テスト実行
        vector_store.add_documents(documents)

        # 検証
        mock_create.assert_called_once_with(documents)
        mock_vectorstore.save_local.assert_called_once_with(
            str(self.temp_dir), index_name='scrum_guide_collection'
        )

    def test_空リストは追加処理を行わない(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        # モックの設定
        mock_load = mocker.patch.object(vector_store, '_load_vectorstore')

        # テスト実行
        vector_store.add_documents([])

        # 検証（読み込みすら呼ばれない）
        mock_load.assert_not_called()

    def test_ドキュメントをバッチ追加できる(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        # 既存のベクトルストアがあるケース
        mock_load = mocker.patch.object(vector_store, '_load_vectorstore')
        mock_vectorstore = mocker.Mock()
        mock_load.return_value = mock_vectorstore
        mock_add_to_existing = mocker.patch.object(vector_store, '_add_to_existing_vectorstore')

        # 150個のドキュメントを作成（バッチサイズ100を超える）
        documents = [Document(page_content=f'test{i}', metadata={}) for i in range(150)]

        # テスト実行
        vector_store.add_documents(documents, batch_size=100)

        # 検証
        mock_add_to_existing.assert_called_once_with(mock_vectorstore, documents, 100)
        mock_vectorstore.save_local.assert_called_once_with(
            str(self.temp_dir), index_name='scrum_guide_collection'
        )

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

        # ファイルを作成してコレクションが存在する状態にする
        vector_store.index_path.touch()
        vector_store.pkl_path.touch()

        vector_store.delete_collection()

        # ファイルが削除されていることを確認
        assert not vector_store.index_path.exists()
        assert not vector_store.pkl_path.exists()

    def test_コレクションが無いと削除処理は何もしない(self, mocker: MockerFixture) -> None:
        # Mock OpenAI embeddings and environment
        mocker.patch('src.vector_store.OpenAIEmbeddings')
        mocker.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
        vector_store = VectorStore(persist_directory=self.temp_dir)

        # ファイルが存在しない状態で削除を実行
        vector_store.delete_collection()

        # エラーが発生しないことを確認（正常終了）
        assert True

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
        mock_load = mocker.patch.object(vector_store, '_load_vectorstore')
        mock_vectorstore = mocker.Mock()
        mock_results = [Document(page_content='result1', metadata={})]
        mock_vectorstore.similarity_search.return_value = mock_results
        mock_load.return_value = mock_vectorstore

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
        mock_load = mocker.patch.object(vector_store, '_load_vectorstore')
        mock_vectorstore = mocker.Mock()
        mock_doc = Document(page_content='result1', metadata={})
        mock_results = [(mock_doc, 0.9)]
        mock_vectorstore.similarity_search_with_score.return_value = mock_results
        mock_load.return_value = mock_vectorstore

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
