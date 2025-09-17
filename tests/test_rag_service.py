"""rag_service モジュールのテスト."""

import pytest
from langchain.schema import Document
from pytest_mock import MockerFixture

from src.config import config
from src.exceptions import OperationFailedError
from src.rag_service import RAGService


class TestRAGサービス:
    def test_初期化(self, mocker: MockerFixture) -> None:
        mock_vector_store = mocker.patch('src.rag_service.VectorStore')
        mock_chat_openai = mocker.patch('src.rag_service.ChatOpenAI')

        rag_service = RAGService()

        # 属性の確認
        assert rag_service.config == config

        # VectorStoreの初期化確認
        mock_vector_store.assert_called_once_with(
            persist_directory=config.chroma_persist_directory,
            embedding_model=config.embedding_model,
            openai_api_base=config.openai_api_base,
        )

        # ChatOpenAIの初期化確認
        if config.openai_api_base:
            mock_chat_openai.assert_called_once_with(
                model=config.llm_model,
                api_key=config.openai_api_key,
                temperature=0.1,
                base_url=config.openai_api_base,
            )
        else:
            mock_chat_openai.assert_called_once_with(
                model=config.llm_model,
                api_key=config.openai_api_key,
                temperature=0.1,
            )

    def test_検索k未指定で文書を取得できる(self, mocker: MockerFixture) -> None:
        # モックの設定
        mock_vector_store = mocker.patch('src.rag_service.VectorStore')
        _ = mocker.patch('src.rag_service.ChatOpenAI')

        mock_vector_store_instance = mocker.Mock()
        mock_documents = [Document(page_content='test', metadata={})]
        mock_vector_store_instance.similarity_search.return_value = mock_documents
        mock_vector_store.return_value = mock_vector_store_instance

        rag_service = RAGService()
        rag_service.vector_store = mock_vector_store_instance

        # テスト実行
        result = rag_service.search_similar_documents('test query')

        # 検証
        assert result == mock_documents
        mock_vector_store_instance.similarity_search.assert_called_once_with(
            'test query', k=config.search_k
        )

    def test_検索k指定で文書を取得できる(self, mocker: MockerFixture) -> None:
        # モックの設定
        mock_vector_store = mocker.patch('src.rag_service.VectorStore')
        _ = mocker.patch('src.rag_service.ChatOpenAI')

        mock_vector_store_instance = mocker.Mock()
        mock_documents = [Document(page_content='test', metadata={})]
        mock_vector_store_instance.similarity_search.return_value = mock_documents
        mock_vector_store.return_value = mock_vector_store_instance

        rag_service = RAGService()
        rag_service.vector_store = mock_vector_store_instance

        # テスト実行
        result = rag_service.search_similar_documents('test query', k=2)

        # 検証
        assert result == mock_documents
        mock_vector_store_instance.similarity_search.assert_called_once_with('test query', k=2)

    def test_検索が失敗する(self, mocker: MockerFixture) -> None:
        # モックの設定
        mock_vector_store = mocker.patch('src.rag_service.VectorStore')
        _ = mocker.patch('src.rag_service.ChatOpenAI')

        mock_vector_store_instance = mocker.Mock()
        mock_vector_store_instance.similarity_search.side_effect = Exception('Search failed')
        mock_vector_store.return_value = mock_vector_store_instance

        rag_service = RAGService()
        rag_service.vector_store = mock_vector_store_instance

        # テスト実行と検証
        with pytest.raises(OperationFailedError):
            rag_service.search_similar_documents('test query')

    def test_コンテキストが空である(self, mocker: MockerFixture) -> None:
        mocker.patch('src.rag_service.VectorStore')
        mocker.patch('src.rag_service.ChatOpenAI')
        rag_service = RAGService()
        result = rag_service.build_context([])
        assert result == ''

    def test_文書からコンテキストを生成できる(self, mocker: MockerFixture) -> None:
        mocker.patch('src.rag_service.VectorStore')
        mocker.patch('src.rag_service.ChatOpenAI')
        rag_service = RAGService()

        documents = [
            Document(page_content='スクラムの基本', metadata={'page': 1}),
            Document(page_content='スプリントについて', metadata={'page': 2}),
            Document(page_content='役割の説明', metadata={}),  # pageなし
        ]

        result = rag_service.build_context(documents)

        expected = (
            '[文書1] (ページ1): スクラムの基本\n\n'
            '[文書2] (ページ2): スプリントについて\n\n'
            '[文書3]: 役割の説明'
        )
        assert result == expected

    def test_コンテキストなしでプロンプトを生成できる(self, mocker: MockerFixture) -> None:
        mocker.patch('src.rag_service.VectorStore')
        mocker.patch('src.rag_service.ChatOpenAI')
        rag_service = RAGService()

        result = rag_service.create_prompt('テスト質問', '')

        assert 'スクラムガイド拡張パックに関する質問に答えてください' in result
        assert 'テスト質問' in result
        assert '関連する情報が見つかりませんでした' in result

    def test_コンテキストありでプロンプトを生成できる(self, mocker: MockerFixture) -> None:
        mocker.patch('src.rag_service.VectorStore')
        mocker.patch('src.rag_service.ChatOpenAI')
        rag_service = RAGService()

        context = '[文書1]: スクラムの基本説明'
        result = rag_service.create_prompt('スクラムとは何ですか', context)

        assert 'スクラムガイド拡張パック(2025.6版)の専門家です' in result
        assert '参考文書の内容に基づいて回答してください' in result
        assert 'スクラムの基本説明' in result
        assert 'スクラムとは何ですか' in result

    def test_回答を生成できる(self, mocker: MockerFixture) -> None:
        # モックの設定
        mock_vector_store = mocker.patch('src.rag_service.VectorStore')
        mock_chat_openai = mocker.patch('src.rag_service.ChatOpenAI')

        mock_vector_store_instance = mocker.Mock()
        mock_documents = [Document(page_content='スクラム説明', metadata={})]
        mock_vector_store_instance.similarity_search.return_value = mock_documents
        mock_vector_store.return_value = mock_vector_store_instance

        mock_llm_instance = mocker.Mock()
        mock_response = mocker.Mock()
        mock_response.content = 'スクラムは...'
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm_instance

        rag_service = RAGService()
        rag_service.vector_store = mock_vector_store_instance
        rag_service.llm = mock_llm_instance

        # テスト実行
        result = rag_service.generate_response('スクラムとは何ですか')

        # 検証
        assert result == 'スクラムは...'
        mock_vector_store_instance.similarity_search.assert_called_once()
        mock_llm_instance.invoke.assert_called_once()

    def test_content属性が無いレスポンスから回答を取得できる(self, mocker: MockerFixture) -> None:
        # モックの設定
        mock_vector_store = mocker.patch('src.rag_service.VectorStore')
        mock_chat_openai = mocker.patch('src.rag_service.ChatOpenAI')

        mock_vector_store_instance = mocker.Mock()
        mock_documents = [Document(page_content='test', metadata={})]
        mock_vector_store_instance.similarity_search.return_value = mock_documents
        mock_vector_store.return_value = mock_vector_store_instance

        mock_llm_instance = mocker.Mock()
        mock_response = 'レスポンス文字列'
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm_instance

        rag_service = RAGService()
        rag_service.vector_store = mock_vector_store_instance
        rag_service.llm = mock_llm_instance

        # テスト実行
        result = rag_service.generate_response('質問')

        # 検証
        assert result == 'レスポンス文字列'

    def test_回答生成が失敗する(self, mocker: MockerFixture) -> None:
        # モックの設定
        mock_vector_store = mocker.patch('src.rag_service.VectorStore')
        _ = mocker.patch('src.rag_service.ChatOpenAI')

        mock_vector_store_instance = mocker.Mock()
        mock_vector_store_instance.similarity_search.side_effect = Exception('Search failed')
        mock_vector_store.return_value = mock_vector_store_instance

        rag_service = RAGService()
        rag_service.vector_store = mock_vector_store_instance

        # テスト実行と検証
        with pytest.raises(OperationFailedError):
            rag_service.generate_response('質問')

    def test_コレクション情報を取得できる(self, mocker: MockerFixture) -> None:
        # モックの設定
        mock_vector_store = mocker.patch('src.rag_service.VectorStore')
        _ = mocker.patch('src.rag_service.ChatOpenAI')

        mock_vector_store_instance = mocker.Mock()
        mock_vector_store_instance.get_collection_count.return_value = 100
        mock_vector_store_instance.collection_exists.return_value = True
        mock_vector_store.return_value = mock_vector_store_instance

        rag_service = RAGService()
        rag_service.vector_store = mock_vector_store_instance

        # テスト実行
        result = rag_service.get_collection_info()

        # 検証
        expected = {'document_count': 100, 'collection_exists': True}
        assert result == expected

    def test_コレクション情報取得が失敗する(self, mocker: MockerFixture) -> None:
        # モックの設定
        mock_vector_store = mocker.patch('src.rag_service.VectorStore')
        _ = mocker.patch('src.rag_service.ChatOpenAI')

        mock_vector_store_instance = mocker.Mock()
        mock_vector_store_instance.get_collection_count.side_effect = Exception('Error')
        mock_vector_store.return_value = mock_vector_store_instance

        rag_service = RAGService()
        rag_service.vector_store = mock_vector_store_instance

        # テスト実行
        result = rag_service.get_collection_info()

        # 検証
        expected = {'document_count': 0, 'collection_exists': False}
        assert result == expected

    def test_KBを構築できる(self, mocker: MockerFixture) -> None:
        # Arrange
        mock_document_loader = mocker.patch('src.rag_service.DocumentLoader')
        mock_vector_store = mocker.patch('src.rag_service.VectorStore')
        _ = mocker.patch('src.rag_service.ChatOpenAI')

        mock_loader_instance = mocker.Mock()
        mock_chunks = [Document(page_content='chunk1', metadata={})]
        mock_loader_instance.process_scrum_guide_pdf.return_value = mock_chunks
        mock_document_loader.return_value = mock_loader_instance

        mock_vector_store_instance = mocker.Mock()
        mock_vector_store_instance.collection_exists.return_value = True
        mock_vector_store.return_value = mock_vector_store_instance

        # Act
        rag_service = RAGService()
        rag_service.vector_store = mock_vector_store_instance

        # Act
        rag_service.build_knowledge_base()

        # Assert
        mock_vector_store_instance.delete_collection.assert_called_once()
        mock_vector_store_instance.add_documents.assert_called_once_with(mock_chunks)

    def test_既存コレクションなしでKBを構築できる(self, mocker: MockerFixture) -> None:
        # Arrange
        mock_document_loader = mocker.patch('src.rag_service.DocumentLoader')
        mock_vector_store = mocker.patch('src.rag_service.VectorStore')
        _ = mocker.patch('src.rag_service.ChatOpenAI')

        mock_loader_instance = mocker.Mock()
        mock_chunks = [Document(page_content='chunk1', metadata={})]
        mock_loader_instance.process_scrum_guide_pdf.return_value = mock_chunks
        mock_document_loader.return_value = mock_loader_instance

        mock_vector_store_instance = mocker.Mock()
        mock_vector_store_instance.collection_exists.return_value = False
        mock_vector_store.return_value = mock_vector_store_instance

        # Act
        rag_service = RAGService()
        rag_service.vector_store = mock_vector_store_instance

        # Act
        rag_service.build_knowledge_base()

        # Assert（delete_collectionは呼ばれない）
        mock_vector_store_instance.delete_collection.assert_not_called()
        mock_vector_store_instance.add_documents.assert_called_once_with(mock_chunks)

    def test_KB構築が失敗する(self, mocker: MockerFixture) -> None:
        # Arrange
        mock_document_loader = mocker.patch('src.rag_service.DocumentLoader')
        mock_vector_store = mocker.patch('src.rag_service.VectorStore')
        _ = mocker.patch('src.rag_service.ChatOpenAI')

        mock_loader_instance = mocker.Mock()
        mock_loader_instance.process_scrum_guide_pdf.side_effect = Exception('PDF error')
        mock_document_loader.return_value = mock_loader_instance

        mock_vector_store_instance = mocker.Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        # Act
        rag_service = RAGService()
        rag_service.vector_store = mock_vector_store_instance

        # Assert
        with pytest.raises(OperationFailedError) as exc_info:
            rag_service.build_knowledge_base()

        assert 'ナレッジベース構築' in str(exc_info.value)


class TestRAGサービス統合:
    def test_統合フローが動作する(self, mocker: MockerFixture) -> None:
        mocker.patch('src.rag_service.VectorStore')
        mocker.patch('src.rag_service.ChatOpenAI')
        # RAGServiceの初期化
        rag_service = RAGService()

        # 設定が正しく保存されていることを確認（パスの詳細には依存しない）
        assert rag_service.config == config
