"""統合テスト用のフィクスチャー."""

import pytest
from pytest_mock import MockerFixture

from src.config import config
from src.document_loader import DocumentLoader
from src.rag_service import RAGService
from src.vector_store import VectorStore


@pytest.fixture
def document_loader(mocker: MockerFixture) -> 'DocumentLoader':
    """DocumentLoaderインスタンスを提供する(外部通信はモック)."""
    # requests.get をモックして外部通信を防止
    mock_response = mocker.Mock()
    mock_response.content = b''
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_response)
    data_dir = config.chroma_persist_directory.parent
    return DocumentLoader(data_dir=data_dir)


@pytest.fixture
def vector_store() -> 'VectorStore':
    """VectorStoreインスタンスを提供する."""
    return VectorStore(
        persist_directory=config.chroma_persist_directory,
        embedding_model=config.embedding_model,
    )


@pytest.fixture
def rag_service() -> 'RAGService':
    """RAGServiceインスタンスを提供する."""
    return RAGService()


@pytest.fixture
def rag_service_with_global_config() -> 'RAGService':
    """グローバルconfigを使用するRAGServiceインスタンスを提供する."""
    return RAGService()
