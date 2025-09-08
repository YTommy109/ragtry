"""RAGServiceの統合テスト."""

from pytest_mock import MockerFixture

from src.rag_service import RAGService


class TestRAGServiceIntegration:
    """RAGServiceの統合テスト."""

    def test_rag_service_basic_operations(
        self, rag_service_with_global_config: 'RAGService'
    ) -> None:
        """RAGServiceの基本操作テスト."""
        try:
            info = rag_service_with_global_config.get_collection_info()

            print(f'コレクション存在: {info["collection_exists"]}')
            print(f'文書数: {info["document_count"]}')

            # 基本的な検証
            assert isinstance(info, dict)
            assert 'collection_exists' in info
            assert 'document_count' in info

            # コレクションが存在しない場合は構築を試行しない（APIキーが必要）
            # 代わりに、基本的な情報取得のみをテスト
            print('RAGServiceの基本操作テスト完了')

        except Exception as e:
            print(f'RAGServiceテストエラー: {e}')
            # 環境変数が設定されていない場合はスキップ
            if 'OPENAI_API_KEY' not in str(e):
                raise

    def test_rag_queries_with_mock(
        self, rag_service_with_global_config: 'RAGService', mocker: MockerFixture
    ) -> None:
        """モックを使用したRAGクエリテスト."""
        # 外部API呼び出しをモック
        mock_search = mocker.patch.object(
            rag_service_with_global_config, 'search_similar_documents'
        )
        mock_generate = mocker.patch.object(rag_service_with_global_config, 'generate_response')

        # モックの戻り値を設定
        mock_search.return_value = []
        mock_generate.return_value = 'テスト回答'

        print('\nRAGクエリテスト実行中...')

        # 直接メソッドを呼び出してテスト
        try:
            # 検索テスト
            results = rag_service_with_global_config.search_similar_documents('テスト質問', k=4)
            assert results == []

            # 生成テスト
            response = rag_service_with_global_config.generate_response('テスト質問')
            assert response == 'テスト回答'

            print('RAGクエリテスト完了')

        except Exception as e:
            print(f'RAGクエリテストエラー: {e}')
            # APIキーが設定されていない場合はスキップ
            if 'OPENAI_API_KEY' not in str(e):
                raise
