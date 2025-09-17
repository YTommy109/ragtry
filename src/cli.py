"""CLI(コマンドラインインターフェース)モジュール.

Clickライブラリを使用してコマンドライン操作を提供する。
"""

import os
import sys
from typing import Any

import click

from src.config import Config, bootstrap_config
from src.rag_service import RAGService

# Constants
MAX_PREVIEW_CHARS = 200


def _handle_existing_collection(info: dict[str, Any]) -> bool:
    """既存コレクションの処理を行う."""
    if info['collection_exists']:
        click.echo(f'既存のコレクションを発見しました(文書数: {info["document_count"]})')
        if not click.confirm('既存のデータを削除して再構築しますか?'):
            click.echo('構築をキャンセルしました。')
            return False
    return True


def _build_knowledge_base_with_progress(rag_service: RAGService) -> None:
    """プログレスバー付きでナレッジベースを構築する."""
    with click.progressbar(length=100, label='PDF処理中') as bar:
        bar.update(20)
        rag_service.build_knowledge_base()
        bar.update(80)


def _check_collection_existence(rag_service: RAGService, env: str) -> bool:
    """コレクションの存在確認を行う."""
    info = rag_service.get_collection_info()
    if not info['collection_exists'] or info['document_count'] == 0:
        click.echo(
            'ベクトルデータベースが存在しません。先に build コマンドを実行してください。',
            err=True,
        )
        click.echo(f'実行例: python main.py build --env {env}', err=True)
        return False
    return True


def _search_documents_with_progress(
    rag_service: RAGService, question: str, search_k: int
) -> list[Any]:
    """プログレスバー付きで文書検索を行う."""
    with click.progressbar(length=100, label='検索中') as bar:
        bar.update(30)
        documents = rag_service.search_similar_documents(question, k=search_k)
        bar.update(70)
    return documents


def _generate_response_with_progress(rag_service: RAGService, question: str) -> str:
    """プログレスバー付きで回答を生成する."""
    with click.progressbar(length=100, label='回答生成中') as bar:
        bar.update(50)
        response = rag_service.generate_response(question)
        bar.update(50)
    return response


def _display_response(response: str) -> None:
    """回答を表示する."""
    click.echo('\n' + '=' * 60)
    click.echo('【回答】')
    click.echo('=' * 60)
    click.echo(response)
    click.echo('=' * 60)


def _display_reference_documents(documents: list[Any]) -> None:
    """参考文書を表示する."""
    if click.confirm('\n参考にした文書を表示しますか?'):
        click.echo('\n【参考文書】')
        for i, doc in enumerate(documents, 1):
            page_info = (
                f' (ページ{doc.metadata.get("page", "不明")})' if 'page' in doc.metadata else ''
            )
            click.echo(f'\n[文書{i}]{page_info}:')
            content = doc.page_content[:MAX_PREVIEW_CHARS]
            if len(doc.page_content) > MAX_PREVIEW_CHARS:
                content += '...'
            click.echo(content)


def _display_system_config(config: Config, env: str) -> None:
    """システム設定情報を表示する."""
    click.echo('【システム情報】')
    click.echo(f'実行環境: {env}')
    click.echo(f'LLMモデル: {config.llm_model}')
    click.echo(f'埋め込みモデル: {config.embedding_model}')
    click.echo(f'チャンクサイズ: {config.chunk_size}文字')
    click.echo(f'オーバーラップ: {config.chunk_overlap}文字')
    click.echo(f'検索文書数: {config.search_k}件')
    click.echo(f'ベクトルDB: {config.faiss_persist_directory}')


def _display_database_info(collection_info: dict[str, Any]) -> None:
    """データベース情報を表示する."""
    click.echo('\n【データベース情報】')
    click.echo(f'コレクション存在: {"はい" if collection_info["collection_exists"] else "いいえ"}')
    click.echo(f'文書数: {collection_info["document_count"]}件')

    if collection_info['collection_exists'] and collection_info['document_count'] > 0:
        click.echo('\n✓ システムは準備完了です')
    else:
        click.echo('\n⚠ ベクトルデータベースが未構築です')
        click.echo('  build コマンドを実行してください')


@click.group()
@click.option(
    '--env',
    type=click.Choice(['dev', 'test', 'prod'], case_sensitive=False),
    default=None,
    help='実行環境を指定します (dev, test, prod)',
)
@click.pass_context
def cli(ctx: click.Context, env: str | None) -> None:
    """スクラムガイド拡張パック RAGシステム.

    スクラムガイド拡張パック(2025.6版)を基にした質問応答システムです。
    """
    # コンテキストに環境情報を保存
    ctx.ensure_object(dict)
    ctx.obj['env'] = env or 'dev'


@cli.command()
@click.pass_context
def build(ctx: click.Context) -> None:
    """ベクトルデータベースを構築します.

    スクラムガイド拡張パックPDFをダウンロードし、
    テキストを抽出してベクトルデータベースを構築します。
    """
    env = ctx.obj['env']

    try:
        click.echo(f'環境: {env}')
        click.echo('ベクトルデータベースの構築を開始します...')

        rag_service = RAGService()

        info = rag_service.get_collection_info()
        if not _handle_existing_collection(info):
            return

        _build_knowledge_base_with_progress(rag_service)

        final_info = rag_service.get_collection_info()
        click.echo(f'✓ 構築完了: {final_info["document_count"]}個の文書チャンクを処理しました')

    except Exception as e:
        click.echo(f'エラー: {e}', err=True)
        sys.exit(1)


def _process_query_request(
    rag_service: RAGService, question: str, search_k: int
) -> tuple[str, list[Any]] | None:
    """クエリリクエストを処理し、回答と文書を返す."""
    documents = _search_documents_with_progress(rag_service, question, search_k)

    if not documents:
        click.echo('\n関連する文書が見つかりませんでした。')
        click.echo('より具体的な質問をお試しください。')
        return None

    response = _generate_response_with_progress(rag_service, question)
    return response, documents


def _prepare_service(env: str) -> tuple[RAGService, Config]:
    """環境に応じたサービスと設定を準備する."""
    # CLIで指定された環境を環境変数に設定
    os.environ['ENV'] = env
    # 設定を再初期化
    config_instance = bootstrap_config()
    service = RAGService(config_instance)
    return service, config_instance


@cli.command()
@click.argument('question')
@click.option('--k', type=int, default=None, help='検索する関連文書の数(デフォルト: 設定値を使用)')
@click.pass_context
def query(ctx: click.Context, question: str, k: int | None) -> None:  # noqa: PLR0915
    """質問に対して回答を生成します.

    QUESTION: 質問内容(例: "スクラムとは何ですか?")
    """
    env = ctx.obj['env']

    try:
        rag_service, config = _prepare_service(env)

        if not _check_collection_existence(rag_service, env):
            sys.exit(1)

        click.echo(f'質問: {question}')
        click.echo('回答を生成中...')

        search_k = k if k is not None else config.search_k
        result = _process_query_request(rag_service, question, search_k)

        if result is None:
            return

        response, documents = result
        _display_response(response)
        _display_reference_documents(documents)

    except Exception as e:
        click.echo(f'エラー: {e}', err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """システム情報を表示します."""
    env = ctx.obj['env']

    try:
        # CLIで指定された環境を環境変数に設定
        os.environ['ENV'] = env
        # 設定を再初期化
        config_instance = bootstrap_config()
        _display_system_config(config_instance, env)

        rag_service = RAGService(config_instance)
        collection_info = rag_service.get_collection_info()
        _display_database_info(collection_info)

    except Exception as e:
        click.echo(f'エラー: {e}', err=True)
        sys.exit(1)


@cli.command()
def version() -> None:
    """バージョン情報を表示します."""
    click.echo('スクラムガイド拡張パック RAGシステム v1.0.0')
    click.echo('対象文書: スクラムガイド拡張パック(2025.6版)')


if __name__ == '__main__':  # pragma: no cover
    cli()
