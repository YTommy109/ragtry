[private]
@default: help


# show help message
@help:
    echo "Usage: just <recipe>"
    echo ""
    just --list

# アプリケーションの実行
# just start      -> 開発環境で実行（
# just start test -> テスト環境で実行
start env='':
    #!/usr/bin/env zsh
    set -euo pipefail

    case '{{env}}' in
        '')
            echo "Running in development mode..."
            python main.py -- --app-env dev
            ;;
        'test')
            echo "Running in test mode..."
            python main.py --server.port -- --app-env test
            ;;
        'prod')
            echo "Running in production mode..."
            python main.py -- --app-env prod
            ;;
        *)
            echo "Unknown environment: '{{env}}'. Available: 'test'"
            exit 1
            ;;
    esac

# 統合テストランナー
# just test     -> pytest --testmon (変更の影響範囲のみテスト)
# just test ut  -> ユニットテスト (変更の影響範囲のみ)
# just test ci  -> CIテスト (変更の影響範囲のみ)
# just test all -> 全件テスト
test suite='':
    #!/usr/bin/env zsh
    set -euo pipefail

    case '{{suite}}' in
        '')
            echo "Running affected tests only..."
            pytest --testmon
            ;;
        'ut')
            echo "Running affected unit tests (excluding CI tests)..."
            pytest --testmon -m "not ci" tests
            ;;
        'ci')
            echo "Running affected CI tests..."
            pytest --testmon -m "ci" tests/ci
            ;;
        'all')
            echo "Running all tests..."
            pytest
            ;;
        *)
            echo "Unknown test suite: '{{suite}}'. Available: 'ut', 'ci', 'all'"
            exit 1
            ;;
    esac

# カバレッジ計測
coverage:
    pytest --cov=src --cov-report=term-missing

# ruff
ruff path='':
    #!/usr/bin/env zsh
    set -euo pipefail

    if [[ '{{path}}' == '' ]]; then
        ruff format . && ruff check --fix .
    else
        ruff format {{path}} && ruff check --fix {{path}}
    fi


# linter/formatter
lint:
    ruff format . && ruff check --fix .
    just vulture
    mypy .

# 'prod'が指定された場合は本番環境の、それ以外は開発環境の依存関係を同期する
sync mode='':
    #!/usr/bin/env zsh
    set -euo pipefail

    if [[ '{{mode}}' == 'prod' ]]; then
        echo "Syncing production dependencies..."
        uv sync
    else
        echo "Syncing development dependencies..."
        uv sync --group dev
    fi

# 仮想環境の作成
venv:
    uv venv -p 3.12

# すべてのチェックとテストを実行
# just test-all        -> lint, test
test-all:
    #!/usr/bin/env zsh
    set -euo pipefail

    echo "Running lint..."
    just lint
    echo "Running tests..."
    just coverage


@watch:
    fswatch -o app tests | xargs -n1 -I{} just test

playwright-mcp:
    npx @playwright/mcp

mdlint path='':
    #!/usr/bin/env zsh
    set -euo pipefail

    if [[ '{{path}}' == '' ]]; then
        markdownlint-cli2 --fix .
    else
        markdownlint-cli2 --fix {{path}}
    fi

# 未使用コードの検出
# 参考: https://scrapbox.io/PythonOsaka/Python%E3%81%AE%E9%9D%99%E7%9A%84%E8%A7%A3%E6%9E%90%E3%83%84%E3%83%BC%E3%83%ABVulture%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E3%81%BF%E3%82%88%E3%81%86
vulture path='':
    #!/usr/bin/env zsh
    set -euo pipefail

    if [[ '{{path}}' == '' ]]; then
        # テストを除外してアプリケーションコードのみをチェック
        vulture main.py src
    else
        vulture {{path}}
    fi

# 型チェック
# 参考: https://github.com/astral-sh/ty
ty:
    uvx ty check

# HTMLドキュメント生成
docs:
    pdoc app --html --output-dir docs
