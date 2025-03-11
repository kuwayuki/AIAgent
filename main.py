import sys  # Python標準ライブラリsysをインポート
from utils import (
    sample,
    WORKFLOW,
)  # utilsモジュールからsample関数とWORKFLOW Enumをインポート

# 実行時の引数を取得（スクリプト名の次の引数を使用）。指定されなければデフォルト値 "ブロックチェーン" を使用
topic = sys.argv[1] if len(sys.argv) > 1 else "ブロックチェーン"

# 指定されたワークフローとトピックを用いてsample関数を実行
sample(WORKFLOW.IMAGE, topic)
