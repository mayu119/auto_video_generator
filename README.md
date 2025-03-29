# 自動動画生成プログラム - M4 Pro 最適化版

このプログラムは、JSONファイルから自動的に上下交互のテロップ動画を生成するツールです。Apple M4 Pro CPUおよびNeural Engineを活用した高性能な動画生成システムです。

## プロジェクト構造

```
auto_video_generator/
├── src/                   # ソースコード
│   └── video_generator.py # メインプログラム
├── data/                  # 台本データ
│   └── learning_efficiency_script.json
├── output/                # 生成された動画の出力先
├── assets/                # アセット（フォントなど）
├── requirements.txt       # 依存ライブラリ
└── README.md              # このファイル
```

## M4 Pro向け最適化点

- **マルチコア活用**: 14コアCPUを最大限に活用するための並列処理の実装
- **Neural Engine最適化**: ビデオエンコーディングにNeural Engineを活用
- **ハードウェアアクセラレーション**: VideoToolbox（Apple Siliconのハードウェアエンコーダ）を使用
- **メモリ効率**: 48GBメモリを活用した効率的なバッチ処理
- **SSD最適化**: 一時ファイルの効率的な読み書き