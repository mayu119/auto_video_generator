#!/bin/bash

# FFmpegのパスを確認
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpegが見つかりません。インストールしてください。"
    exit 1
 fi

# VOICEVOXが起動しているか確認
if ! curl -s http://localhost:50021/version &> /dev/null; then
    echo "警告: VOICEVOX Engineが起動していないようです。"
    echo "VOICEVOXを起動して再試行してください。"
    exit 1
fi

# スクリプトファイルのパスを取得
SCRIPT_PATH="$1"

# スクリプトを実行
python src/video_generator.py "$SCRIPT_PATH" "$@"