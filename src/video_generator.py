#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自動動画生成プログラム - M4 Pro 最適化版
JSONファイルから上下交互にテロップが表示される動画を自動生成します。
Apple M4 Pro CPUおよびNeural Engineを活用した並列処理と最適化を実装。
"""

import json
import os
import sys
import time
import argparse
import requests
import subprocess
import multiprocessing
import concurrent.futures
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import tempfile
import shutil
import wave
import contextlib
import logging
import numpy as np
from datetime import datetime

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_generator.log')
    ]
)
logger = logging.getLogger('VideoGenerator')

# デフォルト設定
DEFAULT_FONT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts", "keifont.ttf")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
DEFAULT_TEMP_DIR = os.path.join(tempfile.gettempdir(), "video_generator_temp")
DEFAULT_VOICEVOX_ENGINE_URL = "http://localhost:50021"

# M4 Pro CPU コア数に合わせたパラメータ (14コアCPUの場合)
MAX_WORKERS = min(14, multiprocessing.cpu_count())
BATCH_SIZE = 30  # フレーム生成のバッチサイズ

# 青山龍星のスピーカーID (実際のIDに要変更)
AOYAMA_RYUSEI_SPEAKER_ID = 0  # 仮の値


class VideoGenerator:
    """動画生成クラス - M4 Pro最適化版"""
    
    def __init__(self, config=None):
        """初期化"""
        self.config = config or {}
        self.font_path = self.config.get("font_path", DEFAULT_FONT_PATH)
        self.output_dir = self.config.get("output_dir", DEFAULT_OUTPUT_DIR)
        self.temp_dir = self.config.get("temp_dir", DEFAULT_TEMP_DIR)
        self.voicevox_url = self.config.get("voicevox_url", DEFAULT_VOICEVOX_ENGINE_URL)
        self.max_workers = self.config.get("max_workers", MAX_WORKERS)
        
        # パフォーマンスカウンター
        self.start_time = time.time()
        self.performance_metrics = {}
        
        # 出力ディレクトリ作成
        for directory in [self.output_dir, self.temp_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # フォントの読み込み（存在確認）
        try:
            self.font = ImageFont.truetype(self.font_path, 48)
            logger.info(f"フォント読み込み成功: {self.font_path}")
        except IOError:
            logger.warning(f"警告: フォントファイル {self.font_path} が見つかりません。システムのデフォルトフォントを使用します。")
            self.font = ImageFont.load_default()
    
    def calculate_font_size(self, text, max_width=1600, min_size=36, max_size=72):
        """テキストの長さに基づいて適切なフォントサイズを計算する（互換性のために残す）"""
        # 文字数に応じた基本的なフォントサイズの計算
        text_length = len(text)
        
        if text_length <= 10:  # 短いテキスト
            font_size = max_size
        elif text_length <= 20:  # 中長テキスト
            # 10文字ならmax_sizeから20文字なら中間サイズまでの線形補間
            font_size = max_size - ((text_length - 10) / 10) * (max_size - min_size) * 0.5
        elif text_length <= 40:  # 長いテキスト
            # 20文字から40文字までの線形補間
            font_size = max_size - ((max_size - min_size) * 0.5) - ((text_length - 20) / 20) * (max_size - min_size) * 0.5
        else:  # 非常に長いテキスト
            font_size = min_size
        
        # 最小値、最大値の範囲に収める
        font_size = max(min_size, min(max_size, font_size))
        
        return int(font_size)
        
    def calculate_font_size_by_length(self, text_length, min_size=42, max_size=72):
        """文字数に基づいて適切なフォントサイズを計算する"""
        if text_length <= 10:
            return max_size
        elif text_length <= 20:
            return max_size - ((text_length - 10) / 10) * (max_size - min_size) * 0.5
        elif text_length <= 40:
            return max_size - ((max_size - min_size) * 0.5) - ((text_length - 20) / 20) * (max_size - min_size) * 0.5
        else:
            return min_size
            
    def find_line_break_position(self, text):
        """テキストの適切な改行位置を見つける"""
        # 理想的な分割位置は文の半分あたり
        mid_point = len(text) // 2
        
        # 文の半分付近で自然な区切り（句読点や助詞）を探す
        search_range = 10  # 中間点から前後10文字以内で探す
        break_chars = '、。,.!?！？ 　'
        
        # 中間点から前後のbreakチャンスを探す
        for i in range(search_range):
            # 中間点より後ろ
            if mid_point + i < len(text) and text[mid_point + i] in break_chars:
                return mid_point + i + 1
            
            # 中間点より前
            if mid_point - i >= 0 and text[mid_point - i] in break_chars:
                return mid_point - i + 1
        
        # 適切な区切りが見つからない場合は単純に半分で分割
        return mid_point
    
    def log_performance(self, task_name, start_time=None):
        """パフォーマンスメトリクスを記録"""
        if start_time is None:
            self.performance_metrics[task_name] = {'start': time.time()}
            return time.time()
        else:
            end_time = time.time()
            duration = end_time - start_time
            self.performance_metrics[task_name] = {
                'start': start_time,
                'end': end_time,
                'duration': duration
            }
            logger.info(f"タスク '{task_name}' 完了: {duration:.2f}秒")
            return end_time
    
    def load_script(self, json_path):
        """台本JSONファイルを読み込む"""
        start_time = self.log_performance('load_script')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.log_performance('load_script', start_time)
            return data
        except Exception as e:
            logger.error(f"エラー: JSONファイルの読み込みに失敗しました: {e}")
            sys.exit(1)
    
    def generate_voice(self, text, speaker_id, output_path):
        """VOICEVOXを使って音声を合成する"""
        try:
            # VOICEVOX API呼び出し
            # クエリの作成
            params = {"text": text, "speaker": speaker_id}
            response = requests.post(
                f"{self.voicevox_url}/audio_query",
                params=params
            )
            response.raise_for_status()
            query_data = response.json()
            
            # 音声合成
            synthesis_response = requests.post(
                f"{self.voicevox_url}/synthesis",
                headers={"Content-Type": "application/json"},
                params=params,
                data=json.dumps(query_data)
            )
            synthesis_response.raise_for_status()
            
            # 音声ファイル保存
            with open(output_path, "wb") as f:
                f.write(synthesis_response.content)
            
            logger.info(f"音声合成完了: {output_path}")
            
            # 音声の長さを取得して返す
            duration = self.get_audio_duration(output_path)
            return duration
            
        except Exception as e:
            logger.warning(f"警告: 音声合成に失敗しました: {e}")
            # ダミー音声ファイル作成 (デバッグ用)
            try:
                # macOS
                subprocess.run(["touch", output_path], check=True)
                return 3.0  # ダミーの長さ
            except Exception as touch_err:
                logger.error(f"エラー: ダミー音声ファイル作成に失敗しました: {touch_err}")
                sys.exit(1)
    
    def get_audio_duration(self, audio_path):
        """WAVファイルの長さを取得する"""
        try:
            with contextlib.closing(wave.open(audio_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            logger.warning(f"警告: 音声ファイルの長さを取得できませんでした: {e}")
            return 3.0  # デフォルト値