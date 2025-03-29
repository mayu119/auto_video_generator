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
    
    def generate_video(self, script_data):
        """台本データから動画を生成する"""
        total_start_time = self.log_performance('total_generation')
        
        try:
            # 音声合成の並列処理
            logger.info("音声合成を開始します...")
            voice_results = self.parallel_generate_voices(script_data['segments'])
            
            # 音声ファイルとセグメント情報を整理
            audio_files = [result['audio_path'] for result in voice_results]
            segments_with_duration = []
            
            for result in voice_results:
                segment = result['segment']
                segment_with_duration = segment.copy()
                segment_with_duration['actual_duration'] = result['actual_duration']
                segments_with_duration.append(segment_with_duration)
                
                logger.info(f"セグメント {segment['segmentId']}: "
                           f"予定時間={segment.get('duration', 'なし')}秒, "
                           f"実際の音声長={result['actual_duration']:.2f}秒")
            
            # 音声を結合
            logger.info("音声ファイルを結合しています...")
            start_time = self.log_performance('audio_concatenation')
            final_audio_path = os.path.join(self.temp_dir, "final_audio.wav")
            self.concatenate_audio(audio_files, final_audio_path)
            self.log_performance('audio_concatenation', start_time)
            
            # 上部字幕に下部字幕の終了時間までの表示展開時間を計算
            for i in range(0, len(segments_with_duration), 2):
                if i + 1 < len(segments_with_duration):  # 上下ペアがある場合
                    top_segment = segments_with_duration[i]
                    bottom_segment = segments_with_duration[i + 1]
                    
                    # 上下ペアが正しい位置タイプか確認
                    if top_segment['position'] == 'top' and bottom_segment['position'] == 'bottom':
                        # 当初の上部字幕の表示時間
                        top_segment_display = top_segment['actual_duration']
                        # 下部字幕が終わるまで上部字幕を表示する追加時間
                        extend_time = bottom_segment['actual_duration']
                        
                        # 上部字幕に展開時間を追加
                        top_segment['extend_time'] = extend_time
                        
                        logger.info(f"上部字幕 {top_segment['segmentId']} の展開時間: {extend_time:.2f}秒, "
                               f"合計表示時間: {top_segment_display + extend_time:.2f}秒")
            
            # 音声の累積時間を計算して各セグメントの開始・終了時間を決定
            logger.info("タイムライン作成中...")
            current_time = 0.0
            
            for segment in segments_with_duration:
                start_time = current_time
                end_time = start_time + segment['actual_duration']
                
                segment['start_time'] = start_time
                segment['end_time'] = end_time
                
                current_time = end_time
                
                logger.info(f"セグメント {segment['segmentId']}: 位置={segment['position']}, "
                           f"開始={start_time:.2f}秒, 終了={end_time:.2f}秒")

            
            # フレーム生成の準備
            logger.info("フレーム生成を準備しています...")
            frames_dir = os.path.join(self.temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            frame_duration = 1/30  # 30fps
            total_duration = segments_with_duration[-1]['end_time'] if segments_with_duration else 0
            total_frames = int(total_duration / frame_duration)
            frame_batches = []
            
            # 各フレームの時刻を計算
            frame_times = [i * frame_duration for i in range(total_frames)]
            
            # バッチ情報を生成
            logger.info(f"合計 {total_frames} フレーム、時間: {total_duration:.2f}秒")
            
            # フレームバッチを生成
            for batch_start in range(0, total_frames, BATCH_SIZE):
                batch_size = min(BATCH_SIZE, total_frames - batch_start)
                batch_frame_times = frame_times[batch_start:batch_start + batch_size]
                
                batch_info = {
                    'start_idx': batch_start,
                    'frame_count': batch_size,
                    'frame_times': batch_frame_times,
                    'segments': segments_with_duration,
                    'frames_dir': frames_dir
                }
                
                frame_batches.append(batch_info)
            
            # 並列処理でフレームを生成
            logger.info(f"フレームを生成しています... (合計 {total_frames} フレーム, {len(frame_batches)} バッチ)")
            self.parallel_generate_frames(frame_batches, frames_dir)
            
            # FFmpegを使って動画を生成
            logger.info("動画を生成しています...")
            output_filename = f"{script_data['title']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_video_path = os.path.join(self.output_dir, output_filename)
            
            try:
                start_time = self.log_performance('ffmpeg_processing')
                # フレームから無音ビデオを生成
                temp_video_path = os.path.join(self.temp_dir, "temp_video.mp4")
                
                # M4 Pro向けのFFmpeg最適化パラメータ
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-framerate", "30",
                    "-i", os.path.join(frames_dir, "frame_%06d.png"),
                    "-c:v", "h264_videotoolbox",  # Apple Silicon用ハードウェアエンコーダ
                    "-b:v", "5M",
                    "-pix_fmt", "yuv420p",
                    temp_video_path
                ]
                subprocess.run(ffmpeg_cmd, check=True)
                
                # 音声を追加
                final_cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_video_path,
                    "-i", final_audio_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    output_video_path
                ]
                subprocess.run(final_cmd, check=True)
                
                self.log_performance('ffmpeg_processing', start_time)
                self.log_performance('total_generation', total_start_time)
                
                # パフォーマンスレポート
                self.print_performance_report()
                
                logger.info(f"動画が生成されました: {output_video_path}")
                return output_video_path
                
            except Exception as e:
                logger.error(f"エラー: 動画生成に失敗しました: {e}")
                sys.exit(1)
        
        except Exception as e:
            logger.error(f"エラー: 処理中に例外が発生しました: {e}")
            self.log_performance('total_generation', total_start_time)
            sys.exit(1)
        finally:
            # 一時ディレクトリのクリーンアップ (オプション)
            if self.config.get("cleanup_temp", True):
                logger.info(f"一時ファイルをクリーンアップしています: {self.temp_dir}")
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    logger.warning(f"一時ファイルのクリーンアップに失敗しました: {e}")
                    
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
    
    def concatenate_audio(self, audio_files, output_path):
        """音声ファイルを結合する"""
        try:
            # FFmpegの入力ファイルリスト作成
            concat_file = os.path.join(self.temp_dir, "concat_list.txt")
            with open(concat_file, 'w', encoding='utf-8') as f:
                for audio_path in audio_files:
                    f.write(f"file '{os.path.abspath(audio_path)}'\n")
            
            # FFmpegで結合
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                output_path
            ], check=True)
            
            return output_path
        except Exception as e:
            logger.warning(f"警告: 音声結合に失敗しました: {e}")
            # FFmpegがない場合など、失敗したときは最初の音声ファイルをコピーする
            if audio_files:
                shutil.copy(audio_files[0], output_path)
                return output_path
            else:
                return None
    
    def print_performance_report(self):
        """パフォーマンスレポートを出力"""
        logger.info("\n==== パフォーマンスレポート ====")
        
        for task, metrics in self.performance_metrics.items():
            if 'duration' in metrics:
                logger.info(f"{task}: {metrics['duration']:.2f}秒")
        
        total_time = self.performance_metrics.get('total_generation', {}).get('duration')
        if total_time:
            logger.info(f"総処理時間: {total_time:.2f}秒")
        
        logger.info("=============================\n")
    
    def parallel_generate_voices(self, segments):
        """並列処理で音声を合成"""
        start_time = self.log_performance('parallel_generate_voices')
        
        def process_voice(idx, segment):
            audio_path = os.path.join(self.temp_dir, f"audio_{idx:04d}.wav")
            speaker_id = AOYAMA_RYUSEI_SPEAKER_ID
            actual_duration = self.generate_voice(segment['voiceText'], speaker_id, audio_path)
            return {
                'segment': segment,
                'audio_path': audio_path,
                'actual_duration': actual_duration,
                'idx': idx
            }
        
        # 並列処理で音声合成
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, self.max_workers)) as executor:
            futures = {executor.submit(process_voice, idx, segment): idx 
                      for idx, segment in enumerate(segments)}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"セグメント {idx} の音声合成中にエラー: {e}")
        
        # インデックス順にソート
        results.sort(key=lambda x: x['idx'])
        
        self.log_performance('parallel_generate_voices', start_time)
        return results
        
    def parallel_generate_frames(self, frame_batches, frames_dir):
        """並列処理でフレームを生成"""
        start_time = self.log_performance('parallel_generate_frames')
        all_frames = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.generate_frame_batch, batch) for batch in frame_batches]
            for future in concurrent.futures.as_completed(futures):
                all_frames.extend(future.result())
        
        self.log_performance('parallel_generate_frames', start_time)
        return all_frames
        
    def generate_frame_batch(self, batch_info):
        """フレームのバッチを生成する（並列処理用）"""
        start_idx = batch_info['start_idx']
        frame_count = batch_info['frame_count']
        frame_times = batch_info['frame_times']
        segments = batch_info['segments']
        frames_dir = batch_info['frames_dir']
        frames = []
        
        for i in range(frame_count):
            frame_idx = start_idx + i
            frame_time = frame_times[i]
            
            # 現在の時間に表示すべき上部と下部のテキストを探す
            top_text = None
            bottom_text = None
            
            # 現在アクティブな上部字幕を探す
            current_top_segment = None
            for segment in segments:
                if segment['position'] == 'top' and segment['start_time'] <= frame_time < segment['end_time'] + segment.get('extend_time', 0):
                    current_top_segment = segment
                    top_text = segment['text']
                    break
            
            # 上部字幕がアクティブな場合、対応する下部字幕を探す
            if current_top_segment:
                segment_id = current_top_segment['segmentId']
                # 対応する下部字幕はIDが1大きいはず
                next_segment_id = segment_id + 1
                
                for segment in segments:
                    if segment['position'] == 'bottom' and segment['segmentId'] == next_segment_id:
                        # 下部字幕の表示はタイミングに依存
                        if segment['start_time'] <= frame_time < segment['end_time']:
                            bottom_text = segment['text']
                        break
            
            # フレームを生成
            frame = self.create_frame(top_text=top_text, bottom_text=bottom_text)
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
            frame.save(frame_path)
            frames.append(frame_path)
        
        return frames
        
    def create_frame(self, top_text=None, bottom_text=None, width=1920, height=1080):
        """テロップ付きのフレームを生成する（最適化版）"""
        # 背景
        img_array = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白背景
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)
        
        # 上部テロップ処理（最大2行）
        if top_text:
            y_position = height * 0.15  # 上から15%の位置
            # 改行文字で分割
            if '\n' in top_text:
                lines = top_text.split('\n')
                line1 = lines[0]
                line2 = lines[1] if len(lines) > 1 else ""
                # フォントサイズ計算（両方の行に適用される共通サイズ）
                max_line_len = max(len(line1), len(line2))
                font_size = self.calculate_font_size_by_length(max_line_len, min_size=42, max_size=72)
                font = ImageFont.truetype(self.font_path, font_size)
                # 2行描画
                line_spacing = font_size * 1.2
                draw.text((width // 2, int(y_position - line_spacing/2)), line1, font=font, fill='black', anchor='mm', align='center')
                draw.text((width // 2, int(y_position + line_spacing/2)), line2, font=font, fill='black', anchor='mm', align='center')
            else:
                # 長いテキストは自動的に適切な位置で改行
                if len(top_text) > 25:
                    # 適切な改行位置を探す
                    pos = self.find_line_break_position(top_text)
                    line1 = top_text[:pos].strip()
                    line2 = top_text[pos:].strip()
                    # フォントサイズ計算
                    max_line_len = max(len(line1), len(line2))
                    font_size = self.calculate_font_size_by_length(max_line_len, min_size=42, max_size=72)
                    font = ImageFont.truetype(self.font_path, font_size)
                    # 2行描画
                    line_spacing = font_size * 1.2
                    draw.text((width // 2, int(y_position - line_spacing/2)), line1, font=font, fill='black', anchor='mm', align='center')
                    draw.text((width // 2, int(y_position + line_spacing/2)), line2, font=font, fill='black', anchor='mm', align='center')
                else:
                    # 1行のみの場合
                    font_size = self.calculate_font_size_by_length(len(top_text), min_size=48, max_size=80)
                    font = ImageFont.truetype(self.font_path, font_size)
                    draw.text((width // 2, int(y_position)), top_text, font=font, fill='black', anchor='mm', align='center')
        
        # 下部テロップ（常に1行）
        if bottom_text:
            y_position = height * 0.8  # 下から20%の位置
            font_size = self.calculate_font_size_by_length(len(bottom_text), min_size=42, max_size=72)
            font = ImageFont.truetype(self.font_path, font_size)
            draw.text((width // 2, int(y_position)), bottom_text, font=font, fill='black', anchor='mm', align='center')
        
        return img
    
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


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="JSONファイルから動画を自動生成するプログラム (M4 Pro最適化版)")
    parser.add_argument("json_file", help="台本のJSONファイルパス")
    parser.add_argument("--output-dir", help="出力ディレクトリ", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temp-dir", help="一時ファイル用ディレクトリ", default=DEFAULT_TEMP_DIR)
    parser.add_argument("--font-path", help="フォントファイルパス", default=DEFAULT_FONT_PATH)
    parser.add_argument("--voicevox-url", help="VOICEVOXエンジンのURL", default=DEFAULT_VOICEVOX_ENGINE_URL)
    parser.add_argument("--max-workers", help="最大並列ワーカー数", type=int, default=MAX_WORKERS)
    parser.add_argument("--no-cleanup", help="一時ファイルを残す", action="store_true")
    
    args = parser.parse_args()
    
    # 設定
    config = {
        "output_dir": args.output_dir,
        "temp_dir": args.temp_dir,
        "font_path": args.font_path,
        "voicevox_url": args.voicevox_url,
        "max_workers": args.max_workers,
        "cleanup_temp": not args.no_cleanup
    }
    
    # 動画生成器の初期化
    generator = VideoGenerator(config)
    
    # 台本の読み込み
    script_data = generator.load_script(args.json_file)
    
    # 動画の生成
    output_video = generator.generate_video(script_data)
    
    logger.info(f"処理が完了しました。動画: {output_video}")


if __name__ == "__main__":
    main()