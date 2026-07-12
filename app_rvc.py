import re, os, hashlib
import requests
import json
import torch
import shutil
import argparse
import threading
from difflib import SequenceMatcher

progress_local = threading.local()

import tqdm
class GradioTqdm(tqdm.tqdm):
    def update(self, n=1):
        super().update(n)
        if hasattr(progress_local, 'progress') and progress_local.progress is not None:
            if self.total and self.total > 0:
                pct = int(self.n / self.total * 100)
                if pct != getattr(self, '_last_pct', -1) and pct % 5 == 0:
                    self._last_pct = pct
                    progress_local.progress(0.4 + (pct / 100.0) * 0.2, desc=f"分离人声 {pct}%")

tqdm.tqdm = GradioTqdm
import tqdm.auto
tqdm.auto.tqdm = GradioTqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--is_nohalf', action='store_true'
)
parser.add_argument(
    '--dml', action='store_true'
)
a = parser.parse_args()

use_dml = a.dml
if use_dml:
    try:
        import torch_directml
        device = torch_directml.device(torch_directml.default_device())
        is_half = False
        print(f"[AMD] 使用 DirectML 设备: {device}")
    except ImportError:
        print("[AMD] torch_directml 未安装，回退到 CPU")
        device = 'cpu'
        is_half = False
else:
    is_half = not a.is_nohalf
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
}
pattern = r'//www\.bilibili\.com/video[^"]*'
models=[]
index=[]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RVC_API_BASE = "http://127.0.0.1:2333"

# ========== 新增：音高优化函数 ==========
def optimize_pitch_shift(key_shift):
    """
    将升降调优化到最小调整幅度，保证最佳音质
    例如：+11 转为 -1，-10 转为 +2
    """
    if key_shift > 6:
        return key_shift - 12
    elif key_shift < -6:
        return key_shift + 12
    else:
        return key_shift
# ======================================

def get_response(song_id):
  print("开始下载歌曲")
  try:
    response = requests.get(f"https://biliplayer.91vrchat.com/player/?url=https://music.163.com/song?id={song_id}",allow_redirects=True, timeout=30)
    if response.status_code == 200:
      return response
  except Exception as e:
    print(f"主源下载失败: {e}")
  
  print("使用备用源下载歌曲")
  try:
      response1 = requests.get(
          f"https://api.vkeys.cn/v2/music/netease?id={song_id}",
          timeout=30
      ).json()["data"]["url"]
      res = requests.get(response1, timeout=30)
      return res
  except Exception as e:
      raise Exception(f"所有下载源均失败: {e}")

def change_model(model):
  """切换模型"""
  try:
    response = requests.post(f"{RVC_API_BASE}/run/infer_change_voice", json={
      "data": [
        model,
        0.33,
        0.33,
    ]}, timeout=10).json()
    print(f"模型已切换为: {model}")
    return f"✅ 成功切换到模型: {model}"
  except Exception as e:
    print(f"切换模型失败: {e}")
    return f"❌ 切换模型失败: {e}"

def show_model():
  """获取可用模型列表"""
  global models, index
  try:
    response = requests.post(f"{RVC_API_BASE}/run/infer_refresh", json={
      "data": []
    }, timeout=10).json()

    models = response["data"][0]["choices"]
    index = response["data"][1]["choices"]
    print(f"已加载 {len(models)} 个模型")
    return models
  except Exception as e:
    print(f"获取模型列表失败: {e}")
    return []

def find_index(model):   
    if not index:
        return None
    
    # 提取模型名（去掉扩展名）
    if isinstance(model, list):
        model = model[0] if model else ""
    model_name = os.path.splitext(model)[0].lower()
    
    # 计算每个 index 文件的相似度
    best_match = None
    best_score = 0
    threshold = 0.4
    
    for index_path in index:
        # 提取 index 文件名（去掉路径和扩展名）
        index_name = os.path.splitext(os.path.basename(index_path))[0].lower()
        
        # 计算相似度
        score = SequenceMatcher(None, model_name, index_name).ratio()
        
        if score > best_score:
            best_score = score
            best_match = index_path
    if best_score < threshold:
        print(f"未找到匹配的 index（最高相似度: {best_score:.2f}）")
        return None
    if best_match:
        best_match="./"+ best_match
        print(f"找到匹配: {best_match}（相似度: {best_score:.2f}）")
    return best_match
    

import sys
MSST_DIR = os.path.join(SCRIPT_DIR, "msst")
if MSST_DIR not in sys.path:
    sys.path.insert(0, MSST_DIR)
from msst.msst_separate import (
    DEFAULT_MODEL_ID,
    get_msst_models,
    resolve_msst_model,
    separate_vocal,
    unload_model,
)

from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.effects import compress_dynamic_range
from pydub.effects import normalize
from pedalboard import Pedalboard, Compressor, Reverb
from scipy.signal import firwin, lfilter, iirfilter
import os
import numpy as np
import librosa
import soundfile
import gradio as gr
import scipy.signal
if not hasattr(scipy.signal, 'hann'):
    scipy.signal.hann = np.hanning
split_model = "MSST"


def _normalize_msst_model(model_name):
    try:
        return resolve_msst_model(model_name)[0]
    except (ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from exc


def _msst_cache_base(cache_name, model_name, batch_size, num_overlap, normalize, use_tta):
    """Keep stems from different quality profiles in separate caches."""
    model_suffix = "" if model_name == DEFAULT_MODEL_ID else f"_{sanitize_filename(model_name)}"
    quality_suffix = (
        f"_b{int(batch_size)}_o{int(num_overlap)}"
        f"_n{int(bool(normalize))}_tta{int(bool(use_tta))}"
    )
    return f"{cache_name}{model_suffix}{quality_suffix}"


def show_msst_models_api():
    models_list = get_msst_models()
    model_ids = ", ".join(item["id"] for item in models_list) or "<none>"
    print(
        f"🔎 [MSST模型列表Debug] 收到插件/API读取请求，返回 {len(models_list)} 个模型: {model_ids}",
        flush=True,
    )
    return models_list


def select_msst_model_api(model_name):
    model_id, model_path, config_path = resolve_msst_model(model_name)
    print(
        f"✅ [MSST模型切换Debug] 插件默认分离模型切换成功: {model_id} | "
        f"checkpoint={os.path.basename(model_path)} | config={os.path.basename(config_path)}",
        flush=True,
    )
    return {"success": True, "id": model_id}

# 替换这个函数
def wwy_downloader(
    filename,
    split_model,
    cache_name=None,
    msst_batch_size=1,
    msst_num_overlap=4,
    msst_normalize=False,
    msst_use_tta=False,
    msst_model=DEFAULT_MODEL_ID,
):
    cache_dir = cache_name if cache_name else filename
    msst_model = _normalize_msst_model(msst_model)
    separation_name = _msst_cache_base(
        cache_dir, msst_model, msst_batch_size, msst_num_overlap,
        msst_normalize, msst_use_tta,
    )
    audio_content = get_response(filename).content
    temp_prefixed_path = "rvc_" + cache_dir + ".wav"
    with open(temp_prefixed_path, mode="wb") as f:
        f.write(audio_content)
    
    audio_orig = AudioSegment.from_file(temp_prefixed_path)
    duration_minutes = len(audio_orig) / 60000
    print(f"Duration: {duration_minutes:.2f} min")
    if duration_minutes > 5:
        print("Audio > 5min, trimming...")
        audio_orig = audio_orig[:300000]
    
    msst_input_path = separation_name + ".wav"
    audio_orig.export(msst_input_path, format="wav")
    
    if os.path.isfile(temp_prefixed_path):
        os.remove(temp_prefixed_path)

    output_dir = f"./output/{split_model}/{cache_dir}/"
    os.makedirs(output_dir, exist_ok=True)
    print("[MSST] Separating vocals (BS-Roformer)...")
    vocal_path, inst_path = separate_vocal(
        msst_input_path, output_dir=output_dir,
        inference_params={
            "batch_size": msst_batch_size,
            "num_overlap": msst_num_overlap,
            "normalize": msst_normalize,
            "use_tta": msst_use_tta,
        },
        release_after=True,
        model_name=msst_model,
    )
    
    if os.path.isfile(msst_input_path):
        os.remove(msst_input_path)

    if vocal_path and inst_path:
        return vocal_path, inst_path
    else:
        raise gr.Error("MSST separation failed, output not found")



def convert(song_name_src, key_shift, vocal_vol, inst_vol, model_dropdown, reverb_intensity = 4, delay_intensity = 0, f0_method = "rmvpe", index_rate = 0.75, filter_radius = 3, uvr5_agg = 10, uvr5_tta = False, uvr5_postprocess = False, uvr5_window_size = 512, uvr5_high_end_process = "mirroring", msst_batch_size = 1, msst_num_overlap = 4, msst_normalize = False, msst_use_tta = False, msst_model=DEFAULT_MODEL_ID, shift_accompaniment=True, progress=gr.Progress()):
  """进行翻唱推理合成"""
  print(f"🎵 [任务开始] RVC模型: {model_dropdown} | 算法: {f0_method} | 检索率: {index_rate} | 滤波: {filter_radius} | 升降调: {key_shift}")
  msst_model = _normalize_msst_model(msst_model)
  print(f"🔧 [MSST 参数] Model: {msst_model} | BatchSize: {msst_batch_size} | Overlap: {msst_num_overlap} | Normalize: {msst_normalize} | TTA: {msst_use_tta}")
  print(f"🔧 [UVR5 参数(忽略，MSST后端不使用)] Agg: {uvr5_agg} | TTA: {uvr5_tta} | PostProcess: {uvr5_postprocess} | WindowSize: {uvr5_window_size} | HighEnd: {uvr5_high_end_process}")
  progress_local.progress = progress
  progress(0.1, desc="正在准备处理歌曲...")
  split_model = "MSST"
  if not song_name_src: raise gr.Error("请输入歌曲ID或链接！")
  
  if song_name_src.startswith("http"):
    try: song_name_src = song_name_src.split('id=')[1].split('&')[0]
    except IndexError: raise gr.Error("无效的网易云链接格式！")
  
  song_name_src = song_name_src.strip()
  print(f"处理歌曲ID: {song_name_src}")
  
  audio_rvc_path = os.path.join(SCRIPT_DIR, "audio_rvc.wav")
  
  # === 检查是否为本地文件路径（支持QQ音乐等外部音频） ===
  is_local_file = os.path.isfile(song_name_src) and not song_name_src.startswith("http")
  
  if is_local_file:
    # 本地文件（QQ音乐等）：生成安全的缓存名称（使用哈希避免中文/特殊字符问题）
    safe_name = f"qqmusic_{abs(hash(song_name_src)) % 10000000}"
    # 【关键修复】替换 song_name_src 为安全名称，确保后续所有路径构建都正确
    original_song_name = song_name_src
    song_name_src = safe_name
    separation_name = _msst_cache_base(
        safe_name, msst_model, msst_batch_size, msst_num_overlap,
        msst_normalize, msst_use_tta,
    )
    vocal_cache_path = f"./output/{split_model}/{safe_name}/{separation_name}_vocals.wav"
    
    if os.path.isfile(vocal_cache_path):
      print("Cached, skipping")
      audio, sr = librosa.load(vocal_cache_path, sr=44100, mono=True)
      soundfile.write(audio_rvc_path, audio, sr)
    else:
      print(f"Loading local file: {os.path.basename(original_song_name)}")
      progress(0.2, desc="加载本地音频文件...")
      audio_orig = AudioSegment.from_file(original_song_name)
      duration_minutes = len(audio_orig) / 60000
      print(f"Duration: {duration_minutes:.2f} min")
      if duration_minutes > 5:
        print("Audio > 5min, trimming...")
        audio_orig = audio_orig[:300000]
      
      msst_input_path = separation_name + ".wav"
      audio_orig.export(msst_input_path, format="wav")
      
      output_dir = f"./output/{split_model}/{safe_name}/"
      os.makedirs(output_dir, exist_ok=True)
      print("[MSST] Separating vocals (BS-Roformer)...")
      progress(0.4, desc="分离人声中(BS-Roformer)...")
      vocal_path, inst_path = separate_vocal(
          msst_input_path, output_dir=output_dir,
          inference_params={
              "batch_size": msst_batch_size,
              "num_overlap": msst_num_overlap,
              "normalize": msst_normalize,
              "use_tta": msst_use_tta,
          },
          release_after=True,
          model_name=msst_model,
      )
      
      if os.path.isfile(msst_input_path):
        os.remove(msst_input_path)
      
      if vocal_path and os.path.isfile(vocal_path):
        audio_rvc, sr_src = librosa.load(vocal_path, sr=44100, mono=True)
        soundfile.write(audio_rvc_path, audio_rvc, sr_src)
      else:
        raise gr.Error(f"MSST separation failed: {vocal_path}")
  else:
    # 网易云音乐：使用 netease_ 前缀标识
    netease_safe_name = f"netease_{song_name_src}"
    separation_name = _msst_cache_base(
        netease_safe_name, msst_model, msst_batch_size, msst_num_overlap,
        msst_normalize, msst_use_tta,
    )
    vocal_cache_path = f"./output/{split_model}/{netease_safe_name}/{separation_name}_vocals.wav"
    
    if os.path.isfile(vocal_cache_path):
      print("✅ 网易云歌曲已缓存，跳过下载")
      audio, sr = librosa.load(vocal_cache_path, sr=44100, mono=True)
      soundfile.write(audio_rvc_path, audio, sr)
    else:
      print("📥 未找到缓存，开始下载和分离（网易云）")
      progress(0.4, desc="网易云下载并分离人声...")
      audio_rvc, sr_src = librosa.load(wwy_downloader(
          song_name_src, split_model, cache_name=netease_safe_name,
          msst_batch_size=msst_batch_size, msst_num_overlap=msst_num_overlap,
          msst_normalize=msst_normalize,
          msst_use_tta=msst_use_tta,
          msst_model=msst_model,
      )[0], sr=44100, mono=True)
      soundfile.write(audio_rvc_path, audio_rvc, sr_src)

  # ========== 缓存检查 ==========
  cache_key = _get_cache_key(
      song_name_src if is_local_file else netease_safe_name,
      model_dropdown, key_shift, vocal_vol, inst_vol,
      reverb_intensity, delay_intensity, f0_method,
      index_rate, filter_radius, uvr5_agg, uvr5_tta,
      uvr5_postprocess, uvr5_window_size, uvr5_high_end_process,
      msst_batch_size, msst_num_overlap, msst_normalize, msst_use_tta,
      msst_model,
      shift_accompaniment
  )
  cache_name = song_name_src if is_local_file else netease_safe_name
  cache_path = f"temp/{sanitize_filename(cache_name)}_{cache_key}_RVC.mp3"
  if os.path.isfile(cache_path):
      print(f"Cache hit, returning: {cache_path}")
      progress(1.0, desc="Cache hit, returning directly!")
      progress_local.progress = None
      return cache_path, "true"
  # =============================

  print("🎤 RVC 推理中...")
  progress(0.55, desc="切换RVC模型中...")
  
  switch_model(model_dropdown)
  progress(0.58, desc="RVC模型推理中...")
  
  try:
      response = requests.post(f"{RVC_API_BASE}/run/infer_convert", json={
        "data": [
          0,
          audio_rvc_path,
          key_shift,
          None,
          f0_method,
          "",
          find_index(model_dropdown),
          index_rate,
          filter_radius,
          0,
          0.25,
          0.33,
      ]}, timeout=600).json()
  except Exception as e:
      print(f"RVC推理请求失败: {e}")
      raise gr.Error(f"RVC推理请求失败: {e}")

  if "data" not in response:
      error_msg = response.get("error", str(response))
      print(f"RVC推理失败: {error_msg}")
      raise gr.Error(f"RVC推理失败: {error_msg}")

  progress(0.75, desc="RVC推理完成，处理音频中...")

  try:
      data_list = response.get("data", [])
      data = None
      if len(data_list) > 1:
          data_obj = data_list[1]
          if isinstance(data_obj, dict) and "name" in data_obj:
              data = data_obj["name"]
          elif isinstance(data_obj, str) and data_obj.endswith(".wav"):
              data = data_obj
      if not data and len(data_list) > 0:
          for item in data_list:
              if isinstance(item, str) and (item.endswith(".wav") or item.endswith(".flac") or item.endswith(".mp3")):
                  data = item
                  break
              if isinstance(item, dict) and "name" in item:
                  data = item["name"]
                  break
  except Exception as e:
      data = None
      print(f"解析返回值失败: {response} {e}")

  print(f"RVC推理结果: {data}")

  if data:
    print("🎛️ 开始处理音频")
    progress(0.78, desc="加载推理结果音频...")
    os.makedirs("./temp", exist_ok=True)

    audio_data, sr = librosa.load(data, sr=None, mono=False)

    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(1, -1)

    progress(0.82, desc="应用音频效果(均衡/压缩/混响)...")
    from pedalboard import Pedalboard, Compressor, Reverb, HighpassFilter, PeakFilter, LowpassFilter, PitchShift, Delay

    # ========== 修正后的智能混响参数计算 ==========
    # 定义参数的锚点
    # 强度级别:   0 (最小)       4 (默认)       10 (最大)
    room_size_map =  (0.15,          0.40,          0.90)
    wet_level_map =  (0.10,          0.25,          0.45)

    # 根据滑块位置，在两段之间进行线性插值
    if reverb_intensity <= 4:
        # 在 0-4 区间
        # 计算当前位置在该区间的百分比
        percent = reverb_intensity / 4.0
        # 在 (最小) 和 (默认) 参数之间插值
        room_size_val = room_size_map[0] + (room_size_map[1] - room_size_map[0]) * percent
        wet_level_val = wet_level_map[0] + (wet_level_map[1] - wet_level_map[0]) * percent
    else:
        # 在 4-10 区间
        # 计算当前位置在该区间的百分比
        percent = (reverb_intensity - 4) / 6.0  # (10 - 4 = 6)
        # 在 (默认) 和 (最大) 参数之间插值
        room_size_val = room_size_map[1] + (room_size_map[2] - room_size_map[1]) * percent
        wet_level_val = wet_level_map[1] + (wet_level_map[2] - wet_level_map[1]) * percent

    # 干信号总是与湿信号互补
    dry_level_val = 1.0 - wet_level_val

    print(f"🎤 混响设置: 强度 {reverb_intensity}/10 => 房间大小={room_size_val:.2f}, 湿润度={wet_level_val:.2f}")
    # ========================================
    # 根据来源类型使用正确的缓存名称构建伴奏路径
    if is_local_file:
        separation_name = _msst_cache_base(
            song_name_src, msst_model, msst_batch_size, msst_num_overlap,
            msst_normalize, msst_use_tta,
        )
        inst_path = f"output/{split_model}/{song_name_src}/{separation_name}_other.wav"
    else:
        separation_name = _msst_cache_base(
            netease_safe_name, msst_model, msst_batch_size, msst_num_overlap,
            msst_normalize, msst_use_tta,
        )
        inst_path = f"output/{split_model}/{netease_safe_name}/{separation_name}_other.wav"    
    effects = [
        HighpassFilter(cutoff_frequency_hz=80),
        PeakFilter(cutoff_frequency_hz=200, gain_db=1.5, q=0.7),
        PeakFilter(cutoff_frequency_hz=3000, gain_db=2.0, q=1.0),
        PeakFilter(cutoff_frequency_hz=7000, gain_db=-3.0, q=2.0),
        LowpassFilter(cutoff_frequency_hz=16000),
        Compressor(
            threshold_db=-18.0,
            ratio=4.0,
            attack_ms=5.0,
            release_ms=150.0
        ),
    ]
    
    # ========== 只有当用户开启延迟时，才执行所有相关计算 ==========
    if delay_intensity > 0:
        print("🎤 启用回声效果，开始准备参数...")
        
        # 1. 自动检测歌曲BPM
        try:
            print("🎵 正在检测歌曲BPM...")
            y_inst, sr_inst = librosa.load(inst_path, sr=None)
            tempo, _ = librosa.beat.beat_track(y=y_inst, sr=sr_inst)
            
            # ========== 新增的健壮性检查 ==========
            # 检查 tempo 是否为 NumPy 数组，如果是，则提取其第一个元素
            # 这可以兼容返回单个浮点数或单元素数组的各种 librosa 版本
            if isinstance(tempo, np.ndarray):
                actual_tempo = tempo[0]
            else:
                actual_tempo = tempo
            # ========================================

            if actual_tempo > 0:
                # 现在使用 actual_tempo 进行所有操作
                print(f"✅ 检测到歌曲BPM约为: {actual_tempo:.1f}")
                delay_seconds_val = (60.0 / actual_tempo) * 0.5 
            else:
                print("⚠️ 未能检测到有效的BPM，将使用默认值。")
                delay_seconds_val = 0.5
                
        except Exception as e:
            # 打印具体的错误信息，方便未来调试
            print(f"⚠️ BPM检测失败: {type(e).__name__}: {e}，将使用默认值。")
            delay_seconds_val = 0.5
            
        # 2. 计算延迟混合度
        delay_mix_val = (delay_intensity / 10.0) * 0.35
        
        # 3. 将 Delay 效果器添加到列表中
        print(f"🎤 回声设置: 强度 {delay_intensity}/10 => 混合度={delay_mix_val:.2f}, 延迟时间={delay_seconds_val:.3f}s (BPM同步)")
        effects.append(
            Delay(
                delay_seconds=delay_seconds_val,
                feedback=0.25,
                mix=delay_mix_val
            )
        )
    # ==========================================================
    
    # 最后添加混响效果器 (这总是在延迟之后)
    effects.append(
        Reverb(
            room_size=room_size_val,
            damping=0.4,
            wet_level=wet_level_val,
            dry_level=dry_level_val,
            width=0.8
        )
    )
    
    # 用最终的效果器列表创建 Pedalboard
    board = Pedalboard(effects)

    processed = board(audio_data, sr)
    processed_int16 = (processed.T * 32768).astype(np.int16)
    processed_audio = AudioSegment(
        processed_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=processed.shape[0]
    )
    
    audio_vocal_adjusted = processed_audio + vocal_vol
    normalized_audio = normalize(audio_vocal_adjusted, headroom=-1.0)
    
    progress(0.88, desc="处理伴奏并混音...")
    # ========== 新增：处理伴奏音高 ==========
    print("🎵 准备伴奏...")
    
    # 确保 temp 目录存在
    os.makedirs("temp", exist_ok=True)
    
    # 当开启伴奏升调且升降调不为0且不是±12（八度）时，同步调整伴奏
    inst_shift = key_shift
    if shift_accompaniment and inst_shift != 0 and abs(inst_shift) != 12:
        print(f"🎹 正在将伴奏音高调整 {inst_shift:+d} 半音以匹配人声...")
        
        try:
            # 加载伴奏
            y_inst, sr_inst = librosa.load(inst_path, sr=None)
            
            # 创建一个只包含音高调整效果的 Pedalboard
            pitch_board = Pedalboard([
                PitchShift(semitones=inst_shift)
            ])
            
            # 应用效果
            y_shifted = pitch_board(y_inst, sr_inst)
            
            # 保存处理后的伴奏为临时文件
            shifted_inst_path = f"temp/shifted_{song_name_src}_inst.wav"
            soundfile.write(shifted_inst_path, y_shifted, sr_inst)
            
            # 从处理后的文件加载为 AudioSegment
            audio_inst = AudioSegment.from_file(shifted_inst_path, format="wav")
            
            print(f"✅ 伴奏音高调整完成")
        except Exception as e:
            print(f"⚠️ 伴奏音高调整失败，使用原始伴奏: {e}")
            audio_inst = AudioSegment.from_file(inst_path, format="wav")
    else:
        # 不需要调整伴奏
        if not shift_accompaniment:
            print("🎹 已关闭伴奏升调，保持原伴奏")
        else:
            print("🎹 不调整伴奏音高")
        audio_inst = AudioSegment.from_file(inst_path, format="wav")

    audio_inst = audio_inst + inst_vol
    combined_audio = normalized_audio.overlay(audio_inst)

    print("💾 导出最终文件...")
    progress(0.95, desc="导出最终音频文件...")
    output_path = cache_path
    combined_audio.export(
        output_path,
        format="MP3",
        bitrate="192k"
    )
    
    if os.path.isfile(data):
      os.remove(data)
    
    print(f"✅ 已导出: {output_path}")
    progress(1.0, desc="处理完成！")
    progress_local.progress = None
    return output_path, "false"

def _get_cache_key(song_name, model, key_shift, vocal_vol, inst_vol, reverb_intensity, delay_intensity, f0_method, index_rate, filter_radius, uvr5_agg, uvr5_tta, uvr5_postprocess, uvr5_window_size, uvr5_high_end_process, msst_batch_size, msst_num_overlap, msst_normalize, msst_use_tta, msst_model, shift_accompaniment):
    params = {
        "song": str(song_name),
        "model": str(model),
        "key_shift": float(key_shift),
        "vocal_vol": float(vocal_vol),
        "inst_vol": float(inst_vol),
        "reverb": float(reverb_intensity),
        "delay": float(delay_intensity),
        "f0": str(f0_method),
        "index_rate": float(index_rate),
        "filter_radius": int(filter_radius),
        "uvr5_agg": int(uvr5_agg),
        "uvr5_tta": bool(uvr5_tta),
        "uvr5_postprocess": bool(uvr5_postprocess),
        "uvr5_window_size": int(uvr5_window_size),
        "uvr5_high_end": str(uvr5_high_end_process),
        "msst_batch": float(msst_batch_size),
        "msst_overlap": float(msst_num_overlap),
        "msst_norm": bool(msst_normalize),
        "msst_tta": bool(msst_use_tta),
        "msst_model": str(msst_model),
        "shift_inst": bool(shift_accompaniment),
    }
    return hashlib.md5(json.dumps(params, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:12]

def sanitize_filename(filename):
    # 定义 Windows 禁止的字符： \ / : * ? " < > |
    # 使用正则表达式移除这些字符
    clean_name = re.sub(r'[\\/:*?"<>|]', '', filename)
    return clean_name

def refresh_models():
    """刷新模型列表的回调函数"""
    models_list = show_model()
    if models_list:
        return gr.Dropdown(choices=models_list, value=models_list[0] if models_list else None)
    else:
        return gr.Dropdown(choices=["无可用模型"], value="无可用模型")

def switch_model(model_name):
    """切换模型的回调函数 - 返回状态信息"""
    if not model_name or model_name == "无可用模型":
        return "❌ 请先选择一个有效的模型"
    result = change_model(model_name)
    return result
    
app = gr.Blocks()

with app:
  gr.Markdown("# <center>RVC一键翻唱、重磅更新！</center>")
  gr.Markdown("## 自动分离人声翻唱并合并，自动混音！</center>")
  
  with gr.Row():
    with gr.Column():
      # 模型选择区域
      with gr.Row():
        model_dropdown = gr.Dropdown(
          label="选择AI模型", 
          choices=[], 
          value=None,
          info="请先点击刷新加载模型列表"
        )
        refresh_btn = gr.Button("🔄 刷新", size="sm")
        switch_btn = gr.Button("✨ 切换模型", size="sm", variant="primary")
      with gr.Row(visible=False):  # 隐藏这个功能
          models_json = gr.JSON()
          get_models_btn = gr.Button("获取模型列表", visible=False)
          get_models_btn.click(show_model, outputs=models_json)
      # 模型状态显示
      with gr.Row():
        model_status = gr.Textbox(label="模型状态", value="请选择模型", interactive=False)
      
      with gr.Row():
        inp1 = gr.Textbox(label="请填写想要AI翻唱的网易云id或链接", placeholder="114514", info="直接填写网易云id或链接")
      
      with gr.Row():
        inp5 = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label="歌曲人声升降调", info="默认为0，+2为升高2个key，以此类推")
        inp6 = gr.Slider(minimum=-3, maximum=3, value=0, step=0.5, label="调节人声音量，默认为0")
        inp7 = gr.Slider(minimum=-3, maximum=3, value=0, step=0.5, label="调节伴奏音量，默认为0")
      # ========== 新增：混响强度滑块 ==========
      with gr.Row():
        inp_reverb = gr.Slider(
            minimum=0, maximum=10, value=4, step=0.5,
            label="混响强度",
            info="0为干声，4为默认值，10为宏大混响"
        )
      # ========================================
        inp_delay = gr.Slider(
            minimum=0, maximum=10, value=0, step=0.5,
            label="回声(延迟)效果",
            info="0为关闭，数值越大回声越明显"
        )
      btn = gr.Button("一键开启AI翻唱之旅吧💕", variant="primary")
    
    with gr.Column():
      out = gr.Audio(label="AI歌手为您倾情演唱的歌曲🎶", type="filepath", interactive=False,streaming=True,)
      cache_flag = gr.Textbox(visible=False)

  # 绑定事件
  refresh_btn.click(refresh_models, outputs=model_dropdown,api_name=False)
  switch_btn.click(switch_model, inputs=model_dropdown, outputs=model_status)
  btn.click(convert, [inp1, inp5, inp6, inp7,model_dropdown, inp_reverb, inp_delay], [out, cache_flag], api_name=False)
  api_model_name = gr.Textbox(visible=False)
  api_f0_method = gr.Dropdown(choices=["rmvpe", "harvest", "crepe", "pm"], value="rmvpe", visible=False)
  api_index_rate = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.75, visible=False)
  api_filter_radius = gr.Slider(minimum=0, maximum=7, step=1, value=3, visible=False)
  api_msst_batch_size = gr.Number(value=1, visible=False)
  api_msst_num_overlap = gr.Number(value=4, visible=False)
  api_msst_normalize = gr.Checkbox(value=False, visible=False)
  api_msst_use_tta = gr.Checkbox(value=False, visible=False)
  api_msst_model = gr.Textbox(value=DEFAULT_MODEL_ID, visible=False)
  api_uvr5_agg = gr.Slider(minimum=0, maximum=20, step=1, value=10, visible=False)
  api_uvr5_tta = gr.Checkbox(value=False, visible=False)
  api_uvr5_postprocess = gr.Checkbox(value=False, visible=False)
  api_uvr5_window_size = gr.Dropdown(choices=[256, 512, 1024], value=512, visible=False)
  api_uvr5_high_end_process = gr.Dropdown(choices=["mirroring", "none"], value="mirroring", visible=False)
  api_shift_accompaniment = gr.Checkbox(value=True, visible=False)
  api_output = gr.Audio(visible=False)
  api_cache_flag = gr.Textbox(visible=False)
  gr.Button("API Show Model", visible=False).click(
      fn=lambda: models,
      inputs=[],
      outputs=[gr.JSON(visible=False)],
      api_name="show_model"
  )
  gr.Button("API Show MSST Models", visible=False).click(
      fn=show_msst_models_api,
      inputs=[],
      outputs=[gr.JSON(visible=False)],
      api_name="show_msst_models"
  )
  gr.Button("API Select MSST Model", visible=False).click(
      fn=select_msst_model_api,
      inputs=[api_msst_model],
      outputs=[gr.JSON(visible=False)],
      api_name="select_msst_model"
  )
  gr.Button("API Convert", visible=False).click(
      convert,
      inputs=[inp1, inp5, inp6, inp7, api_model_name, inp_reverb, inp_delay, api_f0_method, api_index_rate, api_filter_radius, api_uvr5_agg, api_uvr5_tta, api_uvr5_postprocess, api_uvr5_window_size, api_uvr5_high_end_process, api_msst_batch_size, api_msst_num_overlap, api_msst_normalize, api_msst_use_tta, api_msst_model, api_shift_accompaniment],
      outputs=[api_output, api_cache_flag],
      api_name="convert"
  )
  gr.Markdown("### <center>注意❗：请不要生成会对个人以及组织造成侵害的内容，此程序仅供科研、学习及个人娱乐使用。</center>")
  gr.HTML('''
      <div class="footer">
                  <p>🌊🏞️🎶 - 江水东流急，滔滔无尽声。 明·顾璘
                  </p>
      </div>
  ''')


print("正在初始化并加载模型列表...")
initial_models = show_model()
if initial_models:
    print(f"成功加载 {len(initial_models)} 个模型")
else:
    print("⚠️ 警告: 未能加载模型列表，请确保RVC服务正在运行")


app.queue(max_size=40, api_open=True)
app.launch(server_name="0.0.0.0", server_port=3333, share=True, show_error=True)
