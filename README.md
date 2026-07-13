# RVCSVC-API-MSST

面向 [astrbot_plugin_matsuko_cover](https://github.com/sdfsfsk/matsuko_cover) 的高质量 RVC / SVC-Fusion 中间层，使用 BS-Roformer / MSST 分离人声和伴奏，并提供 Gradio API、分离模型切换、结果缓存、音频后处理和进度回传。

> [!CAUTION]
> **此版本仅支持 Windows AMD 显卡（A 卡）。推荐使用原生 Windows ROCm 7.2.1；NVIDIA、Intel GPU 和纯 CPU 环境不在支持范围内。**

本仓库仅发布源码和安装脚本，不包含 Python/ROCm 环境、ZLUDA 环境、MSST 权重、RVC/SVC 模型、歌曲、缓存或生成音频。

## 与 UVR5 版本的区别

| 项目 | RVCSVC-API-MSST | [RVCSVC-API-amd](https://github.com/sdfsfsk/RVCSVC-API-amd) |
|---|---|---|
| 分离器 | BS-Roformer / MSST | UVR5 / HP5 |
| GPU 后端 | Windows AMD ROCm 7.2.1 | DirectML |
| 特点 | 分离质量更高，可切换模型 | 环境较轻、兼容范围较广 |
| 支持显卡 | 仅 AMD | 仅 AMD |

两套中间层使用相同的 3333/9999 端口，不能同时启动。

## 数据流和端口

```text
AstrBot + matsuko_cover
  ├─ RVC 请求 → RVCSVC-API-MSST :3333 → MSST → RVC :2333
  └─ SVC 请求 → RVCSVC-API-MSST :9999 → MSST → SVC-Fusion :7777
```

## 安装原生 AMD ROCm 环境

要求：

- Windows 10/11 x64
- 支持 Windows ROCm 的 AMD Radeon GPU 与驱动
- [uv](https://docs.astral.sh/uv/) 已加入 `PATH`
- FFmpeg 已加入 `PATH`
- 已准备上游 RVC 或 SVC-Fusion 服务

运行：

```text
install_rocm_env.bat
```

脚本会创建 `runtime-rocm/` Python 3.12 环境，并根据 `requirements-rocm.txt` 安装 AMD 官方 Windows ROCm 7.2.1、PyTorch `2.9.1+rocm7.2.1` 及中间层依赖。

`install_env.bat` 会转发到同一个原生 ROCm 安装脚本。旧 `env/` + ZLUDA 仅作为已有本地环境的兼容回退，本仓库不分发该环境，也不建议新安装使用。

## MSST 模型

将分离模型放到：

```text
msst/pretrain/vocal_models/
```

当前配置支持以下文件名：

```text
model_bs_roformer_ep_317_sdr_12.9755.ckpt
bs_roformer_karaoke_frazer_becruily.ckpt
```

对应 YAML 配置已经包含在 `msst/configs/vocal_models/`。权重需从模型作者或 [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) 认可的来源自行获取，本仓库不分发模型文件。

## 启动

先启动上游推理服务，再启动需要的中间层：

```text
启动 rvcapi.bat   # 中间层 3333，需要 RVC 2333
启动 svcapi.bat   # 中间层 9999，需要 SVC-Fusion 7777
```

手动启动：

```powershell
runtime-rocm\Scripts\python.exe launch.py app_rvc.py --is_nohalf
runtime-rocm\Scripts\python.exe launch.py app_svc.py --is_nohalf
```

插件默认地址：

```text
rvc_base_url = http://127.0.0.1:3333/
svc_base_url = http://127.0.0.1:9999/
```

## API

- `/convert`：下载/读取歌曲、MSST 分离、调用上游、混音并返回结果。
- `/show_model`：读取上游 RVC/SVC 模型。
- `/show_msst_models`：返回可用分离模型列表。
- `/select_msst_model`：验证并切换默认 MSST 分离模型。
- `/cache_info`：返回结果缓存与分离缓存的文件数、占用空间。
- `/clear_cache`：在无推理任务运行时清理 `all`、`results` 或 `separation` 缓存。
- 结果按歌曲、音色模型、分离模型及全部关键参数哈希保存在 `temp/`。
- `app.queue(..., api_open=True)` 必须保持启用，否则 AstrBot 的 `gradio_client` 无法调用。

## 性能与质量建议

- `model_bs_roformer_ep_317_sdr_12.9755.ckpt` 适合作为默认高质量人声模型。
- 默认让当前 MSST 模型常驻显存，连续点歌不再每首重复加载权重；显存紧张时设置 `MSST_KEEP_MODEL_LOADED=0` 恢复每次释放。
- 本地音频按内容 SHA256 缓存，RVC/SVC 与 MSST 模型资产变化都会使旧结果缓存自动失效。
- 默认关闭二次 EQ/压缩/混响以保留上游音质；插件开启 `vocal_postprocess` 后才应用这些效果。最终导出带削峰保护，默认 320 kbps。
- `RVCSVC_CACHE_MAX_FILES`（默认 200）限制缓存增长；服务默认只监听 `127.0.0.1`，可通过 `RVCSVC_HOST` 显式修改。
- 16 GB 显存建议从 `batch_size=2`、`num_overlap=2～4` 开始。
- 增大 overlap 或启用 TTA 可能提高平滑度，但会增加耗时和显存占用。
- 长歌曲在 AMD 上可能需要较长时间，插件 `inference_timeout` 建议设置到 `9000` 秒。

## 注意事项

- 服务会调用本机上游端口，不要将 3333/9999 直接暴露到公网。
- `output/`、`temp/` 和下载音频可能包含受版权保护内容，已默认被 Git 忽略。
- Windows ROCm 仍有硬件和算子兼容范围；安装脚本通过 `torch.cuda.is_available()` 验证 AMD GPU。
- 本项目不会自动提供上游 RVC/SVC 或 MSST 权重，使用者需自行遵守模型和音频许可。

## 来源与许可状态

中间层基于 [CCYellowStar2/RVCSVC-API](https://github.com/CCYellowStar2/RVCSVC-API) 修改；该上游仓库在本项目发布时未声明开源许可证，因此本仓库不擅自为继承代码授予额外许可。MSST/BS-Roformer 第三方组件及其许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
