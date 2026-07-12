# MSST 模型下载说明

本目录需要以下 BS-Roformer 人声分离模型。模型权重体积较大，不随本仓库分发。

| 文件名 | 推荐下载源 | 大小 | SHA-256 |
| --- | --- | ---: | --- |
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | [UVR 公共模型 GitHub Release](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt) | 639,331,213 字节 | `5B84F37E8D444C8CB30C79D77F613A41C05868FF9C9AC6C7049C00AEFAE115AA` |
| `bs_roformer_karaoke_frazer_becruily.ckpt` | [作者 Hugging Face 仓库](https://huggingface.co/becruily/bs-roformer-karaoke/resolve/main/bs_roformer_karaoke_frazer_becruily.ckpt) | 204,436,907 字节 | `EB90EE24C1154D83FBCFD27E96182F19E061557CC6E4746953125E08C29389F9` |

`bs_roformer_karaoke_frazer_becruily.ckpt` 若无法访问 Hugging Face，可使用原 RVCSVC-API 所引用的 [ModelScope 镜像](https://modelscope.cn/models/CCYellowStar/bs_roformer_karaoke_frazer_becruily/resolve/master/bs_roformer_karaoke_frazer_becruily.ckpt)。

## 一键下载

在本目录打开 PowerShell，执行：

```powershell
curl.exe -L "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt" -o "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

curl.exe -L "https://modelscope.cn/models/CCYellowStar/bs_roformer_karaoke_frazer_becruily/resolve/master/bs_roformer_karaoke_frazer_becruily.ckpt" -o "bs_roformer_karaoke_frazer_becruily.ckpt"
```

下载完成后必须保持文件名不变，并放在当前目录：

```text
msst/pretrain/vocal_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt
msst/pretrain/vocal_models/bs_roformer_karaoke_frazer_becruily.ckpt
```

与权重配套的 YAML 配置已包含在 `msst/configs/vocal_models/` 中。

## 校验文件

```powershell
Get-FileHash -Algorithm SHA256 ".\model_bs_roformer_ep_317_sdr_12.9755.ckpt"
Get-FileHash -Algorithm SHA256 ".\bs_roformer_karaoke_frazer_becruily.ckpt"
```

输出哈希应与上表一致。若不一致，请删除文件后重新下载。
