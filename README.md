# ComfyUI-H3-Heaton

MiniMax H3 的纯净自用节点包。包内业务节点统一使用 `HT_H3_BUNDLE` 和成对的 AV latent；第三方包只通过 `compat/` 注册适配器接入，不修改第三方文件，也不 monkey-patch。

## 节点对照

| HT 节点 | 替代/统一了哪个包的哪个节点 |
|---|---|
| HT · H3 Unified Loader | `ComfyUI-FeiHou-Easy-H3` 的 `FeiHouEasyH3Loader`（统一成 HT_H3_BUNDLE） |
| HT · H3 Sigma Preset | `ComfyUI-YCNodes-MiniMax-H3` 的 `H3SigmaRefiner`，并对齐 ComfyUI `beta_scheduler` / `SplitSigmas` |
| HT · H3 Turbo LoRA | `comfyui-minimax-h3-turbo` 的 `MiniMaxH3TurboLoRA`（只挂一采） |
| HT · AV Latent Separate / Concat | `ComfyUI-LTXVideo` AV 分合语义与 `ComfyUI-PT_H3ConcatAVLatent` |
| HT · AV Latent Upscale 3D | `Comfyui_Minimax_h3_latent_Upscaler` 的 `MinimaxH3LatentUpscaler3D` 几何 3D 放大路径 |
| HT · MC Context / Trim / SaveLatent / LoadLatent | `ComfyUI-H3-Motion-Context` 四节点语义 |
| HT · H3 Audio Refine Mask | `Adudeguyman/ComfyUI-H3-AudioRefine` 的逐流噪声掩码核心（视频冻结/音频重精修，接原生采样器 denoise<1.0；未吸收其 KV-cache 补丁与 all-in-one 采样器） |
| HT · QA Video Check / Latent Probe | H3 出片前自检；无第三方等价节点 |

## 兼容层

`compat/registry.py` 自动发现 `adapter_*.py`。当前适配器是 FeiHou 的 `MINIMAX_H3_BUNDLE` 双向转换。目标包缺失时适配器下线，不影响其他节点；转换错误包含目标包名并直接抛出。新增适配器只需新增文件、继承 `AdapterBase`、实现转换、使用 `@register_adapter`。

H3 latent 必须同时包含视频流和音频流；纯视频 VAE latent 不能接入 AV 分合节点。Create Video 时请同时将 `video_vae` 和 `audio_vae` 分别接入原生 `VAEDecode` / `VAEDecodeAudio`。二采模型节点的隐藏输入 `second_sampling_output_connected` 由工作流 API 显式传 `true`；Turbo LoRA 业务默认只挂一采，不能把同一输出接到二采。

MC 接龙续链段禁止连接 `first_frame`；应使用上一段的 AV latent 作为 context。保存节点带槽位名，便于重试时加载明确的上一段。

## 开发检查

```text
D:/project/comfyui/new/venv/Scripts/python.exe -c "import ast,glob;[ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('**/*.py',recursive=True)]"
D:/project/comfyui/new/venv/Scripts/python.exe tests/smoke_import.py
```
