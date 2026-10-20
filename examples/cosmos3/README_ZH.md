# Cosmos3 分阶段推理

[English](README.md)

## Cosmos3-Nano + AgiBot

输入 AgiBot 的首帧和 16 步给定动作，用提示词 "Pickup items in the supermarket" 生成一段视频。样例输入已放在 [assets](assets/) 中。

准备包含 `transformer/`、`vae/` 和 `text_tokenizer/` 的 **Cosmos3-Nano** 本地权重，在仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python examples/cosmos3/run_cosmos3_staged.py \
    --checkpoint /path/to/Cosmos3-Nano \
    --sample examples/cosmos3/assets/agibot_forward_dynamics.json \
    --offload \
    --output .cache/cosmos3/agibot.pt
```

Demo 使用单张 GPU，依次运行 VLM → VAE encoder → DiT → VAE decoder。`--offload` 在每个阶段结束后将权重移回 CPU，文本 K/V 保留在 GPU。

输出为 `.cache/cosmos3/agibot.mp4`（17 帧、560×640、10 FPS）和 `agibot.pt`（视频、latent、给定动作）。样例使用第一个动作 chunk，去噪 30 步，guidance 为 1，seed 为 0。

## 自定义输入

不传 `--sample` 时，使用 [customized_inputs.py](customized_inputs.py) 生成合成观测和状态：6 个视觉流、49 个状态/动作 token，合计 583 个 GEN token，用于测试 API 和性能。

```bash
CUDA_VISIBLE_DEVICES=0 uv run python examples/cosmos3/run_cosmos3_staged.py \
    --checkpoint /path/to/Cosmos3-Nano-Policy-DROID \
    --steps 10 --seed 9000 \
    --warmup 3 --repeat 10 --offload \
    --output .cache/cosmos3/customized.pt
```

输出为 `.cache/cosmos3/` 下的 `customized.pt` 和 `customized.mp4`。加上 `--latent-input` 可直接使用合成 latent，跳过 VAE encoder；VLM、DiT 和 VAE decoder 仍会运行。

## 查看性能

测速时选择空闲 GPU。各阶段先运行 `--warmup` 次，再统计 `--repeat` 次，打印毫秒单位的 `run mean`（平均）、`median`（中位数）和 `min`（最小值）：

| 日志阶段 | 统计内容 |
| --- | --- |
| `vlm` | 文本 prefill |
| `vae_encode` | 当前请求的全部 VAE 编码 |
| `dit` | 完整一轮去噪，上述自定义输入命令为 10 步 |
| `vae_decode` | 输出视频解码 |

看 DiT 延迟时，找 `Cosmos3 dit: run mean=... ms`，它不含 VLM 和 VAE。`weight load`、`offload` 单独统计。预热、阶段外的输入预处理和文件保存不计入 `run`，因此四个阶段耗时相加也不是整个程序的总延迟。

默认开启 CUDA Graph。捕获日志如 `warmup=1 measured=0`，表示预热时捕获了图，计时期间没有重新捕获。保持其他参数相同，加上 `--no-cuda-graph` 可对比 eager 性能。

## AgiBot 帧对比

第 0、8、16 帧：上排为 Cosmos Framework，下排为 PHYAI。相同权重和输入下，解码视频的 PSNR 为 49.03 dB。

![Cosmos Framework 与 PHYAI 的 AgiBot 第 0、8、16 帧对比](assets/agibot_comparison.jpg)
