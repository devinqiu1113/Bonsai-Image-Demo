# Bonsai-Image-Demo Jetson Orin 部署报告

> 部署日期: 2026-06-09 ~ 2026-06-10
> 平台: Jetson Orin (ARM64 aarch64, CUDA 12.6, 7.4 GB 统一内存)
> 模型: Bonsai Image 4B (ternary 2-bit + binary 1-bit, gemlite/HQQ 量化)
> Python: 3.10.16 | venv: `.venv` | 框架: diffusers + gemlite + HQQ + Triton

---

## 一、部署坑点 (Pitfalls)

### 1. ❌ 环境依赖版本冲突 (Python 3.11 → 3.10)
- **问题**: 上游 `pyproject.toml` 要求 `>=3.11`，但 JetPack 系统只有 Python 3.10
- **修复**: 将 `requires-python` 降级为 `>=3.10`，锁定 `torch>=2.5.0a0,<2.5.1` 防止 uv 拉取 PyPI 上的 x86_64 torch 导致崩溃
- **影响**: uv sync 必须在 JetPack 原生 torch 环境下运行，否则会下载 2GB + 无用的 nvidia-* wheel

### 2. ❌ Git 代理导致 clone 失败
- **问题**: `setup.sh` 的 `git submodule update` 因全局 `https.proxy=https://192.168.1.42:4067` 配置错误（代理 URL 应该用 `http://` 而非 `https://`）导致 TLS 连接中断
- **修复**: 改为 `http://192.168.1.42:4067`，或临时取消代理
- **教训**: git 使用 `CONNECT` 方法通过 HTTP 代理访问 HTTPS 远程仓库时，代理 URL 本身必须用 `http://`

### 3. ❌ Triton + ptxas 找不到
- **问题**: Jetson Orin 的 CUDA 12.6 的 ptxas 不在默认 PATH 上（在 `/usr/local/cuda-12.6/bin/ptxas`），Triton JIT 编译时 `RuntimeError: Cannot find ptxas`
- **修复**: 在 `scripts/local_backend.py` 和 `scripts/serve.sh` 中设置 `TRITON_PTXAS_PATH=/usr/local/cuda-12.6/bin/ptxas`
- **影响**: text encoder 的 HQQ→gemlite 转换阶段会触发 Triton 编译 `pack_weights_over_cols` kernel，没有 ptxas 则直接崩溃

### 4. ❌ OOM: text encoder + transformer 无法同时驻留 GPU
- **问题**: text encoder (HQQ-4bit, ~2.7GB 加载峰值) 和 transformer (gemlite, ~1.5GB ternary / ~1.1GB binary) 同时加载超过 7.4GB 统一内存上限
- **修复**: 重构 `generate_png()` 为三步流水线：① 删除 transformer → ② 加载 text encoder 编码 prompt → ③ 释放 text encoder + 重载 transformer。不在 prewarm 时加载 text encoder
- **附加**: VAE decode 阶段同样需要清理 transformer + intermediates，否则 CUDA OOM

### 5. ❌ WebUI 前端按钮无法点击 (局域网访问)
- **问题**: Next.js dev 模式默认只允许 `127.0.0.1` 的 HMR WebSocket 连接，局域网 IP 访问时 `onClick` 事件不生效
- **修复**: 修改 `vendor/image-studio/frontend/next.config.ts`，`allowedDevOrigins` 添加 `192.168.1.192` 和 `192.168.1.203`
- **影响**: 不修复则 WebUI 在局域网其他设备上完全不可用

### 6. ❌ npm 无法直接从 venv 使用
- **问题**: `nodejs-wheel-binaries` 安装的 npm 可执行文件在 `.venv/bin/`，但 `setup.sh` 的 `scripts/` 目录搜索不到
- **修复**: `serve.sh` 中显式使用 `.venv/bin/npm`，并在前端命令前追加 `PATH="$VENV_BIN:$PATH"`
- **影响**: 否则 npm install / next dev 会调用系统 node（可能缺失或版本不兼容）

### 7. ❌ uv sync 尝试拉取 macOS/Windows 依赖
- **问题**: `pyproject.toml` 中 `mflux ; sys_platform == 'darwin'` 这类条件依赖让 uv 尝试解析所有平台，在 ARM64 Linux 上安装时出错
- **修复**: 在 `[tool.uv]` 中添加 `environments = ["sys_platform == 'linux'"]`，仅解析当前平台

---

## 二、关键修复 (Key Fixes)

### 2.1 pipeline_gpu.py — OOM 流水线重构
- **text encoder 按需加载**: 从 `prewarm()` 中移除 text encoder 加载，改为 `generate_png()` 内三步流程（删 transformer → 加载TE编码 → 删TE重载transformer）
- **prompt_embeds 预编码**: 在释放 text encoder 前完成 prompt 编码，将结果传给 `diffusion_forward()`，避免函数内部再次加载 text encoder
- **错误恢复**: `generate_png()` 中 `except Exception` 分支做 3 次 `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.synchronize()`，确保失败后 CUDA 状态干净

### 2.2 diffusion_klein.py — 显存释放
- **VAE decode 前释放**: diffusion 循环完成后立即 `del transformer` + `del intermediates` + 多次 gc/empty_cache/sync，为 2.7GB 的 VAE decode 腾出空间
- **`import gc`**: 修复原代码中 `_gc` 未定义的 bug
- **可选 prompt_embeds 参数**: 支持接收预编码的 prompt embeddings，当提供此参数时跳过内部的 text encoder 加载

### 2.3 pyproject.toml — 环境兼容
- Python 版本从 `>=3.11` 降低到 `>=3.10`
- torch 版本锁定 `>=2.5.0a0,<2.5.1`（JetPack 预装）
- uv 环境限制为 `sys_platform == 'linux'`

### 2.4 TRITON_PTXAS_PATH 自动设置
- `local_backend.py`: 检测 `/usr/local/cuda-12.6/bin/ptxas` 是否存在并设置
- `serve.sh`: 在 `env` 启动命令中直接追加 `TRITON_PTXAS_PATH`

### 2.5 WebUI 局域网访问
- `next.config.ts`: `allowedDevOrigins` 扩展为 `["127.0.0.1", "192.168.1.192", "192.168.1.203", "localhost"]`

### 2.6 gemlite 内存优化
- `_load_gemlite_transformer()`: 在 `load_state_dict` 前主动 `del state` + `gc.collect()`，减少峰值内存
- gemlite buffer 从 256MB 降至 32MB（`GEMLITE_BUFFER_SIZE` 相关调整）

---

## 三、模型对比测试 (Benchmark)

| 指标 | Ternary (2-bit) | Binary (1-bit) | 差异 |
|------|----------------|----------------|------|
| 模型文件大小 | 1.5 GB | 1.1 GB | -27% |
| 峰值显存 | 4,642 MiB | 3,143 MiB | **-32%** |
| 首轮生成 (冷启) | ~70s | ~291s (含TE JIT) | binary 慢 |
| 缓存后生成 | ~35s | ~35s | 相同 |
| 图片质量 | 良好 | 可接受，细节略差 | ternary 更好 |

**TTFT 分析 (缓存后, binary):**
1. 删除 transformer + gc: ~0.5s
2. 加载 text encoder (HQQ, 2.7GB 峰值): ~18s
3. prompt 编码 + 释放 TE: ~2s
4. 重载 transformer (gemlite, 1.1GB): ~2.5s
5. diffusion 循环 (4 steps, 512x512): ~8s
6. VAE decode + 后处理: ~4s
**总计: ~35s**

Binary 模型虽然显存节省 1.5GB，但 text encoder 加载步骤（~18s）是固定的瓶颈，和 ternary 一样无法跳过。

---

## 四、最终系统架构

```
┌─────────────────────────────────────────────────┐
│  Web 访问                                       │
│  http://192.168.1.192:3000/  (Next.js 前端)     │
│  http://192.168.1.192:8898/  (FastAPI 后端)     │
├─────────────────────────────────────────────────┤
│  scripts/serve.sh                               │
│  ├── BONSAI_VARIANT=ternary|binary              │
│  ├── backend: uvicorn scripts.local_backend:app │
│  │     ├── backend_gpu/pipeline_gpu.py          │
│  │     │   ├── _load_gemlite_transformer()      │
│  │     │   ├── _load_text_encoder() (按需)      │
│  │     │   ├── _load_vae()                      │
│  │     │   └── generate_png() (OOM-safe 流水线) │
│  │     └── backend_gpu/diffusion_klein.py       │
│  │         └── diffusion_forward()              │
│  └── frontend: next dev :3000                   │
│        └── vendor/image-studio/frontend/        │
├─────────────────────────────────────────────────┤
│  Models: ~/Bonsai-Image-Demo/models/            │
│  ├── bonsai-image-4B-ternary-gemlite/ (1.5 GB)  │
│  └── bonsai-image-4B-binary-gemlite/  (1.1 GB)  │
│        ├── transformer-gemlite-int1/            │
│        ├── text_encoder-hqq-4bit/               │
│        └── vae/                                 │
└─────────────────────────────────────────────────┘
```

---

## 五、使用方式

### 启动 WebUI
```bash
# Ternary (2-bit) — 默认
./scripts/serve.sh

# Binary (1-bit)
BONSAI_VARIANT=binary ./scripts/serve.sh

# 指定端口
BACKEND_PORT=8898 FRONTEND_PORT=3000 ./scripts/serve.sh
```

### API 直接调用 (测试用)
```bash
curl -X POST http://192.168.1.192:8898/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene Japanese garden with cherry blossoms",
    "backend": "bonsai-binary-gemlite",
    "width": 512, "height": 512,
    "steps": 4, "seed": 42
  }' --output output.png
```

### 模型切换
通过 WebUI 前端下拉框切换 ternary/binary，或通过 API 的 `backend` 字段。

---

## 六、已知限制

1. **text encoder 不能常驻 GPU**: ~2.7GB 加载峰值 + ~1.5GB transformer 超出 7.4GB 统一内存上限，需要每请求卸载/重载（~21s 额外开销）
2. **首轮慢**: Triton JIT 编译 gemlite 和 HQQ kernel 耗时 ~2-5 分钟（缓存后消失，缓存位于 `~/.triton/cache/`）
3. **scheduler 不存在**: binary 模型目录中没有 `scheduler/` 子目录，使用 FLUX.2 默认调度器
4. **不支持 1024x1024**: 显存不足，建议保持 512x512
5. **Triton 版本锁定**: 基于 ARM64 自编译 Triton 3.1.0，不能直接升级或降级

---

*报告生成日期: 2026-06-10*
*by Hermes Agent (小O)*
