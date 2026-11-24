# ERNIE-5.5-Flash-Preview on NVIDIA Blackwell sm_120 (RTX PRO 6000 Blackwell / RTX 50-series)
# Base: vLLM 0.32.0 (CUDA 13.0, PyTorch 2.13.0+cu130) pinned by digest.
# Plugin: baidu/vLLM-ERNIE pinned by commit + local sm_120 FlashInfer sparse-MLA dispatch patch.
# Installing with --no-deps keeps all base torch/vllm/flashinfer wheels intact.
FROM vllm/vllm-openai:v0.32.0-x86_64@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9

LABEL org.opencontainers.image.source="https://github.com/milesangelo/vllm-ernie55-sm120"
LABEL org.opencontainers.image.description="vLLM 0.32.0 + baidu/vLLM-ERNIE with NVIDIA Blackwell sm_120 HCA & sparse-MLA attention-sink dispatch"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ARG VLLM_ERNIE_COMMIT=ba322a5634f91d3e6f96ae711df72ddaf3c38b6e

COPY patches/vllm-ernie-sm120.patch /tmp/vllm-ernie-sm120.patch

RUN mkdir -p /tmp/vllm-ernie && \
    curl -fsSL "https://github.com/baidu/vLLM-ERNIE/archive/${VLLM_ERNIE_COMMIT}.tar.gz" \
      | tar -xz -C /tmp/vllm-ernie --strip-components=1 && \
    patch -d /tmp/vllm-ernie -p1 < /tmp/vllm-ernie-sm120.patch && \
    pip install --no-cache-dir --no-deps --no-build-isolation /tmp/vllm-ernie && \
    rm -rf /tmp/vllm-ernie /tmp/vllm-ernie-sm120.patch && \
    python3 -c "import importlib.metadata, vllm; eps = {e.name: e.value for e in importlib.metadata.entry_points(group='vllm.general_plugins')}; assert 'ernie' in eps, eps; import vllm_ernie.models.ernie5_5.nvidia.attention as a; assert hasattr(a, 'Ernie5_5HCAFlashInferSM120Attention') and hasattr(a, 'Ernie5_5FlashInferMLASparseSM120Backend'); print('vllm', vllm.__version__, 'plugins', eps, 'sm120_ok')"
