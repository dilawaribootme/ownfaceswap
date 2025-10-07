# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

# Build arguments for this stage with sensible defaults for standalone builds
ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli + dependencies needed by it to install ComfyUI
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# Upgrade PyTorch if needed (for newer CUDA versions)
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install custom nodes required for face swap workflow
# Install custom nodes using comfy-cli (more reliable method)
RUN comfy --workspace /comfyui node install comfyui-reactor-node || \
    (cd custom_nodes && git clone https://github.com/Gourieff/comfyui-reactor-node.git && \
    cd comfyui-reactor-node && \
    if [ -f requirements.txt ]; then uv pip install -r requirements.txt; fi)

RUN comfy --workspace /comfyui node install comfy_mtb || \
    (cd custom_nodes && git clone https://github.com/melMass/comfy_mtb --depth 1 && \
    cd comfy_mtb && \
    if [ -f requirements.txt ]; then uv pip install -r requirements.txt; fi)

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client

# Add application code and scripts
ADD src/start.sh handler.py test_input.json ./
RUN chmod +x /start.sh

# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

# Prevent pip from asking for confirmation during uninstall steps in custom nodes
ENV PIP_NO_INPUT=1

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Set the default command to run when starting the container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
# Set default model type if none is provided
ARG MODEL_TYPE=flux1-dev-fp8

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories upfront
RUN mkdir -p models/checkpoints models/vae models/unet models/clip models/upscale_models models/insightface models/facerestore_models

# Download existing models based on model type (keeping your original logic)
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -q -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -q -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -q -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "sd3" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-dev-fp8" ]; then \
      wget -q -O models/checkpoints/flux1-dev-fp8.safetensors https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors; \
    fi

# Download face swap specific models (always downloaded regardless of MODEL_TYPE)
# Download face swap specific models (always downloaded regardless of MODEL_TYPE)
# Note: These downloads have fallback URLs and will warn if failed but won't stop the build
# Download Absolute Reality checkpoint (try multiple sources)
# Download Absolute Reality checkpoint
RUN wget --no-verbose --show-progress --timeout=30 --tries=3 \
    -O models/checkpoints/absolutereality_v181.safetensors \
RUN wget -q -O models/checkpoints/absolutereality_v181.safetensors \
    "https://civitai.com/api/download/models/132760?type=Model&format=SafeTensor&size=pruned&fp=fp16" \
    || wget --no-verbose --timeout=30 --tries=3 \
    -O models/checkpoints/absolutereality_v181.safetensors \
    "https://huggingface.co/kayfahaarukku/AbsoluteReality_v1.8.1/resolve/main/absolutereality_v181.safetensors" \
    https://huggingface.co/Lykon/AbsoluteReality/resolve/main/absolutereality_v181.safetensors
    || (echo "WARNING: Could not download Absolute Reality model. You'll need to add it manually." && true)
# Download 4x upscale model
# Download 4x upscale model
RUN wget --no-verbose --show-progress --timeout=30 --tries=3 \
    -O models/upscale_models/4x_foolhardy_Remacri.pth \
RUN wget -q -O models/upscale_models/4x_foolhardy_Remacri.pth \
    "https://huggingface.co/gemasai/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth" \
    https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth
    || (echo "WARNING: Could not download 4x upscale model. You'll need to add it manually." && true)
# Download InsightFace model for ReActor  
# Download InsightFace model for ReActor
RUN wget --no-verbose --show-progress --timeout=30 --tries=3 \
    -O models/insightface/inswapper_128.onnx \
RUN wget -q -O models/insightface/inswapper_128.onnx \
    "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx" \
    || wget --no-verbose --timeout=30 --tries=3 \
    -O models/insightface/inswapper_128.onnx \
    "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx" \
    https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx
    || (echo "WARNING: Could not download inswapper model. You'll need to add it manually." && true)
# Download GFPGAN face restoration model
# Download GFPGAN face restoration model
RUN wget --no-verbose --show-progress --timeout=30 --tries=3 \
    -O models/facerestore_models/GFPGANv1.4.pth \
RUN wget -q -O models/facerestore_models/GFPGANv1.4.pth \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
    || (echo "WARNING: Could not download GFPGAN model. You'll need to add it manually." && true)
# Download detection model for ReActor (YOLOv5)
# Download detection model for ReActor (YOLOv5)
RUN mkdir -p models/ultralytics/bbox && \
RUN mkdir -p models/ultralytics/bbox && \
    wget --no-verbose --show-progress --timeout=30 --tries=3 \
    -O models/ultralytics/bbox/face_yolov5n.pt \
    wget -q -O models/ultralytics/bbox/face_yolov5n.pt \
    "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/detection/bbox/face_yolov5n.pt" \
    https://huggingface.co/Gourieff/ReActor/resolve/main/models/detection/bbox/face_yolov5n.pt
    || (echo "WARNING: Could not download detection model. You'll need to add it manually." && true)

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models