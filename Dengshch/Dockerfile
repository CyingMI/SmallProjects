# 使用CUDA 12.9官方基础镜像
FROM nvidia/cuda:12.9.0-cudnn8-runtime-ubuntu22.04

# 安装Python和基础工具（使用非交互式安装）
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户并设置密码（UID 1000是Linux标准非特权用户）
RUN useradd -m -u 1000 -s /bin/bash developer && \
    echo "developer:developer" | chpasswd && \
    usermod -aG sudo developer && \
    echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/developer

# 创建应用目录并设置权限
RUN mkdir -p /workspace && \
    chown -R developer:developer /workspace

# 切换到非root用户
USER developer
WORKDIR /workspace

# 设置Python用户安装路径
ENV PYTHONUSERBASE=/home/developer/.local
ENV PATH="${PYTHONUSERBASE}/bin:${PATH}"
ENV PYTHONPATH="${PYTHONUSERBASE}/lib/python3.11/site-packages:${PYTHONPATH}"

# 复制依赖文件（保持用户权限）
COPY --chown=developer:developer requirements.txt .

# 安装Python包到用户目录（不使用sudo，完全非root）
RUN pip install --user --no-warn-script-location --no-cache-dir -r requirements.txt

# 复制项目代码
COPY --chown=developer:developer . .

# 设置默认命令
CMD ["/bin/bash"]
