######################### 第一阶段：临时用于准备文件或构建 #########################

FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime AS builder

# 替换软件源为清华镜像
RUN sed -i 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# 防止交互式安装，完全不交互，使用默认值
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# 更新源并安装依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget curl \
    build-essential \
    gcc g++ make \
    xz-utils \
    libgl1 \
    libglib2.0-0 \
    git \
    unzip sox libsox-dev \
    lsb-release gnupg ca-certificates \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# 将项目源代码复制到容器中
COPY . /code

# 下载并解压 FFmpeg 6.0.1 静态版本
RUN mkdir -p /opt/ffmpeg \
    && wget -q -O /tmp/ffmpeg.tar.xz https://file.wddcn.com/lib/ffmpeg/ffmpeg-6.0.1-amd64-static.tar.xz \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1 \
    && rm -f /tmp/ffmpeg.tar.xz

######################### 第二阶段：正式构建最终镜像 #########################

# 使用 PyTorch 官方 CUDA 运行时镜像
# https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# 替换软件源为清华镜像
RUN sed -i 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

ARG BUILD_TIME

LABEL org.opencontainers.image.created=$BUILD_TIME

# 防止交互式安装，完全不交互，使用默认值
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# 设置时区
ENV TZ=Asia/Shanghai

# 更新源并安装依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
    wget curl \
    build-essential \
    gcc g++ make \
    xz-utils \
    libgl1 \
    libglib2.0-0 \
    git \
    unzip sox libsox-dev \
    lsb-release gnupg ca-certificates \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# 设置时区
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# 复制FFmpeg与环境变量
COPY --from=builder /opt/ffmpeg /usr/local/ffmpeg

ENV FFMPEG_PATH="/usr/local/ffmpeg"
ENV PATH="${FFMPEG_PATH}:${PATH}"

# 将项目源代码复制到容器中
COPY --from=builder /code /code

# 设置容器内工作目录为 /code
WORKDIR /code

# 阿里云镜像源
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple

# 升级 pip、setuptools、wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 升级 pip 并安装 Python 依赖：
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir -r api_requirements.txt \
    && rm -rf /root/.cache/pip /tmp/*

# 安装 fasttext
WORKDIR /code/third_party/fastText

RUN pip install --no-cache-dir . \
    && rm -rf /root/.cache/pip /tmp/*

# 设置容器内工作目录为 /code
WORKDIR /code

# 暴露端口
EXPOSE 9983

COPY --from=builder /code/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]