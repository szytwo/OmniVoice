## fork

https://github.com/QwenLM/Qwen3-TTS

## 安装

```
python -m venv venv
venv\Scripts\activate

pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r ./api_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .

hf download k2-fsa/OmniVoice --local-dir checkpoints/OmniVoice

nvidia-smi -L  # 查看GUID

# sox下载
https://sourceforge.net/projects/sox/

# flash attention wheel下载
https://mjunya.com/flash-attention-prebuild-wheels/

下载并安装（Windows 专用）：
https://aka.ms/vs/17/release/vc_redist.x64.exe
安装完成后 重启终端或电脑。

cuda+驱动下载地址
https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
https://developer.nvidia.com/cuda-toolkit-archive

cudnn下载地址
https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html
https://developer.nvidia.com/cudnn-archive

# 版本参考：https://pytorch.org/get-started/previous-versions/

```

## GIT

```

git pull # 拉取
git push # 推送

git submodule add https://github.com/szytwo/fastText.git third_party/fastText # 添加子模块
git submodule update --init --recursive # 初始化子模块

git branch -r # 查看分支
git branch -m main # 重命名分支
git branch --set-upstream-to=origin/main main #关联远程分支origin/main 

git remote -v # 查看远程仓库
git remote remove origin # 移除远程仓库连接，origin，upstream

# 添加新的远程仓库，origin，upstream
git remote add upstream https://github.com/szytwo/Qwen3-TTS.git

git fetch upstream # 从远程仓库拉取更新，origin，upstream
git checkout main # 切换到主分支
git merge upstream/main # 合并到本地分支,主分支名称可能是 ，origin，upstream，master,main 

git reset --hard origin/main # 强制覆盖本地代码

```