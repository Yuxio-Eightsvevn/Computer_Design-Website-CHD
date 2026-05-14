#!/bin/bash
# CHD-AIDE 依赖一键安装脚本 (Linux/macOS)
# 使用方法: bash setup.sh

set -e

echo "========================================"
echo "CHD-AIDE 环境安装脚本"
echo "========================================"

# 1. 检查 Python 版本
echo "[1/6] 检查 Python 版本..."
python3 --version || { echo "错误: 未找到 python3"; exit 1; }

PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info[1])')
if [ "$PYTHON_VERSION" -lt 8 ]; then
    echo "错误: Python 版本需要 3.8+，当前版本: $(python3 --version)"
    exit 1
fi
echo "✅ Python 版本检查通过: $(python3 --version)"

# 2. 安装 ffmpeg
echo "[2/6] 安装 ffmpeg..."
if command -v apt-get &> /dev/null; then
    # Debian/Ubuntu
    sudo apt-get update && sudo apt-get install -y ffmpeg
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    sudo yum install -y ffmpeg
elif command -v dnf &> /dev/null; then
    # Fedora
    sudo dnf install -y ffmpeg
elif command -v brew &> /dev/null; then
    # macOS
    brew install ffmpeg
else
    echo "⚠️ 无法自动安装 ffmpeg，请手动安装: https://ffmpeg.org/download.html"
fi
echo "✅ ffmpeg 检查完成"

# 3. 创建虚拟环境
echo "[3/6] 创建虚拟环境..."
if [ -d "CHD_venv" ]; then
    echo "⚠️ 虚拟环境已存在，跳过创建"
else
    python3 -m venv CHD_venv
    echo "✅ 虚拟环境已创建: CHD_venv"
fi

# 4. 激活虚拟环境
echo "[4/6] 激活虚拟环境..."
source CHD_venv/bin/activate
echo "✅ 虚拟环境已激活 (CHD_venv)"

# 5. 安装 Python 依赖
echo "[5/6] 安装 Python 依赖..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Python 依赖安装完成"

# 6. 安装模型目录额外依赖（如果有）
if [ -f "model/requirements.txt" ]; then
    echo "[6/6] 安装模型额外依赖..."
    pip install -r model/requirements.txt
    echo "✅ 模型依赖安装完成"
fi

echo "========================================"
echo "🎉 安装完成！"
echo "========================================"
echo ""
echo "下一步："
echo "1. 激活虚拟环境: source CHD_venv/bin/activate"
echo "2. 复制模型权重文件到 model/ 目录"
echo "3. 配置 LLM API（可选）: 编辑 config/llm_models.json"
echo "4. 启动服务: python main.py"
echo ""
echo "访问地址: http://localhost:11000/"
echo "========================================"