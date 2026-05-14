#!/bin/bash
# CHD-AIDE 依赖一键安装脚本 (Linux/macOS)
# 使用方法: bash setup.sh

set -e

echo "========================================"
echo "CHD-AIDE 环境安装脚本"
echo "========================================"

# 1. 检查 Python 版本
echo "[1/7] 检查 Python 版本..."
python3 --version || { echo "错误: 未找到 python3"; exit 1; }

PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info[1])')
if [ "$PYTHON_VERSION" -lt 8 ]; then
    echo "错误: Python 版本需要 3.8+，当前版本: $(python3 --version)"
    exit 1
fi
echo "✅ Python 版本检查通过: $(python3 --version)"

# 2. 安装 ffmpeg
echo "[2/7] 安装 ffmpeg..."
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

# 3. 删除旧虚拟环境（如果存在）
echo "[3/7] 清理旧虚拟环境..."
if [ -d "CHD_venv" ]; then
    echo "🗑️ 删除已存在的 CHD_venv..."
    rm -rf CHD_venv
    echo "✅ 已删除 CHD_venv"
else
    echo "ℹ️ 无旧虚拟环境需要清理"
fi

# 4. 创建虚拟环境
echo "[4/7] 创建虚拟环境..."
# Debian/Ubuntu: 确保 python3-venv 和 python3-pip 已安装
if command -v apt-get &> /dev/null; then
    echo "📦 安装 python3-venv 和 python3-pip..."
    sudo apt-get update
    sudo apt-get install -y python3.10-venv python3-pip
fi

if python3 -m venv CHD_venv; then
    echo "✅ 虚拟环境已创建: CHD_venv"
else
    echo "❌ 虚拟环境创建失败"
    exit 1
fi

# 5. 激活虚拟环境
echo "[5/7] 激活虚拟环境..."
if [ -f "CHD_venv/bin/activate" ]; then
    source CHD_venv/bin/activate
    echo "✅ 虚拟环境已激活 (CHD_venv)"
else
    echo "❌ 虚拟环境激活文件不存在"
    exit 1
fi

# 6. 安装 Python 依赖
echo "[6/7] 安装 Python 依赖..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Python 依赖安装完成"

# 7. 添加 .gitignore 条目（如需要）
echo "[7/7] 配置 gitignore..."
GITIGNORE_FILE=".gitignore"
if [ -f "$GITIGNORE_FILE" ]; then
    if grep -q "^CHD_venv/$" "$GITIGNORE_FILE" 2>/dev/null; then
        echo "ℹ️ CHD_venv/ 已在 .gitignore 中，跳过"
    else
        echo "CHD_venv/" >> "$GITIGNORE_FILE"
        echo "✅ 已在 .gitignore 中添加 CHD_venv/"
    fi
else
    echo "⚠️ .gitignore 文件不存在，跳过"
fi

# 安装模型目录额外依赖（如果有）
if [ -f "model/requirements.txt" ]; then
    echo "📦 安装模型额外依赖..."
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