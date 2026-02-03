#!/bin/bash
# 自动安装与 NVIDIA 驱动版本匹配的 libnvidia-gl 包

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}NVIDIA libnvidia-gl 自动安装脚本${NC}"
echo -e "${BLUE}============================================================${NC}"

# 步骤 1: 从 nvidia-smi 获取驱动版本
echo -e "\n${YELLOW}[步骤 1] 检测 NVIDIA 驱动版本...${NC}"

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ 错误: nvidia-smi 命令未找到${NC}"
    echo -e "${RED}  请确保已安装 NVIDIA 驱动${NC}"
    exit 1
fi

# 获取驱动版本 (从 nvidia-smi 输出解析)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')

if [ -z "$DRIVER_VERSION" ]; then
    echo -e "${RED}✗ 错误: 无法获取驱动版本${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 检测到 NVIDIA 驱动版本: ${DRIVER_VERSION}${NC}"

# 提取大版本号 (例如: 570.195.03 -> 570)
MAJOR_VERSION=$(echo "$DRIVER_VERSION" | cut -d'.' -f1)
echo -e "${GREEN}✓ 驱动大版本号: ${MAJOR_VERSION}${NC}"

# 步骤 2: 搜索对应的 libnvidia-gl 包
echo -e "\n${YELLOW}[步骤 2] 搜索匹配的 libnvidia-gl 包...${NC}"

PACKAGE_NAME="libnvidia-gl-${MAJOR_VERSION}"
echo -e "${BLUE}  查找包名: ${PACKAGE_NAME}${NC}"

# 检查包是否存在
if ! apt-cache show "$PACKAGE_NAME" &> /dev/null; then
    echo -e "${RED}✗ 错误: 包 ${PACKAGE_NAME} 在软件源中不存在${NC}"
    echo -e "${YELLOW}  可能的原因:${NC}"
    echo -e "    1. 该版本的驱动没有对应的 libnvidia-gl 包"
    echo -e "    2. 需要添加额外的软件源"
    echo -e "    3. 需要运行 'apt update' 更新软件源"
    exit 1
fi

echo -e "${GREEN}✓ 找到包: ${PACKAGE_NAME}${NC}"

# 步骤 3: 查找精确匹配的版本
echo -e "\n${YELLOW}[步骤 3] 查找精确匹配的版本...${NC}"

# 获取所有可用版本
echo -e "${BLUE}  可用的版本列表:${NC}"
apt-cache madison "$PACKAGE_NAME" | head -10

# 查找与驱动版本完全匹配的包版本
EXACT_VERSION=$(apt-cache madison "$PACKAGE_NAME" | grep -E "^[[:space:]]*${PACKAGE_NAME}[[:space:]]*\|[[:space:]]*${DRIVER_VERSION}" | head -1 | awk -F'|' '{print $2}' | tr -d '[:space:]')

if [ -n "$EXACT_VERSION" ]; then
    echo -e "${GREEN}✓ 找到精确匹配的版本: ${EXACT_VERSION}${NC}"
    INSTALL_VERSION="${PACKAGE_NAME}=${EXACT_VERSION}"
else
    echo -e "${YELLOW}⚠ 未找到精确匹配的版本 ${DRIVER_VERSION}${NC}"
    echo -e "${YELLOW}  将安装最新的 ${MAJOR_VERSION} 系列版本${NC}"
    
    # 获取该大版本的最新版本
    LATEST_VERSION=$(apt-cache madison "$PACKAGE_NAME" | head -1 | awk -F'|' '{print $2}' | tr -d '[:space:]')
    
    if [ -z "$LATEST_VERSION" ]; then
        echo -e "${RED}✗ 错误: 无法确定要安装的版本${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ 将安装版本: ${LATEST_VERSION}${NC}"
    INSTALL_VERSION="${PACKAGE_NAME}=${LATEST_VERSION}"
fi

# 检查当前安装状态
echo -e "\n${YELLOW}[步骤 4] 检查当前安装状态...${NC}"

if dpkg -l | grep -q "^ii.*${PACKAGE_NAME}"; then
    INSTALLED_VERSION=$(dpkg -l | grep "^ii.*${PACKAGE_NAME}" | awk '{print $3}')
    echo -e "${BLUE}  当前已安装版本: ${INSTALLED_VERSION}${NC}"
    
    if [ "$INSTALLED_VERSION" = "$EXACT_VERSION" ] || [ "$INSTALLED_VERSION" = "$LATEST_VERSION" ]; then
        echo -e "${GREEN}✓ 已安装的版本与目标版本一致，无需操作${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠ 已安装版本与目标版本不一致，将进行更新/降级${NC}"
    fi
else
    echo -e "${BLUE}  ${PACKAGE_NAME} 未安装${NC}"
fi

# 步骤 5: 安装包
echo -e "\n${YELLOW}[步骤 5] 安装/更新 ${PACKAGE_NAME}...${NC}"
echo -e "${BLUE}  执行命令: apt install -y --allow-downgrades ${INSTALL_VERSION}${NC}"
echo ""

# 询问用户确认（可选）
if [ "${AUTO_CONFIRM:-0}" != "1" ]; then
    read -p "是否继续安装? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}安装已取消${NC}"
        exit 0
    fi
fi

# 执行安装
if apt install -y --allow-downgrades "$INSTALL_VERSION"; then
    echo -e "\n${GREEN}✅ 成功安装 ${INSTALL_VERSION}${NC}"
    
    # 验证安装
    echo -e "\n${YELLOW}[验证] 检查安装结果...${NC}"
    INSTALLED_VERSION=$(dpkg -l | grep "^ii.*${PACKAGE_NAME}" | awk '{print $3}')
    echo -e "${GREEN}✓ 已安装版本: ${INSTALLED_VERSION}${NC}"
    
    # 显示相关文件
    echo -e "\n${YELLOW}[信息] 已安装的库文件:${NC}"
    dpkg -L "$PACKAGE_NAME" | grep -E "\.so" | head -5
    
else
    echo -e "\n${RED}✗ 安装失败${NC}"
    exit 1
fi

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${GREEN}✅ 脚本执行完成！${NC}"
echo -e "${BLUE}============================================================${NC}"