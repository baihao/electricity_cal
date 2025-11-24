#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
ENV_NAME="${CONDA_ENV_NAME:-electricity_cal}"
OLD_ENV_NAME=".python310"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "找不到环境文件: ${ENV_FILE}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "未检测到 conda，请先安装 Anaconda 或 Miniconda 并确保其在 PATH 中。" >&2
  exit 1
fi

if [[ "${ENV_NAME}" != "${OLD_ENV_NAME}" ]] && conda info --envs | awk '{print $1}' | grep -qx "${OLD_ENV_NAME}"; then
  echo "检测到旧环境 ${OLD_ENV_NAME}，开始删除..."
  conda env remove -n "${OLD_ENV_NAME}"
fi

if conda info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "检测到现有环境 ${ENV_NAME}，执行依赖更新..."
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}"
else
  echo "创建新的 conda 环境 ${ENV_NAME}..."
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

echo
echo "环境就绪。请执行以下命令启用环境："
echo "  conda activate ${ENV_NAME}"

