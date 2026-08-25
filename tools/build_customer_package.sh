#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VERSION="${1:-1.0.0-$(date +%Y%m%d)}"
JOBS="${JOBS:-2}"
CYTHON_PYTHON="${CYTHON_PYTHON:-${PROJECT_ROOT}/.venv-package/bin/python}"

if [[ ! "${VERSION}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[打包失败] 版本号只能包含字母、数字、点、下划线和连字符。" >&2
    exit 2
fi
if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[打包失败] JOBS 必须是正整数。" >&2
    exit 2
fi

PACKAGE_NAME="CVIA_Runtime_${VERSION}_aarch64"
RELEASE_ROOT="${PROJECT_ROOT}/release"
BUILD_DIR="${PROJECT_ROOT}/build-customer"
FINAL_DIR="${RELEASE_ROOT}/${PACKAGE_NAME}"
ARCHIVE_PATH="${RELEASE_ROOT}/${PACKAGE_NAME}.tar.gz"
mkdir -p "${RELEASE_ROOT}"

if [[ -e "${FINAL_DIR}" || -e "${ARCHIVE_PATH}" ]]; then
    echo "[打包失败] 版本 ${VERSION} 已存在；请换一个版本号，现有交付物不会被覆盖。" >&2
    exit 3
fi

STAGING_ROOT="$(mktemp -d "${RELEASE_ROOT}/.staging.XXXXXX")"
cleanup() {
    if [[ -n "${STAGING_ROOT:-}" && "${STAGING_ROOT}" == "${RELEASE_ROOT}/.staging."* && -d "${STAGING_ROOT}" ]]; then
        rm -rf -- "${STAGING_ROOT}"
    fi
}
trap cleanup EXIT INT TERM

PACKAGE_ROOT="${STAGING_ROOT}/${PACKAGE_NAME}"
mkdir -p \
    "${PACKAGE_ROOT}/bin" \
    "${PACKAGE_ROOT}/config" \
    "${PACKAGE_ROOT}/data/source" \
    "${PACKAGE_ROOT}/models/engine" \
    "${PACKAGE_ROOT}/web_monitor/mocap/bin" \
    "${PACKAGE_ROOT}/web_monitor/mocap/lib/aarch64" \
    "${PACKAGE_ROOT}/web_monitor/runtime" \
    "${PACKAGE_ROOT}/web_monitor/static"

echo "[1/8] Release 编译核心推理程序..."
cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCVIA_PRODUCTION_BUILD=ON
cmake --build "${BUILD_DIR}" --target trt -j"${JOBS}"

echo "[2/8] 编译客户启动器与 NOKOV 桥接程序..."
"${CXX:-g++}" -std=c++14 -O2 -DNDEBUG -s \
    "${PROJECT_ROOT}/delivery/launcher.cpp" \
    -o "${PACKAGE_ROOT}/cvia_customer"
"${CXX:-g++}" -std=c++14 -O3 -DNDEBUG -s -pthread \
    -I"${PROJECT_ROOT}/web_monitor/mocap/include" \
    "${PROJECT_ROOT}/web_monitor/mocap/MocapBridge.cpp" \
    -L"${PROJECT_ROOT}/web_monitor/mocap/lib/aarch64" \
    -Wl,-rpath,'$ORIGIN/../lib/aarch64' \
    -lnokov_sdk \
    -o "${PACKAGE_ROOT}/web_monitor/mocap/bin/MocapBridge"

echo "[3/8] 按白名单复制运行文件..."
install -m 0755 "${BUILD_DIR}/trt" "${PACKAGE_ROOT}/bin/cvia_runtime"
strip --strip-all "${PACKAGE_ROOT}/bin/cvia_runtime"
strip --strip-all "${PACKAGE_ROOT}/cvia_customer"
strip --strip-all "${PACKAGE_ROOT}/web_monitor/mocap/bin/MocapBridge"
install -m 0644 \
    "${PROJECT_ROOT}/models/engine/qdy0815-fp16.engine" \
    "${PACKAGE_ROOT}/models/engine/qdy0815-fp16.engine"
install -m 0644 \
    "${PROJECT_ROOT}/web_monitor/mocap/lib/aarch64/libnokov_sdk.so" \
    "${PACKAGE_ROOT}/web_monitor/mocap/lib/aarch64/libnokov_sdk.so"
install -m 0644 "${PROJECT_ROOT}/web_monitor/static/index.html" "${PACKAGE_ROOT}/web_monitor/static/index.html"
install -m 0644 "${PROJECT_ROOT}/web_monitor/static/app.css" "${PACKAGE_ROOT}/web_monitor/static/app.css"
install -m 0644 "${PROJECT_ROOT}/web_monitor/static/app.js" "${PACKAGE_ROOT}/web_monitor/static/app.js"
install -m 0644 "${PROJECT_ROOT}/web_monitor/static/extrinsic_rotation.js" "${PACKAGE_ROOT}/web_monitor/static/extrinsic_rotation.js"
install -m 0644 "${PROJECT_ROOT}/delivery/CUSTOMER_README.txt" "${PACKAGE_ROOT}/CUSTOMER_README.txt"
install -m 0644 "${PROJECT_ROOT}/delivery/THIRD_PARTY_NOTICES.txt" "${PACKAGE_ROOT}/THIRD_PARTY_NOTICES.txt"

echo "[4/8] 把 Python 后端编译为 ARM64 原生模块（不交付 .py/.pyc）..."
if [[ ! -x "${CYTHON_PYTHON}" ]] || ! "${CYTHON_PYTHON}" -c 'import Cython' 2>/dev/null; then
    echo "[打包失败] 缺少 Cython 构建环境：${CYTHON_PYTHON}" >&2
    echo "请在本机构建虚拟环境中安装 Cython 3，或通过 CYTHON_PYTHON 指定解释器。" >&2
    exit 9
fi
BACKEND_ROOT="${STAGING_ROOT}/native-backend"
mkdir -p "${BACKEND_ROOT}/source" "${BACKEND_ROOT}/build"
install -m 0644 "${PROJECT_ROOT}/web_monitor/server.py" "${BACKEND_ROOT}/source/server.py"
install -m 0644 "${PROJECT_ROOT}/web_monitor/mocap_receiver.py" "${BACKEND_ROOT}/source/mocap_receiver.py"
install -m 0644 "${PROJECT_ROOT}/web_monitor/offline_sync.py" "${BACKEND_ROOT}/source/offline_sync.py"
"${CYTHON_PYTHON}" "${PROJECT_ROOT}/tools/build_native_backend.py" \
    --source "${BACKEND_ROOT}/source" \
    --output "${PACKAGE_ROOT}/web_monitor" \
    --build "${BACKEND_ROOT}/build"
strip --strip-all "${PACKAGE_ROOT}/web_monitor/server.so"
strip --strip-all "${PACKAGE_ROOT}/web_monitor/mocap_receiver.so"
strip --strip-all "${PACKAGE_ROOT}/web_monitor/offline_sync.so"

echo "[5/8] 生成脱敏的客户初始配置..."
python3 - "${PROJECT_ROOT}/web_monitor/config.json" "${PACKAGE_ROOT}/config/config.json" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
config = json.loads(source.read_text(encoding="utf-8"))
config["input"].update({
    "mode": "camera",
    "folder_path": "data/source",
    "loop": True,
})
config["network"].update({
    "tcp_enabled": False,
    "tcp_ip": "127.0.0.1",
    "udp_enabled": False,
    "udp_ip": "127.0.0.1",
})
config["mocap"].update({
    "enabled": False,
    "server": "127.0.0.1",
})
config["runtime"]["preview_dir"] = "web_monitor/runtime"
destination.write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
printf '[]\n' > "${PACKAGE_ROOT}/web_monitor/calibration_history.json"
printf '%s\n' '把 JPG/JPEG/PNG/BMP 图片放在此目录，然后在网页选择“文件夹”输入模式。' > "${PACKAGE_ROOT}/data/source/使用说明.txt"

echo "[6/8] 记录目标平台与动态依赖..."
{
    echo "CVIA customer runtime build information"
    echo "package=${PACKAGE_NAME}"
    echo "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "architecture=$(uname -m)"
    echo "python=$({ python3 --version 2>&1; } | head -1)"
    echo "source_revision=$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unavailable)"
    echo
    echo "Core executable:"
    file -b "${PACKAGE_ROOT}/bin/cvia_runtime"
    echo
    echo "Direct dynamic dependencies:"
    ldd "${PACKAGE_ROOT}/bin/cvia_runtime"
    echo
    echo "Relevant installed packages:"
    dpkg-query -W -f='${Package} ${Version}\n' 2>/dev/null | \
        grep -E '^(libnvinfer|libopencv|cuda-|nvidia-jetpack|zed)' || true
} > "${PACKAGE_ROOT}/BUILD_INFO.txt"

echo "[7/8] 执行源码泄漏、架构、依赖和启动检查..."
FORBIDDEN_FILES="$(find "${PACKAGE_ROOT}" -type f \( \
    -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o \
    -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.cu' -o \
    -name '*.py' -o -name '*.pyc' -o -name '*.onnx' -o -name 'CMakeLists.txt' -o \
    -name 'Makefile' \) -print)"
if [[ -n "${FORBIDDEN_FILES}" ]]; then
    echo "[打包失败] 交付目录混入禁止文件：" >&2
    echo "${FORBIDDEN_FILES}" >&2
    exit 10
fi
if strings "${PACKAGE_ROOT}/bin/cvia_runtime" | grep -Fq "${PROJECT_ROOT}"; then
    echo "[打包失败] 核心程序仍包含开发工作区绝对路径。" >&2
    exit 11
fi
if grep -R -a -Fq '/home/wts' "${PACKAGE_ROOT}"; then
    echo "[打包失败] 交付文件仍包含开发机绝对路径 /home/wts。" >&2
    exit 16
fi
if file "${PACKAGE_ROOT}/bin/cvia_runtime" | grep -Fq 'not stripped'; then
    echo "[打包失败] 核心程序仍包含符号表。" >&2
    exit 12
fi
if ! file "${PACKAGE_ROOT}/bin/cvia_runtime" | grep -Fq 'ARM aarch64'; then
    echo "[打包失败] 核心程序不是 ARM64 可执行文件。" >&2
    exit 13
fi
if ldd "${PACKAGE_ROOT}/bin/cvia_runtime" | grep -Fq 'not found'; then
    echo "[打包失败] 构建机缺少核心程序动态库。" >&2
    exit 14
fi
if LD_LIBRARY_PATH="${PACKAGE_ROOT}/web_monitor/mocap/lib/aarch64:${LD_LIBRARY_PATH:-}" \
    ldd "${PACKAGE_ROOT}/web_monitor/mocap/bin/MocapBridge" | grep -Fq 'not found'; then
    echo "[打包失败] NOKOV 桥接程序存在缺失动态库。" >&2
    exit 15
fi
"${PACKAGE_ROOT}/cvia_customer" --help >/dev/null
(
    cd "${PACKAGE_ROOT}"
    "${PACKAGE_ROOT}/bin/cvia_runtime" --check-assets config/config.json >/dev/null
)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PACKAGE_ROOT}/web_monitor" \
    python3 -OO -c 'import server; server.main()' --help >/dev/null

echo "[8/8] 生成完整性清单与压缩包..."
(
    cd "${PACKAGE_ROOT}"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
rm -rf -- "${BACKEND_ROOT}"
mv -- "${PACKAGE_ROOT}" "${FINAL_DIR}"
rmdir -- "${STAGING_ROOT}"
STAGING_ROOT=""
tar -C "${RELEASE_ROOT}" -czf "${ARCHIVE_PATH}" "${PACKAGE_NAME}"
(
    cd "${RELEASE_ROOT}"
    sha256sum "${PACKAGE_NAME}.tar.gz" > "${PACKAGE_NAME}.tar.gz.sha256"
)

echo
echo "[完成] 客户运行目录：${FINAL_DIR}"
echo "[完成] 客户压缩包：  ${ARCHIVE_PATH}"
echo "[完成] 外部校验文件：${ARCHIVE_PATH}.sha256"
echo "[提示] 发送前请确认 THIRD_PARTY_NOTICES.txt 中的 NOKOV SDK 再分发许可。"
