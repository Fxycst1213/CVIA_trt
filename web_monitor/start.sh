#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TRT_BINARY="${TRT_BINARY:-${PROJECT_ROOT}/trt}"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/config.json}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-8765}"
WEB_PORT_MAX_TRIES="${WEB_PORT_MAX_TRIES:-20}"
WEB_PID=""
TRT_PID=""
TRT_PID_FILE="${SCRIPT_DIR}/runtime/trt.pid"

port_is_available() {
    python3 - "${WEB_HOST}" "$1" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    if [[ -n "${TRT_PID}" ]] && kill -0 "${TRT_PID}" 2>/dev/null; then
        echo
        echo "[CVIA] 正在优雅停止推理进程（PID ${TRT_PID}）..."
        kill -TERM "${TRT_PID}" 2>/dev/null || true
        wait "${TRT_PID}" 2>/dev/null || true
    fi
    rm -f "${TRT_PID_FILE}"

    if [[ -n "${WEB_PID}" ]] && kill -0 "${WEB_PID}" 2>/dev/null; then
        echo
        echo "[CVIA] 正在关闭网页服务（PID ${WEB_PID}）..."
        kill "${WEB_PID}" 2>/dev/null || true
        wait "${WEB_PID}" 2>/dev/null || true
    fi

    exit "${exit_code}"
}

trap cleanup EXIT INT TERM

if [[ ! -x "${TRT_BINARY}" ]]; then
    echo "[CVIA] 未找到可执行的推理程序：${TRT_BINARY}" >&2
    echo "[CVIA] 请先自行完成 C++ 编译，然后重新运行本脚本。" >&2
    exit 1
fi

if ! grep -aFq "CVIA runtime revision: dual60-latest-frame-v3" "${TRT_BINARY}"; then
    echo "[CVIA] 当前推理程序不是本次双 60 FPS 优化后的版本：${TRT_BINARY}" >&2
    echo "[CVIA] 请重新编译，或通过 TRT_BINARY 指向新生成的 trt。" >&2
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[CVIA] 未找到配置文件：${CONFIG_FILE}" >&2
    exit 1
fi

if [[ ! "${WEB_PORT}" =~ ^[0-9]+$ ]] || (( WEB_PORT < 1 || WEB_PORT > 65535 )); then
    echo "[CVIA] WEB_PORT 必须是 1～65535 之间的整数，当前值：${WEB_PORT}" >&2
    exit 1
fi
if [[ ! "${WEB_PORT_MAX_TRIES}" =~ ^[0-9]+$ ]] || (( WEB_PORT_MAX_TRIES < 1 )); then
    echo "[CVIA] WEB_PORT_MAX_TRIES 必须是正整数，当前值：${WEB_PORT_MAX_TRIES}" >&2
    exit 1
fi

REQUESTED_WEB_PORT="${WEB_PORT}"
PORT_FOUND=false
for ((PORT_ATTEMPT = 0; PORT_ATTEMPT < WEB_PORT_MAX_TRIES; PORT_ATTEMPT++)); do
    if (( WEB_PORT > 65535 )); then
        break
    fi
    if port_is_available "${WEB_PORT}"; then
        PORT_FOUND=true
        break
    fi
    WEB_PORT=$((WEB_PORT + 1))
done

if [[ "${PORT_FOUND}" != true ]]; then
    LAST_WEB_PORT=$((WEB_PORT - 1))
    echo "[CVIA] 无可用网页端口：已检查 ${REQUESTED_WEB_PORT}～${LAST_WEB_PORT}。" >&2
    echo "[CVIA] 可通过 WEB_PORT 指定其他端口，例如：WEB_PORT=9000 web_monitor/start.sh" >&2
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/runtime"
rm -f "${TRT_PID_FILE}"
cd "${PROJECT_ROOT}"

echo "[CVIA] 项目目录：${PROJECT_ROOT}"
echo "[CVIA] 配置文件：${CONFIG_FILE}"
if [[ "${WEB_PORT}" != "${REQUESTED_WEB_PORT}" ]]; then
    echo "[CVIA] 端口 ${REQUESTED_WEB_PORT} 已被占用，自动改用 ${WEB_PORT}。"
fi
echo "[CVIA] 正在启动网页控制台 http://${WEB_HOST}:${WEB_PORT} ..."

python3 -u "${SCRIPT_DIR}/server.py" --host "${WEB_HOST}" --port "${WEB_PORT}" &
WEB_PID=$!

# 给网页进程一个短暂的初始化窗口，并确认它没有因端口占用等原因退出。
sleep 0.6
if ! kill -0 "${WEB_PID}" 2>/dev/null; then
    wait "${WEB_PID}" || true
    echo "[CVIA] 网页服务启动失败，请检查上方错误（尝试端口：${WEB_PORT}）。" >&2
    exit 1
fi

echo "[CVIA] 网页服务已启动（PID ${WEB_PID}）"
if [[ "${WEB_HOST}" == "0.0.0.0" ]]; then
    echo "[CVIA] 已监听全部网络接口；同一局域网设备可通过 AGX Orin 的 IP 访问。"
    LOCAL_IPS="$(hostname -I 2>/dev/null || true)"
    if [[ -n "${LOCAL_IPS// }" ]]; then
        for LOCAL_IP in ${LOCAL_IPS}; do
            [[ "${LOCAL_IP}" == *:* ]] && continue
            echo "[CVIA] 局域网地址：http://${LOCAL_IP}:${WEB_PORT}"
        done
    else
        echo "[CVIA] 局域网地址：http://<AGX-Orin-IP>:${WEB_PORT}"
    fi
    echo "[CVIA] 安全提示：网页没有登录鉴权，请仅在可信局域网使用。"
else
    echo "[CVIA] 网页地址：http://${WEB_HOST}:${WEB_PORT}"
fi
echo "[CVIA] 正在启动推理程序。完成采集后请在网页点击“停止推理并保存”。"
echo "[CVIA] 终端 Ctrl+C 会同时关闭推理和网页服务，不会弹出笔记本另存为面板。"
echo

"${TRT_BINARY}" "${CONFIG_FILE}" &
TRT_PID=$!
printf '%s\n' "${TRT_PID}" > "${TRT_PID_FILE}"

TRT_EXIT=0
wait "${TRT_PID}" || TRT_EXIT=$?
TRT_PID=""
rm -f "${TRT_PID_FILE}"

echo
echo "[CVIA] 推理进程已停止（退出码 ${TRT_EXIT}）。"
echo "[CVIA] 网页服务继续运行，请在浏览器中选择位置并保存本次 PnP CSV。"
echo "[CVIA] 保存完成后回到终端按 Ctrl+C 关闭网页服务。"

wait "${WEB_PID}"
