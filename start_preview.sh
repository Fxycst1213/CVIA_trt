#!/usr/bin/env bash
# Start on the Jetson; keep the PC's SSH forward for localhost:8080 open.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    printf '用法：./start_preview.sh [trt 参数，例如 --jpeg-quality 70]\n'
    printf '启动网页接收与相机推理，关闭串口输出；Ctrl+C 停止本脚本启动的进程。\n'
    printf '本机浏览器：http://127.0.0.1:8080（需保持 SSH 端口转发）\n'
    exit 0
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if [[ ! -x ./trt ]]; then
    printf '找不到可执行文件 %s/trt，请先运行 make。\n' "$project_dir" >&2
    exit 1
fi

preview_status() {
    python3 - <<'PY'
import json
import sys
from urllib.request import urlopen

try:
    with urlopen('http://127.0.0.1:8080/status.json', timeout=0.5) as response:
        status = json.load(response)
    if not isinstance(status.get('connected'), bool) or 'received' not in status:
        sys.exit(1)
    print('streaming' if status['connected'] and status['received'] > 0 else 'waiting')
except (OSError, ValueError):
    sys.exit(1)
PY
}

child_pids=()
cleanup() {
    local pid
    # Only stop children created here; leave pre-existing preview processes alone.
    for pid in "${child_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${child_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

status="$(preview_status 2>/dev/null || true)"
if [[ "$status" == streaming ]]; then
    printf '预览已在运行，直接打开 http://127.0.0.1:8080\n'
    exit 0
fi

log_dir="$project_dir/build/preview"
mkdir -p "$log_dir"
if [[ -z "$status" ]]; then
    python3 -u tools/server.py --host 127.0.0.1 --no-display >"$log_dir/server.log" 2>&1 &
    server_pid=$!
    child_pids+=("$server_pid")
    for ((attempt = 0; attempt < 40; attempt++)); do
        status="$(preview_status 2>/dev/null || true)"
        [[ -n "$status" ]] && break
        if ! kill -0 "$server_pid" 2>/dev/null; then
            cat "$log_dir/server.log" >&2
            exit 1
        fi
        sleep 0.25
    done
    if [[ -z "$status" ]]; then
        printf '网页服务启动失败，请检查 %s/server.log\n' "$log_dir" >&2
        exit 1
    fi
fi

if pgrep -x trt >/dev/null; then
    printf '检测到已有 trt 进程，等待它连接本机预览服务。\n'
else
    ./trt --preview --coordinate-frames --no-serial "$@" >"$log_dir/trt.log" 2>&1 &
    child_pids+=("$!")
fi

printf '等待相机数据，日志目录：%s\n' "$log_dir"
for ((attempt = 0; attempt < 120; attempt++)); do
    status="$(preview_status 2>/dev/null || true)"
    [[ "$status" == streaming ]] && break
    for pid in "${child_pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            printf '预览进程退出，请检查 %s 中的日志。\n' "$log_dir" >&2
            exit 1
        fi
    done
    sleep 0.25
done
if [[ "$status" != streaming ]]; then
    printf '未收到相机数据；已有 trt 必须使用 --preview。日志：%s\n' "$log_dir" >&2
    exit 1
fi

printf '相机数据已连接，打开 http://127.0.0.1:8080；Ctrl+C 停止。\n'
if ((${#child_pids[@]})); then
    wait -n "${child_pids[@]}"
fi
