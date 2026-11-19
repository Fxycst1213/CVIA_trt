#!/usr/bin/env bash
# Orin owns MediaMTX and publishes locally. The PC/sky device only reads the stream.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
script_path="$project_dir/start_groundstation.sh"
log_dir="$project_dir/build/groundstation"
pid_file="$log_dir/supervisor.pid"
startup_file="$log_dir/startup.status"
configuration_file="$log_dir/configuration"
cd "$project_dir"

running_supervisor() {
    [[ -s "$pid_file" ]] || return 1
    local saved_pid arguments
    read -r saved_pid < "$pid_file"
    [[ "$saved_pid" =~ ^[0-9]+$ ]] || return 1
    arguments="$(ps -p "$saved_pid" -o args= 2>/dev/null || true)"
    [[ "$arguments" == *"$script_path --run" || "$arguments" == *"$script_path --run "* ]] || return 1
    printf '%s' "$saved_pid"
}

stop_supervisor() {
    local supervisor_pid attempt
    if ! supervisor_pid="$(running_supervisor)"; then return 0; fi
    kill "$supervisor_pid"
    for ((attempt = 0; attempt < 150; attempt++)); do
        if ! running_supervisor >/dev/null; then return 0; fi
        sleep 0.1
    done
    printf '停止信号已发送，进程尚未退出；请检查 %s。\n' "$log_dir" >&2
    return 1
}

if [[ "${1:-}" == --help ]]; then
    printf '用法：./start_groundstation.sh [--2d-only] [本地接收电脑IPv4]\n'
    printf 'MediaMTX 在 Orin 上运行；视频发布到本机，由天空端/电脑读取 Orin 的 :8554/live。\n'
    printf '可选电脑 IP 仅用于 UDP 9000 位姿接收；不填写时只输出带位姿的视频。\n'
    printf '示例：./start_groundstation.sh 192.168.2.61\n'
    printf '自动接管旧预览；后台运行；停止：./start_groundstation.sh --stop\n'
    printf '默认 RTSP 1920×1080、20 FPS、4 Mbps；推理输入保持不变。\n'
    printf '默认关闭网页；CVIA_WEB_ENABLE=1 可启用网页预览。\n'
    printf '默认 SDK 原始右目；CVIA_CAMERA_EYE=left 可切回左目，标定自动同步。\n'
    printf '默认启用 /dev/ttyUSB0 串口，9600 波特率，10 Hz 发送位姿。\n'
    exit 0
fi
if [[ "${1:-}" == --stop ]]; then
    stop_supervisor
    printf 'Orin 推流已停止。\n'
    exit 0
fi

foreground=false
if [[ "${1:-}" == --run ]]; then foreground=true; shift; fi
keypoints_only=false
groundstation_ip=""
for argument in "$@"; do
    case "$argument" in
        --2d-only) keypoints_only=true ;;
        --*) printf '未知选项：%s；参见 --help。\n' "$argument" >&2; exit 1 ;;
        *)
            if [[ -n "$groundstation_ip" ]]; then
                printf '只需填写一个可选的位姿接收电脑 IP。\n' >&2
                exit 1
            fi
            groundstation_ip="$argument"
            ;;
    esac
done
if [[ "$keypoints_only" == true ]]; then groundstation_ip=""; fi
# The first launch builds TensorRT engines and can take several minutes.
startup_timeout=600
camera_eye="${CVIA_CAMERA_EYE:-left}"
rtsp_port="${CVIA_RTSP_PORT:-8554}"
# Rendering/RTSP encoding only; this does not limit inference or TCP sending.
video_fps="${CVIA_VIDEO_FPS:-50}"
video_width="${CVIA_VIDEO_WIDTH:-1920}"
web_width="${CVIA_WEB_WIDTH:-1280}"
web_enabled="${CVIA_WEB_ENABLE:-1}"
web_port=0
if [[ "$web_enabled" == 1 ]]; then web_port=8080; fi
bitrate="${CVIA_BITRATE:-4M}"
pose_port="${CVIA_POSE_PORT:-9000}"
mediamtx_bin="${CVIA_MEDIAMTX_BIN:-$project_dir/tools/bin/mediamtx}"
rtsp_url="rtsp://127.0.0.1:$rtsp_port/live"

if [[ ! -x ./trt ]]; then printf '找不到 trt，请先运行 make。\n' >&2; exit 1; fi
if ! trt_help="$(./trt --help 2>/dev/null)" || [[ "$trt_help" != *--coordinate-frames* ]]; then
    printf 'trt 需要重新编译，请运行：./start_groundstation.sh --stop && make -j2 && ./start_groundstation.sh 192.168.2.61\n' >&2
    exit 1
fi
if [[ "$keypoints_only" == true && "$trt_help" != *--2d-only* ]]; then
    printf 'trt 尚不支持仅 2D 模式，请先运行 make -j2。\n' >&2
    exit 1
fi
# Validate output settings before interrupting a working preview or the previous mode.
python3 - "$groundstation_ip" "$rtsp_port" "$video_fps" "$bitrate" "$pose_port" "$video_width" "$web_width" "$web_enabled" "$camera_eye" <<'PY'
import ipaddress
import sys
from tools.groundstation_stream import RtspPublisher
try:
    if sys.argv[1]:
        ipaddress.IPv4Address(sys.argv[1])
    port, pose_port = int(sys.argv[2]), int(sys.argv[5])
    if not 1 <= port <= 65535 or not 1 <= pose_port <= 65535:
        raise ValueError('端口必须在 1..65535 范围')
    width, web_width = int(sys.argv[6]), int(sys.argv[7])
    if width < 0 or width > 1920 or width % 2 or (width != 0 and width < 64):
        raise ValueError('视频宽度必须为 0 或 64..1920 的偶数')
    if web_width < 0 or web_width > 4096 or (web_width != 0 and web_width < 64):
        raise ValueError('网页宽度必须为 0 或 64..4096')
    if sys.argv[8] not in ('0', '1'):
        raise ValueError('CVIA_WEB_ENABLE 必须为 0 或 1')
    if sys.argv[9] not in ('left', 'right'):
        raise ValueError('CVIA_CAMERA_EYE 必须为 left 或 right')
    RtspPublisher(f'rtsp://127.0.0.1:{port}/live', fps=int(sys.argv[3]), bitrate=sys.argv[4])
except ValueError as error:
    raise SystemExit(f'推流参数错误：{error}')
PY
if [[ ! -x "$mediamtx_bin" ]]; then
    if [[ -n "${CVIA_MEDIAMTX_BIN:-}" ]]; then
        printf 'MediaMTX 不可执行：%s\n' "$mediamtx_bin" >&2
        exit 1
    fi
    python3 tools/install_mediamtx.py
fi
mediamtx_bin="$(realpath "$mediamtx_bin")"
"$mediamtx_bin" --version >/dev/null
mkdir -p "$log_dir"
configuration="orin-local-v9-2d-keypoints|$keypoints_only|$groundstation_ip|$rtsp_port|$video_fps|$video_width|$web_width|$web_enabled|$bitrate|$pose_port|$camera_eye|$mediamtx_bin"

print_endpoints() {
    if [[ "$keypoints_only" == true ]]; then
        printf '当前为仅 2D 模式：循环读取照片，显示发送端的关键点坐标；PnP、LSTM、串口关闭。\n'
    fi
    python3 - "$groundstation_ip" "$rtsp_port" "$pose_port" "$web_enabled" <<'PY'
import ipaddress
import json
import subprocess
import sys
receiver, port, pose_port, web_enabled = sys.argv[1:]
result = subprocess.run(['ip', '-j', '-4', 'address', 'show', 'dev', 'eno1'], capture_output=True, text=True)
addresses = []
for interface in json.loads(result.stdout or '[]'):
    addresses += [ipaddress.IPv4Interface(f"{entry['local']}/{entry['prefixlen']}")
                  for entry in interface.get('addr_info', []) if entry.get('family') == 'inet']
matching = [address for address in addresses if receiver and ipaddress.IPv4Address(receiver) in address.network]
addresses = matching or addresses
print('MediaMTX 已在 Orin 运行，本地电脑/天空端读取：')
for address in addresses:
    print(f'  视频：rtsp://{address.ip}:{port}/live')
    if web_enabled == '1':
        print(f'  网页：http://{address.ip}:8080')
        print(f'  位姿 JSON：http://{address.ip}:8080/status.json')
if not addresses:
    print(f'  rtsp://<Orin网口IPv4>:{port}/live')
if receiver:
    print(f'  独立位姿：UDP {receiver}:{pose_port}（电脑只接收）')
if web_enabled == '0':
    print('  网页预览已关闭（CVIA_WEB_ENABLE=1 可开启）')
PY
    printf '日志：%s\n' "$log_dir"
}

if [[ "$foreground" == false ]]; then
    if running_supervisor >/dev/null; then
        if [[ -f "$configuration_file" && "$(cat "$configuration_file")" == "$configuration" ]]; then
            printf 'Orin 本机推流已在运行。\n'
            print_endpoints
            exit 0
        fi
        printf '正在切换旧推流进程到 Orin 本机 MediaMTX 模式……\n'
        stop_supervisor
    fi
    printf '%s\n' "$configuration" > "$configuration_file"
    printf 'starting\n' > "$startup_file"
    nohup setsid bash "$script_path" --run "$@" >"$log_dir/launcher.log" 2>&1 </dev/null &
    supervisor_pid=$!
    printf '%s\n' "$supervisor_pid" > "$pid_file"
    for ((attempt = 0; attempt < (startup_timeout + 15) * 4; attempt++)); do
        if ! kill -0 "$supervisor_pid" 2>/dev/null; then
            cat "$log_dir/launcher.log" >&2
            printf '启动失败，请检查 %s。\n' "$log_dir" >&2
            exit 1
        fi
        [[ "$(cat "$startup_file")" == ready ]] && break
        if ((attempt > 0 && attempt % 120 == 0)); then
            printf '仍在初始化；首次生成 TensorRT engine 可能需要数分钟。日志：%s/trt.log\n' "$log_dir"
        fi
        sleep 0.25
    done
    if [[ "$(cat "$startup_file")" != ready ]]; then
        printf '后台尚未完成初始化，请检查 %s。\n' "$log_dir" >&2
        exit 1
    fi
    printf 'Orin 推流已在后台启动，无需保持 SSH 连接。\n'
    print_endpoints
    exit 0
fi

child_pids=()
cleanup() {
    local index child
    for ((index = ${#child_pids[@]} - 1; index >= 0; index--)); do
        child="${child_pids[index]}"
        kill "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 - "$rtsp_port" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('0.0.0.0', int(sys.argv[1])))
    except OSError as error:
        raise SystemExit(f'Orin RTSP 端口已被其他服务占用：{error}')
PY
python3 - "$project_dir/tools/mediamtx_orin.yml" "$log_dir/mediamtx.yml" "$rtsp_port" <<'PY'
from pathlib import Path
import sys
source, destination, port = sys.argv[1:]
Path(destination).write_text(Path(source).read_text().replace('rtspAddress: :8554', f'rtspAddress: :{port}'))
PY
(
    cd "$log_dir"
    exec "$mediamtx_bin" "$log_dir/mediamtx.yml"
) >"$log_dir/mediamtx.log" 2>&1 &
child_pids+=("$!")
python3 - "$rtsp_port" <<'PY'
import socket
import sys
import time
for attempt in range(40):
    try:
        with socket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=.2):
            sys.exit(0)
    except OSError:
        time.sleep(.1)
raise SystemExit('Orin MediaMTX 未就绪，请检查 mediamtx.log')
PY

python3 tools/stop_preview.py
if pgrep -x trt >/dev/null; then
    printf '仍有非预览 trt 进程，未自动中断其他控制任务。\n' >&2
    exit 1
fi
python3 - "$web_port" <<'PY'
import socket
import sys
reserved = []
try:
    ports = [1234] + ([int(sys.argv[1])] if int(sys.argv[1]) else [])
    for port in ports:
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        reserved.append(sock)
except OSError as error:
    raise SystemExit(f'本机图像桥接端口仍被其他服务占用，未启动第二个相机进程：{error}')
finally:
    for sock in reserved:
        sock.close()
PY
pose_args=()
if [[ -n "$groundstation_ip" ]]; then pose_args=(--pose-target "$groundstation_ip" --pose-port "$pose_port"); fi
mode_args=()
if [[ "$keypoints_only" == true ]]; then mode_args=(--2d-only); fi
nice -n 10 python3 -u tools/server.py --host 127.0.0.1 --no-display --web-host 0.0.0.0 \
    --web-port "$web_port" \
    --rtsp-url "$rtsp_url" "${pose_args[@]}" "${mode_args[@]}" --video-fps "$video_fps" --render-fps "$video_fps" --bitrate "$bitrate" \
    --video-width "$video_width" --web-width "$web_width" \
    >"$log_dir/server.log" 2>&1 &
child_pids+=("$!")
./trt --preview --coordinate-frames --camera-eye "$camera_eye" \
    --jpeg-width "$video_width" "${mode_args[@]}" >"$log_dir/trt.log" 2>&1 &
child_pids+=("$!")
for ((attempt = 0; attempt < startup_timeout * 4; attempt++)); do
    for child in "${child_pids[@]}"; do
        if ! kill -0 "$child" 2>/dev/null; then
            printf '推流进程退出，请检查 %s 中的日志。\n' "$log_dir" >&2
            tail -n 15 "$log_dir/trt.log" "$log_dir/server.log" >&2
            exit 1
        fi
    done
    if python3 - "$rtsp_port" <<'PY'
import socket
import sys
try:
    port = int(sys.argv[1])
    with socket.create_connection(('127.0.0.1', port), timeout=.5) as connection:
        connection.settimeout(.5)
        connection.sendall(f'DESCRIBE rtsp://127.0.0.1:{port}/live RTSP/1.0\r\nCSeq: 1\r\nAccept: application/sdp\r\n\r\n'.encode())
        header = b''
        while b'\r\n' not in header and len(header) < 4096:
            block = connection.recv(1024)
            if not block:
                sys.exit(1)
            header += block
    sys.exit(0 if header.startswith(b'RTSP/1.0 200 ') else 1)
except (OSError, ValueError, KeyError):
    sys.exit(1)
PY
    then
        printf 'ready\n' > "$startup_file"
        break
    fi
    sleep 0.25
done
if [[ "$(cat "$startup_file")" != ready ]]; then
    printf 'Orin 视频流未就绪，请检查 trt.log / server.log / mediamtx.log。\n' >&2
    exit 1
fi
wait -n "${child_pids[@]}"
