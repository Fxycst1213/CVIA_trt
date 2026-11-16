"""Stop only this checkout's camera preview before switching to ground station mode."""
import os
from pathlib import Path
import signal
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]


def process_state(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]  # State and start time protect against PID reuse.
    except (OSError, IndexError):
        return None


def matches_file(argument, cwd, target):
    path = Path(argument)
    return (path if path.is_absolute() else cwd / path).resolve() == target


def preview_processes():
    matches = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            cwd = Path(os.readlink(directory / "cwd")).resolve()
            if cwd != PROJECT:
                continue
            args = [os.fsdecode(value) for value in (directory / "cmdline").read_bytes().split(b"\0") if value]
            if not args:
                continue
            executable = Path(args[0]).name
            is_launcher = executable in ("bash", "sh") and any(
                matches_file(arg, cwd, PROJECT / "start_preview.sh") for arg in args[1:])
            is_camera = (matches_file(args[0], cwd, PROJECT / "trt")
                         and "--preview" in args and "--no-serial" in args)
            has_rtsp = any(arg == "--rtsp-url" or arg.startswith("--rtsp-url=") for arg in args)
            is_bridge = (executable.startswith("python") and not has_rtsp and any(
                matches_file(arg, cwd, PROJECT / "tools/server.py") for arg in args[1:]))
            if is_launcher or is_camera or is_bridge:
                state = process_state(int(directory.name))
                if state and state[0] != "Z":
                    matches.append((int(directory.name), state[1], is_launcher))
        except (OSError, ValueError):
            continue
    return sorted(matches, key=lambda item: not item[2])


def still_running(process):
    state = process_state(process[0])
    return state is not None and state[0] != "Z" and state[1] == process[1]


def main():
    processes = preview_processes()
    for process in processes:
        if still_running(process):
            try:
                os.kill(process[0], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    while any(still_running(process) for process in processes):
        if time.monotonic() >= deadline:
            remaining = ", ".join(str(process[0]) for process in processes if still_running(process))
            print(f"预览进程尚未退出：{remaining}；未强制结束，请检查进程状态。", file=sys.stderr)
            return 1
        time.sleep(.1)
    if processes:
        print("已停止本项目的原预览，正在切换到地面站推流。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
