#!/usr/bin/env python3
"""Measure the timestamp difference between a NOKOV/XING frame and AGX Orin.

This program starts the existing ``MocapBridge`` only.  It does not open a
camera and does not start the detector.  For every selected rigid-body frame,
the bridge gives us both:

* ``mocap_timestamp_ms``: SDK ``sFrameOfMocapData::iTimeStamp``;
* ``receive_unix_ns``: Orin ``CLOCK_REALTIME`` sampled at SDK callback entry.

If both values use the Unix/PTP epoch, the reported apparent difference is::

    orin_minus_mocap_ms = receive_unix_ns / 1e6 - mocap_timestamp_ms

It includes motion-capture processing, network transport and SDK callback
latency, so it is not a pure PTP clock-servo error measurement.  If the SDK
timestamp is relative to device start rather than Unix time, the program does
not claim an absolute offset; it reports relative drift and delivery jitter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence, TextIO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR.parent / "config.json"
DEFAULT_BRIDGE = SCRIPT_DIR / "bin" / "MocapBridge"
UNIX_2000_MS = 946_684_800_000
UNIX_2100_MS = 4_102_444_800_000
ONE_DAY_MS = 86_400_000.0
TAI_MINUS_UTC_MS = 37_000.0


@dataclass(frozen=True)
class ClockSample:
    sample_index: int
    mocap_frame: int
    mocap_timestamp_ms: int
    orin_receive_unix_ns: int
    orin_receive_monotonic_ns: int
    python_read_unix_ns: int
    tracker_id: int
    tracker_name: str

    @property
    def orin_receive_unix_ms(self) -> float:
        return self.orin_receive_unix_ns / 1_000_000.0

    @property
    def orin_minus_mocap_ms(self) -> float:
        return self.orin_receive_unix_ms - self.mocap_timestamp_ms

    @property
    def python_pipe_delay_ms(self) -> float:
        return (self.python_read_unix_ns - self.orin_receive_unix_ns) / 1_000_000.0


@dataclass(frozen=True)
class MetricStats:
    count: int
    minimum: float
    mean: float
    stdev: float
    p50: float
    p95: float
    p99: float
    maximum: float
    peak_to_peak: float


@dataclass(frozen=True)
class Analysis:
    time_domain: str
    time_domain_message: str
    raw_offset: MetricStats
    relative_residual: MetricStats
    detrended_jitter: MetricStats
    sdk_period: MetricStats | None
    orin_period: MetricStats | None
    pipe_delay: MetricStats
    drift_ms_per_s: float
    drift_ppm: float
    affine_scale: float
    affine_rate_ppm: float
    affine_residual: MetricStats
    sdk_rate_hz: float | None
    orin_rate_hz: float | None


def parse_pose_line(line: str, sample_index: int, read_unix_ns: int) -> ClockSample:
    """Parse only the timestamp fields needed from a 17-column POSE line."""
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 17 or fields[0] != "POSE":
        raise ValueError(f"期望 17 列 POSE 数据，实际 {len(fields)} 列")
    return ClockSample(
        sample_index=sample_index,
        mocap_frame=int(fields[4]),
        mocap_timestamp_ms=int(fields[5]),
        orin_receive_unix_ns=int(fields[6]),
        orin_receive_monotonic_ns=int(fields[7]),
        python_read_unix_ns=read_unix_ns,
        tracker_id=int(fields[2]),
        tracker_name=urllib.parse.unquote(fields[3]),
    )


def parse_clock_line(line: str, sample_index: int, read_unix_ns: int) -> ClockSample:
    """Parse a FrameGroup CLOCK record which does not require a visible body."""
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 5 or fields[0] != "CLOCK":
        raise ValueError(f"期望 5 列 CLOCK 数据，实际 {len(fields)} 列")
    return ClockSample(
        sample_index=sample_index,
        mocap_frame=int(fields[1]),
        mocap_timestamp_ms=int(fields[2]),
        orin_receive_unix_ns=int(fields[3]),
        orin_receive_monotonic_ns=int(fields[4]),
        python_read_unix_ns=read_unix_ns,
        tracker_id=0,
        tracker_name="FrameGroup CLOCK",
    )


def percentile(values: Sequence[float], percent: float) -> float:
    """Return a linearly interpolated percentile without third-party packages."""
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metric_stats(values: Iterable[float]) -> MetricStats:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("cannot calculate statistics for no samples")
    minimum = min(data)
    maximum = max(data)
    return MetricStats(
        count=len(data),
        minimum=minimum,
        mean=statistics.fmean(data),
        stdev=statistics.stdev(data) if len(data) > 1 else 0.0,
        p50=percentile(data, 50),
        p95=percentile(data, 95),
        p99=percentile(data, 99),
        maximum=maximum,
        peak_to_peak=maximum - minimum,
    )


def linear_fit(x_values: Sequence[float], y_values: Sequence[float]) -> tuple[float, float]:
    """Return (intercept, slope) for y = intercept + slope*x."""
    if len(x_values) != len(y_values) or not x_values:
        raise ValueError("linear-fit inputs must be non-empty and have equal length")
    if len(x_values) == 1:
        return float(y_values[0]), 0.0
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator <= 0:
        return y_mean, 0.0
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    ) / denominator
    return y_mean - slope * x_mean, slope


def classify_time_domain(samples: Sequence[ClockSample]) -> tuple[str, str]:
    sdk_median = percentile([sample.mocap_timestamp_ms for sample in samples], 50)
    offset_median = percentile([sample.orin_minus_mocap_ms for sample in samples], 50)
    epoch_looking = UNIX_2000_MS <= sdk_median <= UNIX_2100_MS

    if epoch_looking and abs(offset_median + TAI_MINUS_UTC_MS) <= 5_000:
        return (
            "possible_tai",
            "SDK 时间戳像绝对时间，但比 Orin UTC 约快 37 秒；可能是 TAI，而 Orin CLOCK_REALTIME 是 UTC。",
        )
    if epoch_looking and abs(offset_median) <= ONE_DAY_MS:
        return (
            "unix_like",
            "SDK 时间戳具有 Unix 历元外观，可计算端到端表观时差；这不证明它与 Orin 同钟，仍需根据线性漂移判断是否为独立时钟。",
        )
    if epoch_looking:
        return (
            "epoch_mismatch",
            "SDK 时间戳像绝对时间，但与 Orin 相差超过 24 小时；两端历元或系统时间可能未对齐。",
        )
    return (
        "relative",
        "SDK iTimeStamp 不像 Unix 毫秒时间戳，不能由它直接得到绝对钟差；以下相对结果只反映时钟漂移和可变传输延迟。",
    )


def positive_deltas(values: Sequence[float]) -> list[float]:
    return [
        current - previous
        for previous, current in zip(values, values[1:])
        if current > previous
    ]


def analyse(samples: Sequence[ClockSample]) -> Analysis:
    if not samples:
        raise ValueError("至少需要一个采样")

    time_domain, message = classify_time_domain(samples)
    raw_offsets = [sample.orin_minus_mocap_ms for sample in samples]
    first = samples[0]
    relative_residuals = [
        (sample.orin_receive_monotonic_ns - first.orin_receive_monotonic_ns) / 1_000_000.0
        - (sample.mocap_timestamp_ms - first.mocap_timestamp_ms)
        for sample in samples
    ]
    elapsed_seconds = [
        (sample.orin_receive_monotonic_ns - first.orin_receive_monotonic_ns) / 1_000_000_000.0
        for sample in samples
    ]
    fit_values = raw_offsets if time_domain in {"unix_like", "possible_tai"} else relative_residuals
    intercept, slope = linear_fit(elapsed_seconds, fit_values)
    detrended = [
        value - (intercept + slope * elapsed)
        for elapsed, value in zip(elapsed_seconds, fit_values)
    ]

    sdk_elapsed_ms = [
        float(sample.mocap_timestamp_ms - first.mocap_timestamp_ms)
        for sample in samples
    ]
    orin_elapsed_ms = [
        (sample.orin_receive_monotonic_ns - first.orin_receive_monotonic_ns)
        / 1_000_000.0
        for sample in samples
    ]
    affine_intercept, affine_scale = linear_fit(sdk_elapsed_ms, orin_elapsed_ms)
    affine_residuals = [
        orin_elapsed - (affine_intercept + affine_scale * sdk_elapsed)
        for sdk_elapsed, orin_elapsed in zip(sdk_elapsed_ms, orin_elapsed_ms)
    ]

    sdk_periods = positive_deltas(
        [float(sample.mocap_timestamp_ms) for sample in samples]
    )
    orin_periods = positive_deltas(
        [sample.orin_receive_monotonic_ns / 1_000_000.0 for sample in samples]
    )
    sdk_period_stats = metric_stats(sdk_periods) if sdk_periods else None
    orin_period_stats = metric_stats(orin_periods) if orin_periods else None
    sdk_rate = 1000.0 / sdk_period_stats.p50 if sdk_period_stats and sdk_period_stats.p50 > 0 else None
    orin_rate = 1000.0 / orin_period_stats.p50 if orin_period_stats and orin_period_stats.p50 > 0 else None

    return Analysis(
        time_domain=time_domain,
        time_domain_message=message,
        raw_offset=metric_stats(raw_offsets),
        relative_residual=metric_stats(relative_residuals),
        detrended_jitter=metric_stats(detrended),
        sdk_period=sdk_period_stats,
        orin_period=orin_period_stats,
        pipe_delay=metric_stats(sample.python_pipe_delay_ms for sample in samples),
        drift_ms_per_s=slope,
        drift_ppm=slope * 1_000.0,
        affine_scale=affine_scale,
        affine_rate_ppm=(affine_scale - 1.0) * 1_000_000.0,
        affine_residual=metric_stats(affine_residuals),
        sdk_rate_hz=sdk_rate,
        orin_rate_hz=orin_rate,
    )


def load_mocap_defaults(config_path: Path) -> tuple[str, str]:
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取配置 {config_path}: {exc}") from exc
    mocap = config.get("mocap", {})
    return str(mocap.get("server", "")).strip(), str(mocap.get("tracker", "")).strip()


def stream_reader(stream: TextIO, channel: str, output: queue.Queue[tuple[str, str, int]]) -> None:
    try:
        for raw_line in stream:
            output.put((channel, raw_line.rstrip("\r\n"), time.time_ns()))
    finally:
        output.put((f"{channel}_eof", "", time.time_ns()))


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def collect_samples(
    bridge_path: Path,
    server: str,
    tracker: str,
    duration_seconds: float,
    sample_limit: int,
    connect_timeout_seconds: float,
    print_every: int,
) -> tuple[list[ClockSample], list[tuple[int, str]], str | None, list[str]]:
    if not bridge_path.is_file():
        raise RuntimeError(f"未找到动捕桥接程序：{bridge_path}")
    if not os.access(bridge_path, os.X_OK):
        raise RuntimeError(f"动捕桥接程序不可执行：{bridge_path}")

    command = [str(bridge_path), "--server", server, "--tracker", tracker]
    environment = os.environ.copy()
    local_library_dir = bridge_path.parent.parent / "lib" / "aarch64"
    if local_library_dir.is_dir():
        old_library_path = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            f"{local_library_dir}:{old_library_path}"
            if old_library_path
            else str(local_library_dir)
        )

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
    except OSError as exc:
        raise RuntimeError(f"无法启动动捕桥接程序：{exc}") from exc

    assert process.stdout is not None
    assert process.stderr is not None
    output: queue.Queue[tuple[str, str, int]] = queue.Queue()
    stdout_thread = threading.Thread(
        target=stream_reader, args=(process.stdout, "stdout", output), daemon=True
    )
    stderr_thread = threading.Thread(
        target=stream_reader, args=(process.stderr, "stderr", output), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    samples: list[ClockSample] = []
    descriptions: dict[int, str] = {}
    errors: list[str] = []
    sdk_version: str | None = None
    initial_deadline = time.monotonic() + connect_timeout_seconds
    sampling_deadline: float | None = None
    try:
        while True:
            now = time.monotonic()
            deadline = sampling_deadline if sampling_deadline is not None else initial_deadline
            if now >= deadline:
                break
            if sample_limit > 0 and len(samples) >= sample_limit:
                break
            try:
                channel, line, read_unix_ns = output.get(timeout=min(0.25, deadline - now))
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue

            if channel == "stderr":
                if line:
                    errors.append(line)
                    errors = errors[-20:]
                continue
            if channel != "stdout" or not line:
                if process.poll() is not None and channel == "stdout_eof":
                    break
                continue

            fields = line.split("\t")
            if fields[0] == "DESC" and len(fields) >= 3:
                try:
                    descriptions[int(fields[1])] = urllib.parse.unquote(fields[2])
                except ValueError:
                    pass
                continue
            if fields[0] == "READY" and len(fields) >= 2:
                sdk_version = fields[1]
                continue
            if fields[0] not in {"CLOCK", "POSE"}:
                continue

            try:
                if fields[0] == "CLOCK":
                    sample = parse_clock_line(
                        line, len(samples) + 1, read_unix_ns
                    )
                else:
                    sample = parse_pose_line(
                        line, len(samples) + 1, read_unix_ns
                    )
            except (ValueError, OverflowError) as exc:
                errors.append(f"无法解析 {fields[0]}：{exc}")
                errors = errors[-20:]
                continue
            # New bridges emit CLOCK immediately before any POSE records for
            # the same FrameGroup.  Keep one timing sample per frame while old
            # POSE-only bridges remain supported.
            if samples and sample.mocap_frame == samples[-1].mocap_frame:
                continue
            samples.append(sample)
            if sampling_deadline is None:
                sampling_deadline = time.monotonic() + duration_seconds
            if print_every > 0 and (len(samples) == 1 or len(samples) % print_every == 0):
                print(
                    f"采样 {len(samples):6d} | frame={sample.mocap_frame:8d} | "
                    f"Orin-SDK={sample.orin_minus_mocap_ms:+.3f} ms",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，使用已采集的数据生成统计。", file=sys.stderr)
    finally:
        stop_process(process)

    return samples, sorted(descriptions.items()), sdk_version, errors


CSV_COLUMNS = (
    "sample_index",
    "mocap_frame",
    "mocap_timestamp_ms",
    "orin_receive_unix_ns",
    "orin_receive_monotonic_ns",
    "orin_receive_unix_ms",
    "orin_minus_mocap_ms",
    "relative_residual_ms",
    "python_read_unix_ns",
    "python_pipe_delay_ms",
    "tracker_id",
    "tracker_name",
)


def write_csv(path: Path, samples: Sequence[ClockSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = samples[0]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for sample in samples:
            relative_residual = (
                (sample.orin_receive_monotonic_ns - first.orin_receive_monotonic_ns)
                / 1_000_000.0
                - (sample.mocap_timestamp_ms - first.mocap_timestamp_ms)
            )
            writer.writerow({
                "sample_index": sample.sample_index,
                "mocap_frame": sample.mocap_frame,
                "mocap_timestamp_ms": sample.mocap_timestamp_ms,
                "orin_receive_unix_ns": sample.orin_receive_unix_ns,
                "orin_receive_monotonic_ns": sample.orin_receive_monotonic_ns,
                "orin_receive_unix_ms": f"{sample.orin_receive_unix_ms:.6f}",
                "orin_minus_mocap_ms": f"{sample.orin_minus_mocap_ms:.6f}",
                "relative_residual_ms": f"{relative_residual:.6f}",
                "python_read_unix_ns": sample.python_read_unix_ns,
                "python_pipe_delay_ms": f"{sample.python_pipe_delay_ms:.6f}",
                "tracker_id": sample.tracker_id,
                "tracker_name": sample.tracker_name,
            })


def format_stats(label: str, stats: MetricStats, unit: str = "ms") -> str:
    return (
        f"{label}: mean={stats.mean:+.3f} {unit}, std={stats.stdev:.3f}, "
        f"P50={stats.p50:+.3f}, P95={stats.p95:+.3f}, P99={stats.p99:+.3f}, "
        f"min={stats.minimum:+.3f}, max={stats.maximum:+.3f}, "
        f"峰峰值={stats.peak_to_peak:.3f}"
    )


def format_utc_from_ms(timestamp_ms: float) -> str:
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return "无法转换"


def print_analysis(samples: Sequence[ClockSample], analysis: Analysis) -> None:
    first = samples[0]
    last = samples[-1]
    elapsed = (
        last.orin_receive_monotonic_ns - first.orin_receive_monotonic_ns
    ) / 1_000_000_000.0
    print("\n========== 动捕 / Orin 时间戳测量结果 ==========")
    print(f"有效采样: {len(samples)} 帧；实际跨度: {elapsed:.3f} s")
    print(f"时间样本源: {first.tracker_id}:{first.tracker_name}")
    print(f"时间域判定: {analysis.time_domain}")
    print(f"说明: {analysis.time_domain_message}")
    print(f"首帧 SDK 时间:  {first.mocap_timestamp_ms} ms ({format_utc_from_ms(first.mocap_timestamp_ms)})")
    print(f"首帧 Orin 时间: {first.orin_receive_unix_ms:.3f} ms ({format_utc_from_ms(first.orin_receive_unix_ms)})")

    if analysis.time_domain in {"unix_like", "possible_tai", "epoch_mismatch"}:
        print(format_stats("Orin回调时间 - SDK时间", analysis.raw_offset))
    else:
        print(
            f"原始数值 Orin-SDK 的 P50={analysis.raw_offset.p50:+.3f} ms，"
            "但因为历元未知，它不是可解释的绝对钟差。"
        )
    print(format_stats("相对首帧的时间差变化", analysis.relative_residual))
    print(format_stats("去线性漂移后的抖动", analysis.detrended_jitter))
    print(
        f"线性漂移: {analysis.drift_ms_per_s:+.6f} ms/s "
        f"({analysis.drift_ppm:+.3f} ppm)"
    )
    print(
        "SDK独立时钟 → Orin 仿射速率修正: "
        f"a={analysis.affine_scale:.9f} "
        f"({analysis.affine_rate_ppm:+.3f} ppm)"
    )
    print(format_stats("仿射速率校正后的剩余回调残差", analysis.affine_residual))

    if analysis.sdk_period is not None:
        print(
            format_stats("SDK 时间戳帧间隔", analysis.sdk_period)
            + (f"；按 P50 约 {analysis.sdk_rate_hz:.3f} Hz" if analysis.sdk_rate_hz else "")
        )
    if analysis.orin_period is not None:
        print(
            format_stats("Orin 回调帧间隔", analysis.orin_period)
            + (f"；按 P50 约 {analysis.orin_rate_hz:.3f} Hz" if analysis.orin_rate_hz else "")
        )
    print(format_stats("桥接回调到 Python 读到数据的附加延迟", analysis.pipe_delay))
    print(
        "\n注意: Orin-SDK 是未校正的独立时钟原始差值，跨次测试继续增长是正常现象；"
        "工程使用仿射模型消除其速率漂移。绝对差值仍包含动捕曝光/解算、网络传输和 SDK "
        "分发时间，剩余固定延迟需要通过视觉/动捕运动曲线互相关估计 offset_ms。"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="仅启动 NOKOV/XING SDK 桥接，测量动捕 iTimeStamp 与 AGX Orin 时间戳差值。"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"配置文件（默认：{DEFAULT_CONFIG}）")
    parser.add_argument("--server", help="动捕服务器 IP；默认读取 config.json 的 mocap.server")
    parser.add_argument("--tracker", help="刚体选择器，如 name:Tracker3 或 id:1；默认读取配置")
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE, help=f"MocapBridge 路径（默认：{DEFAULT_BRIDGE}）")
    parser.add_argument("--duration", type=float, default=10.0, help="收到首帧后的采样秒数（默认：10）")
    parser.add_argument("--samples", type=int, default=0, help="达到该帧数后提前结束；0 表示不限制")
    parser.add_argument("--connect-timeout", type=float, default=10.0, help="等待首帧的秒数（默认：10）")
    parser.add_argument("--print-every", type=int, default=30, help="每 N 帧打印一次即时值；0 表示关闭（默认：30）")
    parser.add_argument("--csv", type=Path, help="可选：保存每帧原始时间戳及差值的 CSV 路径")
    parser.add_argument("--json", type=Path, help="可选：保存汇总统计 JSON 路径")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.duration <= 0:
        parser.error("--duration 必须大于 0")
    if args.samples < 0:
        parser.error("--samples 不能小于 0")
    if args.connect_timeout <= 0:
        parser.error("--connect-timeout 必须大于 0")
    if args.print_every < 0:
        parser.error("--print-every 不能小于 0")

    config_server = ""
    config_tracker = ""
    if args.server is None or args.tracker is None:
        try:
            config_server, config_tracker = load_mocap_defaults(args.config)
        except RuntimeError as exc:
            parser.error(str(exc))
    server = (args.server or config_server).strip()
    tracker = (args.tracker or config_tracker).strip()
    if not server:
        parser.error("未指定 --server，配置中也没有 mocap.server")
    if not tracker:
        parser.error("未指定 --tracker，配置中也没有 mocap.tracker")

    print("仅测量动捕时间戳，不启动相机与检测。")
    print(f"服务器: {server}；刚体: {tracker}；采样: {args.duration:g} s")
    try:
        samples, descriptions, sdk_version, errors = collect_samples(
            bridge_path=args.bridge.resolve(),
            server=server,
            tracker=tracker,
            duration_seconds=args.duration,
            sample_limit=args.samples,
            connect_timeout_seconds=args.connect_timeout,
            print_every=args.print_every,
        )
    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if not samples:
        print("错误: 在等待时间内没有收到目标刚体的 POSE 数据。", file=sys.stderr)
        if sdk_version:
            print(f"SDK 已连接，版本 {sdk_version}。", file=sys.stderr)
        if descriptions:
            discovered = "、".join(f"{tracker_id}:{name}" for tracker_id, name in descriptions)
            print(f"已发现刚体: {discovered}", file=sys.stderr)
        if errors:
            print(f"桥接程序最后一条错误: {errors[-1]}", file=sys.stderr)
        print("请确认动捕软件正在发送数据、服务器 IP 正确，并用已发现的刚体名称设置 --tracker。", file=sys.stderr)
        return 3

    analysis = analyse(samples)
    print_analysis(samples, analysis)

    if args.csv:
        write_csv(args.csv.resolve(), samples)
        print(f"逐帧数据已保存: {args.csv.resolve()}")
    if args.json:
        payload = {
            "server": server,
            "tracker": tracker,
            "sdk_version": sdk_version,
            "sample_count": len(samples),
            "first_sample": asdict(samples[0]),
            "last_sample": asdict(samples[-1]),
            "analysis": asdict(analysis),
        }
        args.json.resolve().parent.mkdir(parents=True, exist_ok=True)
        with args.json.resolve().open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"汇总 JSON 已保存: {args.json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
