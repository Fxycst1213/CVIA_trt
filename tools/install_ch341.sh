#!/usr/bin/env bash
# Linux stable CH341 driver, built against this Orin's installed kernel headers.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
script_path="$project_dir/tools/install_ch341.sh"
kernel_release="$(uname -r)"
source_version=5.15.148
source_sha256=f66d070eab6235b8a5c7a06a283d2feecb8fa3d81bc1323c847c2d2cbf7bd410
source_url="https://raw.githubusercontent.com/gregkh/linux/v$source_version/drivers/usb/serial/ch341.c"
build_dir="$project_dir/build/drivers/ch341/$kernel_release"
module_path="$build_dir/ch341.ko"
mode="${1:---install}"

if [[ "$mode" == --help ]]; then
    printf '用法：./tools/install_ch341.sh [--prepare]\n'
    printf '下载并编译 Linux 5.15.148 CH341 驱动，再通过 sudo 安装和加载。\n'
    printf '%s\n' '--prepare 仅下载和编译，不修改系统；可在现场断网前准备。'
    exit 0
fi
if (($# > 1)) || [[ "$mode" != --prepare && "$mode" != --install && "$mode" != --install-only ]]; then
    printf '参数错误，请运行 ./tools/install_ch341.sh --help。\n' >&2
    exit 1
fi
if [[ "$kernel_release" != "$source_version-tegra" ]]; then
    printf '此脚本针对 5.15.148-tegra；当前为 %s，请重新匹配驱动源码和头文件。\n' "$kernel_release" >&2
    exit 1
fi

verify_module() {
    local module_vermagic dependency_vermagic
    module_vermagic="$(modinfo -F vermagic "$module_path")"
    dependency_vermagic="$(modinfo -F vermagic usbserial)"
    if [[ "${module_vermagic%% *}" != "$kernel_release" || "$module_vermagic" != "$dependency_vermagic" ]]; then
        printf '驱动与当前内核/usbserial 的版本标记不一致，停止安装。\n' >&2
        exit 1
    fi
    [[ "$(modinfo -F name "$module_path")" == ch341 ]]
}

if [[ "$mode" != --install-only ]]; then
    kernel_build="/lib/modules/$kernel_release/build"
    if [[ ! -f "$kernel_build/Makefile" || ! -s "$kernel_build/Module.symvers" ]]; then
        printf '缺少当前内核的头文件或 Module.symvers：%s\n' "$kernel_build" >&2
        exit 1
    fi
    mkdir -p "$build_dir"
    source_path="$build_dir/ch341.c"
    if [[ ! -f "$source_path" ]] || [[ "$(sha256sum "$source_path" | cut -d ' ' -f 1)" != "$source_sha256" ]]; then
        curl --fail --silent --show-error --location --retry 2 --connect-timeout 10 --max-time 60 \
            "$source_url" --output "$source_path.download"
        if [[ "$(sha256sum "$source_path.download" | cut -d ' ' -f 1)" != "$source_sha256" ]]; then
            printf '驱动源码 SHA256 不匹配，停止编译。\n' >&2
            exit 1
        fi
        mv -- "$source_path.download" "$source_path"
    fi
    printf 'obj-m := ch341.o\n' > "$build_dir/Kbuild"
    make -C "$kernel_build" M="$build_dir" -j2 modules
fi
verify_module
if [[ "$mode" == --prepare ]]; then
    printf '驱动已准备：%s\n' "$module_path"
    printf '版本已匹配：%s\n' "$kernel_release"
    exit 0
fi

if ((EUID != 0)); then
    printf '需要 sudo 将已编译的驱动安装到系统，请在终端输入密码。\n'
    exec sudo bash "$script_path" --install-only
fi

install -D -m 644 "$module_path" "/lib/modules/$kernel_release/updates/ch341.ko"
depmod -a "$kernel_release"
modprobe ch341

# Release only CH340 interfaces claimed by userspace (e.g. brltty); do not
# terminate services or detach other USB devices such as the ZED camera.
for device in /sys/bus/usb/devices/*; do
    [[ -f "$device/idVendor" && -f "$device/idProduct" ]] || continue
    [[ "$(<"$device/idVendor")" == 1a86 && "$(<"$device/idProduct")" == 7523 ]] || continue
    device_name="${device##*/}"
    for interface in /sys/bus/usb/devices/"$device_name":*; do
        [[ -d "$interface" ]] || continue
        interface_name="${interface##*/}"
        if [[ -L "$interface/driver" && "$(basename -- "$(readlink -f "$interface/driver")")" == usbfs ]]; then
            printf '释放 CH340 用户态占用：%s\n' "$interface_name"
            printf '%s\n' "$interface_name" > "$interface/driver/unbind"
        fi
        if [[ ! -L "$interface/driver" ]]; then
            printf '%s\n' "$interface_name" > /sys/bus/usb/drivers_probe
        fi
    done
done
udevadm settle --timeout=5
python3 - <<'PY'
from pathlib import Path
import time

ports = []
for attempt in range(25):
    ports = []
    for tty in Path('/sys/class/tty').glob('ttyUSB*'):
        for parent in tty.resolve().parents:
            try:
                matched = ((parent / 'idVendor').read_text().strip() == '1a86'
                           and (parent / 'idProduct').read_text().strip() == '7523')
            except OSError:
                continue
            if matched and Path('/dev', tty.name).exists():
                ports.append('/dev/' + tty.name)
                break
    if ports:
        break
    time.sleep(.2)
if not ports:
    raise SystemExit('ch341 已加载，但 CH340 串口尚未出现；请重新插拔转换器后再运行本脚本。')
for port in ports:
    print('CH340 串口已就绪：' + port)
if '/dev/ttyUSB0' not in ports:
    raise SystemExit('实际端口不是 C++ 默认的 /dev/ttyUSB0，请按以上设备名调整后再启动。')
print('请重启 start_groundstation.sh，使 C++ 重新打开串口。')
PY
