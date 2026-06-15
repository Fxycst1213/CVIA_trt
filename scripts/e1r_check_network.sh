#!/usr/bin/env bash
set -e

LIDAR_IFACE="${LIDAR_IFACE:-eno1}"
LIDAR_IP="${LIDAR_IP:-192.168.1.200}"
MSOP_PORT="${MSOP_PORT:-6699}"
DIFOP_PORT="${DIFOP_PORT:-7788}"
CAPTURE_COUNT="${CAPTURE_COUNT:-10}"

echo "[E1R] Network interface: ${LIDAR_IFACE}"
ip addr show "${LIDAR_IFACE}"

echo
echo "[E1R] Route to lidar: ${LIDAR_IP}"
ip route get "${LIDAR_IP}"

echo
echo "[E1R] Capturing ${CAPTURE_COUNT} UDP packets on ${LIDAR_IFACE} ports ${MSOP_PORT}/${DIFOP_PORT}"
echo "[E1R] sudo may ask for your password."
sudo tcpdump -i "${LIDAR_IFACE}" -nn "udp port ${MSOP_PORT} or udp port ${DIFOP_PORT}" -c "${CAPTURE_COUNT}"
