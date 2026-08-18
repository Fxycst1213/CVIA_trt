#ifndef CVIA_CAMERA_TIMESTAMP_SYNC_HPP
#define CVIA_CAMERA_TIMESTAMP_SYNC_HPP

#include <cstdint>

namespace camera_time
{
enum class Clock
{
    Realtime,
    Monotonic,
    Unknown,
};

// 将 V4L2 CLOCK_MONOTONIC 缓冲时间戳映射到 PTP 驯服的 CLOCK_REALTIME。
// REALTIME 在 MONOTONIC 前后夹读，使用中点减少线程调度引入的映射误差。
constexpr uint64_t to_realtime_ns(
    uint64_t driver_timestamp_ns,
    Clock clock,
    uint64_t realtime_before_ns,
    uint64_t monotonic_ns,
    uint64_t realtime_after_ns)
{
    if (driver_timestamp_ns == 0) return 0;
    if (clock == Clock::Realtime) return driver_timestamp_ns;
    if (clock != Clock::Monotonic || monotonic_ns == 0) return 0;
    const uint64_t realtime_midpoint = realtime_before_ns <= realtime_after_ns
        ? realtime_before_ns + (realtime_after_ns - realtime_before_ns) / 2
        : realtime_after_ns + (realtime_before_ns - realtime_after_ns) / 2;
    if (driver_timestamp_ns > monotonic_ns)
        return realtime_midpoint + (driver_timestamp_ns - monotonic_ns);
    const uint64_t age_ns = monotonic_ns - driver_timestamp_ns;
    return age_ns <= realtime_midpoint ? realtime_midpoint - age_ns : 0;
}

static_assert(
    to_realtime_ns(900, Clock::Monotonic, 1990, 1000, 2010) == 1900,
    "monotonic frame timestamp must map into the realtime/PTP domain");
static_assert(
    to_realtime_ns(1234, Clock::Realtime, 1990, 1000, 2010) == 1234,
    "realtime driver timestamp must remain unchanged");
static_assert(
    to_realtime_ns(1234, Clock::Unknown, 1990, 1000, 2010) == 0,
    "unknown clock domain must not be guessed by the conversion primitive");
}

#endif
