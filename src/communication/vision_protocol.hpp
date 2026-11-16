#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// 2026-4-7-数据传输协议定义-视觉（西电）.docx
// RS-422 full duplex. Multibyte values: big endian.
// Checksum: sum of the length byte and payload, modulo 256.
// No type byte, CRC8, or CR/LF trailer from the old RS485 protocol.
namespace vision_protocol
{
    constexpr std::size_t TX_PAYLOAD_SIZE = 18;
    constexpr std::size_t RX_PAYLOAD_SIZE = 14;
    constexpr std::size_t TX_FRAME_SIZE = 22;
    constexpr std::size_t RX_FRAME_SIZE = 18;

    struct TxData
    {
        std::array<int16_t, 3> absolute_mm{}; // Tanker coordinate system: X, Y, Z.
        std::array<int16_t, 3> relative_mm{}; // Refueling boom coordinate system: X, Y, Z.
        std::array<int16_t, 3> rotation_cdeg{}; // Relative X/Y/Z rotation, units of 0.01 degree.
    };

    struct TxMeasurements
    {
        std::array<double, 3> absolute_mm{};
        std::array<double, 3> relative_mm{};
        std::array<double, 3> rotation_deg{};
    };

    struct RxData
    {
        int32_t pitch1_mdeg = 0; // Units of 0.001 degree.
        int32_t pitch2_mdeg = 0;
        int32_t roll_mdeg = 0;
        uint16_t extension_mm = 0;
    };

    using TxFrame = std::array<uint8_t, TX_FRAME_SIZE>;
    using RxFrame = std::array<uint8_t, RX_FRAME_SIZE>;

    uint8_t checksum(const uint8_t *data, std::size_t size);
    TxFrame encode_tx(const TxData &data);
    RxFrame encode_rx(const RxData &data); // Controller simulator / protocol tests.
    bool decode_tx(const uint8_t *frame, std::size_t size, TxData &data);
    bool decode_rx(const uint8_t *frame, std::size_t size, RxData &data);

    // Round mm and centidegrees to nearest integer (ties away from zero).
    // Reject NaN, infinity, and int16 overflow; never silently wrap or clamp.
    // The output is unchanged on failure.
    bool quantize(const TxMeasurements &measurements, TxData &data);

    class RxParser
    {
    public:
        RxParser();
        std::vector<RxData> feed(const uint8_t *bytes, std::size_t size);
        void reset();

    private:
        std::vector<uint8_t> _buffer;
    };
}
