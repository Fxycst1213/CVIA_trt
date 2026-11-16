#include "vision_protocol.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace vision_protocol
{
    namespace
    {
        void write_u16(uint8_t *out, uint16_t value)
        {
            out[0] = static_cast<uint8_t>(value >> 8);
            out[1] = static_cast<uint8_t>(value);
        }

        void write_u32(uint8_t *out, uint32_t value)
        {
            for (int i = 0; i < 4; ++i)
                out[i] = static_cast<uint8_t>(value >> (24 - 8 * i));
        }

        uint16_t read_u16(const uint8_t *in)
        {
            return (static_cast<uint16_t>(in[0]) << 8) | in[1];
        }

        int16_t read_i16(const uint8_t *in)
        {
            const auto bits = read_u16(in);
            return static_cast<int16_t>(bits <= 0x7FFF ? int32_t(bits) : int32_t(bits) - 0x10000);
        }

        int32_t read_i32(const uint8_t *in)
        {
            uint32_t bits = 0;
            for (int i = 0; i < 4; ++i)
                bits = (bits << 8) | in[i];
            return static_cast<int32_t>(bits <= 0x7FFFFFFFU ? int64_t(bits) : int64_t(bits) - 0x100000000LL);
        }

        bool valid_frame(const uint8_t *frame, std::size_t size, uint8_t direction,
                         std::size_t payload_size)
        {
            return frame && size == payload_size + 4 && frame[0] == 0xEB &&
                   frame[1] == direction && frame[2] == payload_size &&
                   checksum(frame + 2, payload_size + 1) == frame[size - 1];
        }

        bool quantize_value(double value, double scale, int16_t &out)
        {
            const double rounded = std::round(value * scale);
            if (!std::isfinite(rounded) || rounded < std::numeric_limits<int16_t>::min() ||
                rounded > std::numeric_limits<int16_t>::max())
                return false;
            out = static_cast<int16_t>(rounded);
            return true;
        }
    }

    uint8_t checksum(const uint8_t *data, std::size_t size)
    {
        uint32_t sum = 0;
        for (std::size_t i = 0; i < size; ++i)
            sum += data[i];
        return static_cast<uint8_t>(sum & 0xFF);
    }

    TxFrame encode_tx(const TxData &data)
    {
        TxFrame frame{};
        frame[0] = 0xEB;
        frame[1] = 0x90;
        frame[2] = TX_PAYLOAD_SIZE;
        for (std::size_t i = 0; i < 3; ++i)
        {
            write_u16(frame.data() + 3 + 2 * i, static_cast<uint16_t>(data.absolute_mm[i]));
            write_u16(frame.data() + 9 + 2 * i, static_cast<uint16_t>(data.relative_mm[i]));
            write_u16(frame.data() + 15 + 2 * i, static_cast<uint16_t>(data.rotation_cdeg[i]));
        }
        frame.back() = checksum(frame.data() + 2, TX_PAYLOAD_SIZE + 1);
        return frame;
    }

    RxFrame encode_rx(const RxData &data)
    {
        RxFrame frame{};
        frame[0] = 0xEB;
        frame[1] = 0x91;
        frame[2] = RX_PAYLOAD_SIZE;
        write_u32(frame.data() + 3, static_cast<uint32_t>(data.pitch1_mdeg));
        write_u32(frame.data() + 7, static_cast<uint32_t>(data.pitch2_mdeg));
        write_u32(frame.data() + 11, static_cast<uint32_t>(data.roll_mdeg));
        write_u16(frame.data() + 15, data.extension_mm);
        frame.back() = checksum(frame.data() + 2, RX_PAYLOAD_SIZE + 1);
        return frame;
    }

    bool decode_tx(const uint8_t *frame, std::size_t size, TxData &data)
    {
        if (!valid_frame(frame, size, 0x90, TX_PAYLOAD_SIZE))
            return false;
        TxData decoded;
        for (std::size_t i = 0; i < 3; ++i)
        {
            decoded.absolute_mm[i] = read_i16(frame + 3 + 2 * i);
            decoded.relative_mm[i] = read_i16(frame + 9 + 2 * i);
            decoded.rotation_cdeg[i] = read_i16(frame + 15 + 2 * i);
        }
        data = decoded;
        return true;
    }

    bool decode_rx(const uint8_t *frame, std::size_t size, RxData &data)
    {
        if (!valid_frame(frame, size, 0x91, RX_PAYLOAD_SIZE))
            return false;
        RxData decoded;
        decoded.pitch1_mdeg = read_i32(frame + 3);
        decoded.pitch2_mdeg = read_i32(frame + 7);
        decoded.roll_mdeg = read_i32(frame + 11);
        decoded.extension_mm = read_u16(frame + 15);
        data = decoded;
        return true;
    }

    bool quantize(const TxMeasurements &measurements, TxData &data)
    {
        TxData converted;
        for (std::size_t i = 0; i < 3; ++i)
        {
            if (!quantize_value(measurements.absolute_mm[i], 1, converted.absolute_mm[i]) ||
                !quantize_value(measurements.relative_mm[i], 1, converted.relative_mm[i]) ||
                !quantize_value(measurements.rotation_deg[i], 100, converted.rotation_cdeg[i]))
                return false;
        }
        data = converted;
        return true;
    }

    RxParser::RxParser()
    {
        _buffer.reserve(RX_FRAME_SIZE);
    }

    std::vector<RxData> RxParser::feed(const uint8_t *bytes, std::size_t size)
    {
        std::vector<RxData> messages;
        for (std::size_t i = 0; i < size; ++i)
        {
            _buffer.push_back(bytes[i]);
            while (!_buffer.empty())
            {
                if (_buffer[0] != 0xEB || (_buffer.size() >= 2 && _buffer[1] != 0x91) ||
                    (_buffer.size() >= 3 && _buffer[2] != RX_PAYLOAD_SIZE))
                {
                    _buffer.erase(_buffer.begin());
                    continue;
                }
                if (_buffer.size() < RX_FRAME_SIZE)
                    break;
                RxData message;
                if (decode_rx(_buffer.data(), _buffer.size(), message))
                {
                    messages.push_back(message);
                    _buffer.clear();
                }
                else
                {
                    // Keep scanning: a valid header may start inside damaged data.
                    _buffer.erase(_buffer.begin());
                }
            }
        }
        return messages;
    }

    void RxParser::reset()
    {
        _buffer.clear();
    }
}
