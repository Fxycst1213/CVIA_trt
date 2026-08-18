#include "NokovSDKClient.h"
#include "NokovSDKTypes.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct TrackerSelector {
    enum class Kind { Id, Name };

    Kind kind;
    int id;
    std::string name;
    std::string original;
};

std::atomic<bool> g_running(true);
std::mutex g_output_mutex;
std::map<int, std::string> g_rigid_body_names;
std::vector<TrackerSelector> g_selectors;

long long system_time_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

long long monotonic_time_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string percent_encode(const std::string& value)
{
    std::ostringstream encoded;
    encoded << std::uppercase << std::hex;
    for (unsigned char character : value) {
        if (std::isalnum(character) || character == '-' || character == '_'
            || character == '.' || character == '~') {
            encoded << character;
        } else {
            encoded << '%' << std::setw(2) << std::setfill('0')
                    << static_cast<int>(character);
        }
    }
    return encoded.str();
}

std::string lowercase(std::string value)
{
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char character) { return std::tolower(character); });
    return value;
}

bool parse_int(const std::string& text, int& result)
{
    if (text.empty()) {
        return false;
    }
    std::size_t parsed = 0;
    try {
        result = std::stoi(text, &parsed);
    } catch (...) {
        return false;
    }
    return parsed == text.size();
}

bool parse_selector(const std::string& text, TrackerSelector& selector)
{
    selector.original = text;
    selector.id = 0;
    if (text.compare(0, 3, "id:") == 0) {
        selector.kind = TrackerSelector::Kind::Id;
        return parse_int(text.substr(3), selector.id);
    }
    if (text.compare(0, 5, "name:") == 0) {
        selector.kind = TrackerSelector::Kind::Name;
        selector.name = text.substr(5);
        return !selector.name.empty();
    }

    // A bare integer is an ID; all other values are rigid-body names.
    if (parse_int(text, selector.id)) {
        selector.kind = TrackerSelector::Kind::Id;
    } else {
        selector.kind = TrackerSelector::Kind::Name;
        selector.name = text;
    }
    return !text.empty();
}

bool matches(
    const TrackerSelector& selector, const sRigidBodyData& body,
    const std::string& body_name)
{
    if (selector.kind == TrackerSelector::Kind::Id) {
        return selector.id == body.ID;
    }
    return lowercase(selector.name) == lowercase(body_name);
}

void emit_line(const std::string& line)
{
    std::lock_guard<std::mutex> lock(g_output_mutex);
    std::cout << line << '\n' << std::flush;
}

void data_handler(sFrameOfMocapData* data, void*)
{
    if (!data || !g_running.load()) {
        return;
    }

    const long long received_unix_ns = system_time_ns();
    const long long received_monotonic_ns = monotonic_time_ns();
    std::ostringstream frame_output;
    frame_output << "CLOCK\t" << data->iFrame
                 << '\t' << data->iTimeStamp
                 << '\t' << received_unix_ns
                 << '\t' << received_monotonic_ns
                 << '\n';
    for (int index = 0; index < data->nRigidBodies; ++index) {
        const sRigidBodyData& body = data->RigidBodies[index];
        const auto found = g_rigid_body_names.find(body.ID);
        const std::string body_name =
            found == g_rigid_body_names.end() ? std::string() : found->second;

        for (const TrackerSelector& selector : g_selectors) {
            if (!matches(selector, body, body_name)) {
                continue;
            }
            frame_output << std::setprecision(9)
                         << "POSE\t" << percent_encode(selector.original)
                         << '\t' << body.ID
                         << '\t' << percent_encode(body_name)
                         << '\t' << data->iFrame
                         << '\t' << data->iTimeStamp
                         << '\t' << received_unix_ns
                         << '\t' << received_monotonic_ns
                         << '\t' << body.x
                         << '\t' << body.y
                         << '\t' << body.z
                         << '\t' << body.qx
                         << '\t' << body.qy
                         << '\t' << body.qz
                         << '\t' << body.qw
                         << '\t' << body.MeanError
                         << '\t' << body.params
                         << '\n';
        }
    }
    std::lock_guard<std::mutex> lock(g_output_mutex);
    std::cout << frame_output.str() << std::flush;
}

void stop_handler(int)
{
    g_running.store(false);
}

void print_usage(const char* executable)
{
    std::cerr
        << "Usage: " << executable
        << " --server IP --tracker id:1 [--tracker \"name:tracker 1\"]\n";
}

}  // namespace

int main(int argc, char* argv[])
{
    std::string server;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--server" && index + 1 < argc) {
            server = argv[++index];
        } else if (argument == "--tracker" && index + 1 < argc) {
            TrackerSelector selector;
            if (!parse_selector(argv[++index], selector)) {
                std::cerr << "Invalid tracker selector: " << argv[index] << '\n';
                return 2;
            }
            g_selectors.push_back(selector);
        } else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << argument << '\n';
            print_usage(argv[0]);
            return 2;
        }
    }

    if (server.empty() || g_selectors.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    std::signal(SIGINT, stop_handler);
    std::signal(SIGTERM, stop_handler);

    NokovSDKClient client;
    unsigned char version[4] = {0, 0, 0, 0};
    client.NokovSDKVersion(version);

    std::vector<char> server_buffer(server.begin(), server.end());
    server_buffer.push_back('\0');
    const int result = client.Initialize(server_buffer.data());
    if (result != ErrorCode_OK) {
        std::cerr << "XING SDK initialization failed with error code "
                  << result << " (server " << server << ")\n";
        return 3;
    }

    sServerDescription server_description;
    std::memset(&server_description, 0, sizeof(server_description));
    if (client.GetServerDescription(&server_description) != ErrorCode_OK
        || !server_description.HostPresent) {
        std::cerr << "XING server is not present at " << server << '\n';
        client.Uninitialize();
        return 4;
    }

    sDataDescriptions* descriptions = nullptr;
    if (client.GetDataDescriptions(&descriptions) == ErrorCode_OK
        && descriptions) {
        for (int index = 0; index < descriptions->nDataDescriptions; ++index) {
            const sDataDescription& description =
                descriptions->arrDataDescriptions[index];
            if (description.type == Descriptor_RigidBody
                && description.Data.RigidBodyDescription) {
                const sRigidBodyDescription& rigid_body =
                    *description.Data.RigidBodyDescription;
                g_rigid_body_names[rigid_body.ID] = rigid_body.szName;
                emit_line(
                    "DESC\t" + std::to_string(rigid_body.ID) + "\t"
                    + percent_encode(rigid_body.szName));
            }
        }
        client.FreeDataDescriptions(descriptions);
    }

    if (client.SetDataCallback(data_handler, nullptr) != ErrorCode_OK) {
        std::cerr << "XING SDK rejected the motion-capture data callback\n";
        client.Uninitialize();
        return 5;
    }

    std::ostringstream ready;
    ready << "READY\t" << static_cast<int>(version[0]) << '.'
          << static_cast<int>(version[1]) << '.'
          << static_cast<int>(version[2]) << '.'
          << static_cast<int>(version[3]);
    emit_line(ready.str());

    while (g_running.load() && std::cin.good()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    g_running.store(false);
    client.Uninitialize();
    return 0;
}
