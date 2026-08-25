#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{
std::string executable_directory(const char *argv0)
{
    char buffer[PATH_MAX + 1]{};
    const ssize_t length = readlink("/proc/self/exe", buffer, PATH_MAX);
    std::string path;
    if (length > 0)
        path.assign(buffer, static_cast<size_t>(length));
    else
    {
        char *resolved = realpath(argv0, buffer);
        path = resolved ? resolved : argv0;
    }
    const std::string::size_type separator = path.find_last_of('/');
    return separator == std::string::npos ? "." : path.substr(0, separator);
}

std::string environment_or(const char *name, const char *fallback)
{
    const char *value = std::getenv(name);
    return value && *value ? value : fallback;
}

bool valid_port(const std::string &value)
{
    if (value.empty()) return false;
    char *end = nullptr;
    errno = 0;
    const long port = std::strtol(value.c_str(), &end, 10);
    return errno == 0 && end && *end == '\0' && port >= 1 && port <= 65535;
}

bool require_file(const std::string &path, bool executable = false)
{
    const int mode = executable ? X_OK : R_OK;
    if (access(path.c_str(), mode) == 0) return true;
    std::cerr << "[CVIA] 缺少运行文件或权限不足：" << path << '\n';
    return false;
}
}

int main(int argc, char **argv)
{
    std::string host = environment_or("WEB_HOST", "0.0.0.0");
    std::string port = environment_or("WEB_PORT", "8765");
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h")
        {
            std::cout << "CVIA 客户端运行器\n\n"
                      << "用法：./cvia_customer [--host 地址] [--port 端口]\n"
                      << "默认监听 0.0.0.0:8765，也可使用 WEB_HOST/WEB_PORT 环境变量。\n";
            return 0;
        }
        if ((argument == "--host" || argument == "--port") && index + 1 < argc)
        {
            const std::string value = argv[++index];
            if (argument == "--host") host = value;
            else port = value;
            continue;
        }
        std::cerr << "[CVIA] 无效参数：" << argument << "；使用 --help 查看说明。\n";
        return 2;
    }
    if (host.empty() || !valid_port(port))
    {
        std::cerr << "[CVIA] 主机地址不能为空，端口必须为 1～65535。\n";
        return 2;
    }

    const std::string root = executable_directory(argv[0]);
    const std::string runtime = root + "/bin/cvia_runtime";
    const std::string config = root + "/config/config.json";
    const std::string server = root + "/web_monitor/server.so";
    const std::string bridge = root + "/web_monitor/mocap/bin/MocapBridge";
    const std::string sdk_library_directory = root + "/web_monitor/mocap/lib/aarch64";
    if (!require_file(runtime, true) || !require_file(config) ||
        !require_file(server) || !require_file(bridge, true))
        return 3;

    const std::string old_library_path = environment_or("LD_LIBRARY_PATH", "");
    const std::string library_path = sdk_library_directory +
        (old_library_path.empty() ? "" : ":" + old_library_path);
    setenv("CVIA_TRT_BINARY", runtime.c_str(), 1);
    setenv("CVIA_CONFIG_FILE", config.c_str(), 1);
    setenv("CVIA_MOCAP_BRIDGE", bridge.c_str(), 1);
    setenv("LD_LIBRARY_PATH", library_path.c_str(), 1);
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);
    setenv("PYTHONOPTIMIZE", "2", 1);
    setenv("PYTHONNOUSERSITE", "1", 1);
    const std::string old_python_path = environment_or("PYTHONPATH", "");
    const std::string python_path = root + "/web_monitor" +
        (old_python_path.empty() ? "" : ":" + old_python_path);
    setenv("PYTHONPATH", python_path.c_str(), 1);

    if (chdir(root.c_str()) != 0)
    {
        std::cerr << "[CVIA] 无法进入安装目录：" << std::strerror(errno) << '\n';
        return 4;
    }

    const std::string python = environment_or("PYTHON_BIN", "python3");
    std::vector<std::string> arguments = {
        python, "-OO", "-c", "import server; server.main()",
        "--host", host, "--port", port,
    };
    std::vector<char *> raw_arguments;
    raw_arguments.reserve(arguments.size() + 1);
    for (std::string &argument : arguments)
        raw_arguments.push_back(&argument[0]);
    raw_arguments.push_back(nullptr);

    std::cout << "[CVIA] 安装目录：" << root << '\n'
              << "[CVIA] 网页控制台：http://" << host << ':' << port << '\n'
              << "[CVIA] 推理初始为停止状态；请在网页保存参数后点击‘开始推理’。\n"
              << "[CVIA] 按 Ctrl+C 将安全关闭网页和推理进程。\n";
    std::cout.flush();
    execvp(python.c_str(), raw_arguments.data());
    std::cerr << "[CVIA] 无法启动 Python 3：" << std::strerror(errno) << '\n';
    return 5;
}
