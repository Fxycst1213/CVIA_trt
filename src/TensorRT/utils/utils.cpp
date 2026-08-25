#include <experimental/filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "utils.hpp"
#include "NvInfer.h"
#include "model.hpp"
#include "logger.hpp"

using namespace std;

bool fileExists(const string fileName)
{
    if (!experimental::filesystem::exists(
            experimental::filesystem::path(fileName)))
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool fileRead(const string &path, vector<unsigned char> &data, size_t &size)
{
    stringstream trtModelStream;
    ifstream cache(path);

    /* 将engine的内容写入trtModelStream中*/
    trtModelStream.seekg(0, trtModelStream.beg);
    trtModelStream << cache.rdbuf();
    cache.close();

    /* 计算model的大小*/
    trtModelStream.seekg(0, ios::end);
    size = trtModelStream.tellg();

    vector<uint8_t> tmp;
    trtModelStream.seekg(0, ios::beg);
    tmp.resize(size);

    /* 将trtModelStream中的stream通过read函数写入modelMem中*/
    trtModelStream.read((char *)data[0], size);
    return true;
}

vector<unsigned char> loadFile(const string &file)
{
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    vector<uint8_t> data;
    if (length > 0)
    {
        in.seekg(0, ios::beg);
        data.resize(length);
        in.read((char *)&data[0], length);
    }
    in.close();
    return data;
}

vector<string> loadDataList(const string &file)
{
    vector<string> list;
    auto *f = fopen(file.c_str(), "r");
    if (!f)
        LOGE("Failed to open %s", file.c_str());

    char str[512];
    while (fgets(str, 512, f) != NULL)
    {
        for (int i = 0; str[i] != '\0'; ++i)
        {
            if (str[i] == '\n')
            {
                str[i] = '\0';
                break;
            }
        }
        list.push_back(str);
    }
    fclose(f);
    return list;
}

string printDims(const nvinfer1::Dims dims)
{
    int n = 0;
    char buff[100];
    string result;

    n += snprintf(buff + n, sizeof(buff) - n, "[ ");
    for (int i = 0; i < dims.nbDims; i++)
    {
        n += snprintf(buff + n, sizeof(buff) - n, "%d", dims.d[i]);
        if (i != dims.nbDims - 1)
        {
            n += snprintf(buff + n, sizeof(buff) - n, ", ");
        }
    }
    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}

string printTensor(float *tensor, int size)
{
    int n = 0;
    char buff[100];
    string result;
    n += snprintf(buff + n, sizeof(buff) - n, "[ ");
    for (int i = 0; i < size; i++)
    {
        n += snprintf(buff + n, sizeof(buff) - n, "%8.4lf", tensor[i]);
        if (i != size - 1)
        {
            n += snprintf(buff + n, sizeof(buff) - n, ", ");
        }
    }
    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}

string printTensorShape(nvinfer1::ITensor *tensor)
{
    string str;
    str += "[";
    auto dims = tensor->getDimensions();
    for (int j = 0; j < dims.nbDims; j++)
    {
        str += to_string(dims.d[j]);
        if (j != dims.nbDims - 1)
        {
            str += " x ";
        }
    }
    str += "]";
    return str;
}

string changePath(string srcPath, string relativePath, string postfix, string tag)
{
    const experimental::filesystem::path source(srcPath);
    string filename = source.stem().string();
    if (!tag.empty()) filename += "-" + tag;
    filename += postfix;
    // Lexically normalize "onnx/../engine" before checking the filesystem.
    // std::experimental::filesystem in the C++14 toolchain has no
    // path::lexically_normal(), so resolve the relative components explicitly.
    experimental::filesystem::path destination = source.parent_path();
    for (const auto &component : experimental::filesystem::path(relativePath))
    {
        if (component == ".") continue;
        if (component == "..") destination = destination.parent_path();
        else destination /= component;
    }
    return (destination / filename).string();
}

string getOutputPath(string srcPath, string postfix)
{
    int pos = srcPath.rfind(".");
    string tarPath;
    tarPath = srcPath.substr(0, pos);
    tarPath += "_" + postfix + ".png";
    return tarPath;
}

string getFileType(string filePath)
{
    int pos = filePath.rfind(".");
    string suffix;
    suffix = filePath.substr(pos, filePath.length());
    return suffix;
}

string getFileName(string filePath)
{
    int pos = filePath.rfind("/");
    string suffix;
    suffix = filePath.substr(pos + 1, filePath.length());
    return suffix;
}

string getPrecision(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT:
        return "FP32";
    case nvinfer1::DataType::kHALF:
        return "FP16";
    case nvinfer1::DataType::kINT32:
        return "INT32";
    case nvinfer1::DataType::kINT8:
        return "INT8";
    case nvinfer1::DataType::kUINT8:
        return "UINT8";
    default:
        return "unknown";
    }
}

string getPrecision(model::precision prec)
{
    switch (prec)
    {
    case model::precision::FP16:
        return "FP16";
    case model::precision::INT8:
        return "INT8";
    default:
        return "FP32";
    }
}
