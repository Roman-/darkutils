#include <helpers.h>

// for absolute path
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <libgen.h>         // dirname
#include <unistd.h>         // readlink
#include <linux/limits.h>   // PATH_MAX
#include <unistd.h>         // readlink
#include <dirent.h>         // struct dirent
#include <chrono>

using std::string;
using std::vector;
using std::to_string;

std::string getExecutablePath() {
    static std::string exePath;
    if (exePath == "") {
        char result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
        if (count == -1)
            LOG(FATAL) << "can not extract absolute path of a process, exiting";

        exePath = dirname(result);
    }
    return exePath;
}

std::string extractFilenameFromFullPath(const std::string& pathOriginal) {
    std::string s = pathOriginal;
    // remove /path/to/
    size_t lastIoSlash = s.find_last_of("/\\");
    s = s.substr(lastIoSlash + 1);
    size_t ioDot = s.find_last_of('.');
    if (std::string::npos != ioDot)
        s = s.substr(0, ioDot);
    return s;
}


bool saveToFile(const std::string& path, const std::string& content, bool append) {
    std::ofstream outfile;
    outfile.open(path, append ? std::ios_base::app : std::ios_base::out);
    if (!outfile.is_open())
        return false;

    outfile << content;
    outfile.close();
    return true;
}

std::string getFileContents(const std::string& filename) {
    // TODO STOPPED THERE
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        LOG(ERROR) << "getFileContents: can\'t open file " << filename;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();

    file.close();
    return buffer.str();
}

template<class T>
std::string vectorToString(const std::vector<T>& v) {
    std::string result;
    for (const T& t: v)
        result += std::to_string(t) + " ";
    return result;
}

std::vector<std::string> listFilesInDir(const std::string& dirPath) {
    std::vector<std::string> result;
    DIR *d;
    struct dirent *dir;
    d = opendir(dirPath.c_str());
    if (!d) {
        LOG(ERROR) << "failed to list files in dir " << dirPath;
        return result;
    }

    while ((dir = readdir(d)) != NULL) {
        std::string filename = std::string(dir->d_name);
        if (filename != "." && filename != "..") {
            result.push_back(filename);
        }
    }
    closedir(d);
    return result;
}

std::vector<std::string> splitString(const std::string inputString, char c) {
    std::vector<std::string> result;
    std::istringstream f(inputString);
    string part;
    while (getline(f, part, c)) {
        result.push_back(part);
    }
    return result;
}

bool ifFolderExists(const std::string& path) {
    struct stat sb;
    return (stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode));
}

bool createFolderIfDoesntExist(const std::string& path) {
    if (!ifFolderExists(path)) {
        if (mkdir(path.c_str(), 0777) == -1) {
            LOG(ERROR) << "Failed to create path " << path;
            return false;
        } else {
            LOG(INFO) << "Created directory: " << path;
        }
    }
    return true;
}

std::vector<std::string> getFileContentsAsStringVector(const std::string& filename, bool quiet) {
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        if (!quiet)
            LOG(ERROR) << "getFileContents: can\'t open file " << filename;
        return std::vector<std::string>();
    }
    vector<string> result;
    for (std::string line; std::getline(file, line); ) {
        result.push_back(line);
    }

    file.close();
    return result;
}

uint32_t currentTimestamp() {
    auto p = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::seconds>(p.time_since_epoch()).count();
}

std::string leadingZeros(int number, int length) {
    std::string s = to_string(number);
    int numZeros = length - s.size();
    return (numZeros > 0) ? (std::string(numZeros, '0') + s) : s;
}

std::string currentDateString(const std::string& format) {
    static const int kBufferSize = 80;
    time_t rawtime;
    struct tm* timeinfo;
    char buffer[kBufferSize];
    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime (buffer, kBufferSize, format.c_str(), timeinfo);
    return std::string(buffer);
}

