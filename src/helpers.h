#ifndef SMC_HELPERS_H
#define SMC_HELPERS_H

#include "easylogging++.h"
#include <string>
#include <vector>
#include <cmath>

// returns file contents as string
std::string getFileContents(const std::string& filename);

std::vector<std::string> getFileContentsAsStringVector(const std::string& filename, bool quiet = false);

// returns absolute path to executable
std::string getExecutablePath();

// get full app path
std::string applicationPath();

// "/path/to/abc.txt" -> "abc"
std::string extractFilenameFromFullPath(const std::string& path);
// if /path/to/dir doesn't have trailing slash, add it
inline std::string addSlash(const std::string& path) {return (!path.empty() && path.back() == '/' ? path : path+"/");}

// write text to file. Returns true if successful
bool saveToFile(const std::string& path, const std::string& content, bool append = false);

// ls command
std::vector<std::string> listFilesInDir(const std::string& dirPath);

template<class T>
std::string vectorToString(const std::vector<T>& v);

// string with space-separated values to array, e.g. stringToArray<float>("0.1 0.2 0.3") -> {0.1, 0.2, 0.3}
template<class T>
std::vector<T> stringToArray(const std::string& str) {
    std::stringstream ss(str);
    std::vector<T> result;
    T val;
    while (ss >> val)
        result.push_back(val);
    return result;
}

// split string s by character c
std::vector<std::string> splitString(const std::string s, char c);

// returns true if folder with this path exists
bool ifFolderExists(const std::string& path);

// check if folder exist and create one (non-recursively). Returns false if failed to create
bool createFolderIfDoesntExist(const std::string& path);

// returns true if @param fullString ends with @param ending
static inline bool strEndsWith(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

// true if |f1-f2| < delta
inline bool almostEqual(float f1, float f2) {
    constexpr float delta = 1e-7;
    return fabsf(f1 - f2) < delta;
}


// returns current timestamp in seconds
uint32_t currentTimestamp();

// converts int to string of size @param length with leanding zeros
std::string leadingZeros(int number, int length);

// returns current date string with format specified in ctime/strftime
std::string currentDateString(const std::string& format = "%Y%m%d");

#endif // SMC_HELPERS_H
