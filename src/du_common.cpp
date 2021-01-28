#include <du_common.h>
#include "helpers.h"
#include <algorithm>
#include <string>
#include <vector>

using namespace std;

std::string ComparisonResult::toString() const {
    float x = bbox.x + bbox.width / 2;
    float y = bbox.y + bbox.height / 2;
    return to_string(classId) + " " + to_string(x) + " " + to_string(y) + " " + to_string(bbox.width)
           + " " + to_string(bbox.height) + " " + to_string(prob) + " " + to_string(iou) + " " + filename;
}

std::string to_string(const ComparisonResults& results) {
    std::string s;
    for (const auto& r: results)
        s += r.toString() + "\n";
    return s;
}

std::string to_string(const cv::Rect2f& r) {
    return to_string(r.x) + ":" + to_string(r.y) + ":" + to_string(r.width) + "x" + to_string(r.height);
}

std::string LoadedDetection::toString() const {
    return to_string(classId) + " " + to_string(bbox) + ", " + filename;
}

bool LoadedDetection::isValid() const {
    return !filename.empty() && classId >= 0
        && bbox.x >= 0 && bbox.x <= 1
        && bbox.y >= 0 && bbox.y <= 1
        && bbox.width >= 0 && bbox.width <= 1
        && bbox.height >= 0 && bbox.height <= 1;
}

float intersectionOverUnion(const cv::Rect2f& r1, const cv::Rect2f& r2) {
    float r1left = r1.x;
    float r1right = r1.x + r1.width;
    float r1top = r1.y;
    float r1bottom = r1.y + r1.height;

    float r2left = r2.x;
    float r2right = r2.x + r2.width;
    float r2top = r2.y;
    float r2bottom = r2.y + r2.height;

    float left = max(r1left, r2left);
    float right = min(r1right, r2right);
    float bottom = min(r1bottom, r2bottom);
    float top = max(r1top, r2top);

    if (left >= right || top >= bottom)
        return 0; // do not intersect

    float areaOfIntersection = (right - left) * (bottom - top);
    float areaOfUnion = r1.area() + r2.area() - areaOfIntersection;
    float iou = areaOfIntersection / areaOfUnion;
    return iou;
}

std::vector<std::string> loadTrainImageFilenames(const std::string& path) {
    std::vector<std::string> filesList = listFilesInDir(path);
    std::vector<std::string> result;
    std::sort(filesList.begin(), filesList.end());

    // add file to 'result' if two consecutive files have same basenames and end with ".jpg" and ".txt" accordingly
    for (size_t i = 0; i < filesList.size() - 1; ++i) {
        std::string& curr = filesList[i];
        std::string& next = filesList[i + 1];
        if (curr.size() < 5 || next.size() < 5 || curr.find(".jpg") == string::npos || next.find(".txt") == string::npos)
            continue;
        std::string baseFileName = curr.substr(0, curr.find('.'));
        if (baseFileName.size() > 0 && next.rfind(baseFileName, 0) == 0) {
            result.push_back(baseFileName);
            ++i; // skip next file to skip the pair
        }
    }
    return result;
}

LoadedDetections loadedDetectionsFromFile(const std::string& path) {
    vector<LoadedDetection> result;
    auto content = getFileContents(path);
    auto l = splitString(content, '\n');
    for (const std::string& s: l) {
        auto parts = splitString(s, ' ');
        if (parts.size() != 5) {
            LOG(ERROR) << "loadedDetectionsFromFile: bad line \"" << s << "\"";
            continue;
        }

        int classId;
        float midX, midY, relW, relH;
        try {
            classId = stoi(parts[0].c_str());
            midX = stof(parts[1].c_str());
            midY = stof(parts[2].c_str());
            relW = stof(parts[3].c_str());
            relH = stof(parts[4].c_str());
        } catch (const std::exception& ex) {
            LOG(ERROR) << "loadedDetectionsFromFile: failed to parse line in " << path << ". Line:\n" << s;
            continue;
        }

        cv::Rect2f bbox(midX - relW/2, midY - relH/2, relW, relH);
        LoadedDetection d{classId, bbox, extractFilenameFromFullPath(path)};
        result.push_back(d);
    }

    return result;
}

ComparisonResult ComparisonResult::fromString(const std::string& str) {
    static const ComparisonResult invalidResult{-1, cv::Rect2f(), -1, -1, ""};
    ComparisonResult r{-1, cv::Rect2f(), -1, -1, ""};
    // c x y w h p iou filename
    if (std::count(str.begin(), str.end(), ' ') < 7) {
        LOG(ERROR) << "ComparisonResult: bad string " << str;
        return invalidResult;
    }
    std::istringstream iss(str);

    float midX, midY;

    if (!(iss >> r.classId >> midX >> midY >> r.bbox.width >> r.bbox.height >> r.prob >> r.iou)) {
        LOG(ERROR) << "failed to convert string to ComparisonResult: " << str;
        return invalidResult;
    }
    r.bbox.x = midX - r.bbox.width/2;
    r.bbox.y = midY - r.bbox.height/2;

    // get the rest of the line as filename
    constexpr int kMaxFileNameSize = 64;
    char buff[kMaxFileNameSize] = {0};
    iss.getline(buff, kMaxFileNameSize, '\n');

    r.filename = std::string(buff).substr(1); // remove extra space form the beginning

    return r;
}

bool ComparisonResult::isValid() const {
    if (filename.empty() || string::npos != filename.find('/'))
        return false;
    if (classId < 0 || iou < 0 || prob < 0)
        return false;
    return true;
}

LoadedDetection ComparisonResult::toLoadedDet() const {
    return LoadedDetection{classId, bbox, filename};
}

ComparisonResults comparisonResultsFromFile(const std::string& filename) {
    ComparisonResults rs;
    auto lines = getFileContentsAsStringVector(filename);
    for (const auto& l: lines) {
        ComparisonResult r = ComparisonResult::fromString(l);
        if (r.isValid()) {
            rs.push_back(r);
        } else {
            LOG(ERROR) << "Can not parse line to ComparisonResults: " << l;
        }
    }

    return rs;
}

int findDetection(const LoadedDetections& dets, const LoadedDetection& needle) {
    constexpr float kIouThresh = 0.99; // if iou(r1, r2) > 0.99, consider r1 ~= r2
    for (int i = 0; i < int(dets.size()); ++i) {
        if (dets[i].classId == needle.classId
                && dets[i].filename == needle.filename
                && intersectionOverUnion(dets[i].bbox, needle.bbox) > kIouThresh)
            return i;
    }
    return -1;
}
