#include "du_common.h"
#include "helpers.h"
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

static const std::string kDotJpg{".jpg"};

std::string ComparisonResult::toString() const {
    double x = bbox.x + bbox.width / 2;
    double y = bbox.y + bbox.height / 2;
    char treatedChar = treated ? 't' : 'f';
    ostringstream ss;
    ss << filename << '\t' << classId << '\t' << x << '\t' << y << '\t' << bbox.width
           << '\t' << bbox.height << '\t' << prob << '\t' << iou << '\t' << treatedChar;
    return ss.str();
}

std::string to_string(const ComparisonResults& results) {
    std::string s;
    for (const auto& r: results)
        s += r.toString() + "\n";
    return s;
}

std::string to_string(const LoadedDetections& dets) {
    std::string s;
    for (const auto& d: dets)
        s += d.toString() + "\n";
    // remove last '\n'
    if (!s.empty())
        s.pop_back();
    return s;
}

std::string LoadedDetection::toHumanString() const {
    return to_string(classId) + " " + to_human_string(bbox) + ", " + filename;
}

std::string LoadedDetection::toString() const {
    double x = bbox.x + bbox.width / 2.;
    double y = bbox.y + bbox.height / 2.;
    stringstream ss;
    ss << classId << ' ' << x << ' ' << y << ' ' << bbox.width << ' ' << bbox.height;
    return ss.str();
}

bool LoadedDetection::isValid() const {
    return !filename.empty() && classId >= 0
        && bbox.x >= 0 && bbox.x <= 1
        && bbox.y >= 0 && bbox.y <= 1
        && bbox.width >= 0 && bbox.width <= 1
        && bbox.height >= 0 && bbox.height <= 1;
}

float intersectionOverUnioneconst cv::Rect2d& r1, const cv::Rect2d& r2) {
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

std::vector<std::string> loadTrainImageFilenames(const std::string& path, bool labeledFiles) {
    std::vector<std::string> filesList = listFilesInDir(path);
    std::vector<std::string> result;
    result.reserve(filesList.size()/2);
    std::sort(filesList.begin(), filesList.end());

    // add file to 'result' if two consecutive files have same basenames and end with ".jpg" and ".txt" accordingly
    for (size_t i = 0; i < filesList.size() - 1; ++i) {
        std::string& curr = filesList[i];
        if (std::string::npos == curr.rfind(kDotJpg))
            continue;
        // find basename.txt file follows after
        std::string baseFileName = getBaseFileName(curr);
        int txtFileIndex = i;
        bool txtFileExists = false;
        while (!txtFileExists
                && txtFileIndex < filesList.size()
                && baseFileName == getBaseFileName(filesList[txtFileIndex])) {
            txtFileExists |= (baseFileName + ".txt" == filesList[txtFileIndex]);
            ++txtFileIndex;
        }
        if ((labeledFiles && txtFileExists) || (!labeledFiles && !txtFileExists))
            result.push_back(baseFileName);
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
        double midX, midY, relW, relH;
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

        cv::Rect2d bbox(midX - relW/2, midY - relH/2, relW, relH);
        LoadedDetection d{classId, bbox, extractFilenameFromFullPath(path)};
        result.push_back(d);
    }

    return result;
}

ComparisonResult ComparisonResult::fromString(const std::string& str) {
    static const ComparisonResult invalidResult = ComparisonResult::generateInvalid();
    ComparisonResult r{invalidResult};
    vector<string> parts = splitString(str, '\t');
    // "filename c x y w h p iou treated", tab-separated
    if (parts.size() != 9 || (parts[8] != "t" && parts[8] != "f")) {
        LOG(ERROR) << "ComparisonResult: bad string " << str;
        return invalidResult;
    }

    double midX, midY;
    char treatedChar;
    r.filename = parts[0];

    try {
        r.classId =     stoi(parts[1]);
        midX =          stof(parts[2]);
        midY =          stof(parts[3]);
        r.bbox.width =  stof(parts[4]);
        r.bbox.height = stof(parts[5]);
        r.prob =        stof(parts[6]);
        r.iou =         stof(parts[7]);
        treatedChar =        parts[8].front();
    } catch (std::exception& e) {
        LOG(ERROR) << "ComparisonResult::fromString: bad string " << str << ". " << e.what();
        return invalidResult;
    }

    r.bbox.x = midX - r.bbox.width/2;
    r.bbox.y = midY - r.bbox.height/2;
    r.treated = (treatedChar == 't');

    return r;
}

bool ComparisonResult::isValid() const {
    return (!filename.empty() && classId >= 0 && iou >= 0 && prob >= 0);
}

ComparisonResult ComparisonResult::generateInvalid() {
    return ComparisonResult{-1, cv::Rect2d(-1,-1,-1,-1), -1, -1, "", false};
}

bool ComparisonResult::isToAdd() const {
    return !treated
            && prob >= kValidationProbThresh
            && iou < kStrongIntersectionThresh;
}
bool ComparisonResult::isToRemove() const {
    return !treated
            && prob < kValidationProbThresh
            && iou < kStrongIntersectionThresh;
}

LoadedDetection ComparisonResult::toLoadedDet() const {
    return LoadedDetection{classId, bbox, filename};
}

ComparisonResults comparisonResultsFromFile(const std::string& filename, bool ignoreTreatedDets) {
    ComparisonResults rs;
    auto lines = getFileContentsAsStringVector(filename);
    for (const auto& l: lines) {
        ComparisonResult r = ComparisonResult::fromString(l);
        if (!r.isValid()) {
            LOG(ERROR) << "Can not parse line to ComparisonResults: " << l;
        } else if (!ignoreTreatedDets || !r.treated) {
            rs.push_back(r);
        }
    }

    return rs;
}

int findDetection(const LoadedDetections& dets, const LoadedDetection& needle) {
    const string needleFilename = extractFilenameFromFullPath(needle.filename);
    constexpr float kIouThresh = 0.99; // if iou(r1, r2) > 0.99, consider r1 ~= r2
    for (int i = 0; i < int(dets.size()); ++i) {
        if (dets[i].classId == needle.classId
                && extractFilenameFromFullPath(dets[i].filename) == needleFilename
                && intersectionOverUnion(dets[i].bbox, needle.bbox) > kIouThresh)
            return i;
    }
    return -1;
}

std::vector<string> loadPathsToImages(const string &pathToTrainTxt) {
    std::vector<std::string> result;

    // retreive the location of train.txt file
    std::string listPath;
    auto ios = pathToTrainTxt.find_last_of('/');
    if (std::string::npos != ios)
        listPath = pathToTrainTxt.substr(0, ios + 1);

    // load list of image paths from train.txt
    auto list = getFileContentsAsStringVector(pathToTrainTxt, false);
    int lineNumber{-1};
    for (std::string s: list) {
        ++lineNumber;
        if (!strEndsWith(s, kDotJpg)) {
            LOG_N_TIMES(1, ERROR) << "Bad image path. Line#" << lineNumber << " in " << pathToTrainTxt << ": " << s
                                     << ". Other errors truncated";
            continue;
        }
        s = s.substr(0, s.size() - kDotJpg.size());
        if ('/' == s.front())
            result.push_back(s); // absolute image path
        else
            result.push_back(listPath + s); // image path relative to train.txt
    }
    return result;
}
