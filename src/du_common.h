#ifndef DU_COMMON_H
#define DU_COMMON_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <DarkHelp.hpp>

using std::to_string;
static constexpr float kStrongIntersectionThresh = 0.45;

// returns relaive bbox of prediction result. Note that x,y are still the coordinates of top-left corner, just in [0,1] interval.
inline cv::Rect2f relativeBbox(const DarkHelp::PredictionResult& r) {
    return cv::Rect2f(
            r.original_point.x - r.original_size.width/2,
            r.original_point.y - r.original_size.height/2,
            r.original_size.width,
            r.original_size.height);
}

// returns probability of class \param classId, or 0 if class isn't there
inline float getProb(const DarkHelp::PredictionResult& r, int classId) {
    return (r.all_probabilities.cend() == r.all_probabilities.find(classId)) ? 0 : r.all_probabilities.at(classId);
}

// Ground truth detection loaded from .txt file in darknet format
struct LoadedDetection {
    int classId;
    // relative bbox
    cv::Rect2f bbox;
    std::string filename;

    // outputs detection as "c x y w h % iou filename", where (x,y) is relative mid-point, (w,h) is relative size
    std::string toString() const;
    // bbox is between 0 and 1; classId > 0, filename not empty
    bool isValid() const;
};

struct ComparisonResult {
    int classId; // as predicted by darknet
    cv::Rect2f bbox; // relative bb as predicted by darknet
    float prob; // probability of that class (0-1). If marked by human but wasn't detected by darknet, prob = -1
    float iou;
    std::string filename; // with no extension nor slashes

    // to specified format
    std::string toString() const;
    // read from string in .duv format
    static ComparisonResult fromString(const std::string& str);
    // returns false if classId < 0, prop<0, iou < 0 or filename is empty or slashy
    bool isValid() const;
};
typedef std::vector<ComparisonResult> ComparisonResults;
// newline-separated results, with "\n" at the end as well
std::string to_string(const ComparisonResults& results);
ComparisonResults comparisonResultsFromFile(const std::string& filename);

// rect to human-readable string (not compatible with darknet mark .txt files!)
template<class Tp>
inline std::string to_string(const cv::Rect_<Tp> r) {
    return to_string(r.x) + ":" + to_string(r.y) + ":" + to_string(r.width) + "x" + to_string(r.height);
}

// area of Intersection over area of Union [0-1]
float intersectionOverUnion(const cv::Rect2f& r1, const cv::Rect2f& r2);

// returns list of filenames (training images) in folder without extension, sorted alphabetically
std::vector<std::string> loadTrainImageFilenames(const std::string& path);
std::vector<LoadedDetection> loadDetsFromFile(const std::string& path);

#endif // DU_COMMON_H
