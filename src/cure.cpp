#include "cure.h"
#include "helpers.h"
#include "cv_funcs.h"
#include "du_common.h"

using namespace cv;
using namespace cvColors;
constexpr const char* windowName = "cure your dataset";

// pathToTrainData - path to dir with .txt and .jpg files, pathToDuv - /path/to/compareResults.duv
void cureDataset(const std::string& pathToTrainData
               , const std::string& pathToDuv
               , const std::string& pathToNames) {
    // load
    auto cmpResults = comparisonResultsFromFile(pathToDuv);
    auto names = getFileContentsAsStringVector(pathToNames);

    // sort: false positive duv by probability, false negative by... area?
    ComparisonResults toAdd;
    ComparisonResults toRemove;

    for (const auto& r: cmpResults) {
        if (r.prob >= kValidationProbThresh && r.iou < kStrongIntersectionThresh)
            toAdd.push_back(r);
        if (r.prob < kValidationProbThresh && r.iou < kStrongIntersectionThresh)
            toRemove.push_back(r);
    }

    LOG(INFO) << "loaded " << toAdd.size() << " marks to add and " << toRemove.size() << " marks to remove";

    std::sort(toAdd.begin(), toAdd.end(), ProbIsBigger);
    std::sort(toRemove.begin(), toRemove.end(), AreaIsBigger);

    // show things to add interactively
    int indexOfToAdd = -1, indexOfToRemove = -1;
    bool showingToAdd = true;
    while (true) {
        ++(showingToAdd ? indexOfToAdd : indexOfToRemove);
        if (indexOfToAdd >= toAdd.size() && indexOfToRemove >= toRemove.size()) {
            LOG(INFO) << "Finished cure. Toadd index = " << indexOfToAdd << ", ToRemove index = " << indexOfToRemove;
            break;
        }
        if (showingToAdd && indexOfToAdd >= toAdd.size()) {
            showingToAdd = false;
            LOG(WARNING) << "cureDataset: no more marks to add. Switching to toRemove";
            continue;
        }
        if (!showingToAdd && indexOfToRemove >= toRemove.size()) {
            showingToAdd = true;
            LOG(WARNING) << "cureDataset: no more marks to remove. Switching to toAdd";
            continue;
        }
        ComparisonResults& cr = showingToAdd ? toAdd : toRemove;
        auto imgPath = pathToTrainData + "/" + cr[indexOfToAdd].filename + ".jpg";
        auto detsPath = pathToTrainData + "/" + cr[indexOfToAdd].filename + ".txt";
        cv::Mat img = imread(imgPath);
        if (nullptr == img.data) {
            LOG(ERROR) << "failed to load image " << imgPath;
            continue;
        }
        LoadedDetections dets = loadedDetectionsFromFile(detsPath);

        if (showingToAdd) {
            auto& detToAdd = toAdd[indexOfToAdd];
            // see if we've already added this
            int foundDetIndex = findDetection(dets, detToAdd.toLoadedDet());
            if (foundDetIndex >= 0) {
                LOG(WARNING) << "found detection that has already been added to dataset: \""
                    << detToAdd.toString() << "\" is already in file " << detsPath <<". Please generate new .duv file.";
                continue;
            }
            drawBbox(img, detToAdd.bbox, cvColorRed, 2);
            imshow(windowName, img);
            int key = cv::waitKey(0);
            LOG(INFO) << "key pressed = " << key;
        } else {
            LOG(INFO) << "toRemove = not implemented yet ";
        }
    }
}
