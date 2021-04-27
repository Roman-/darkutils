#include "cure.h"
#include "helpers.h"
#include "cv_funcs.h"
#include "du_common.h"

using namespace cv;
using namespace cvColors;
constexpr const char* windowName = "cure your dataset";
constexpr const char* backupFolderPath = "backup_dataset/";
constexpr int kWindowWidth = 1000;
constexpr int kWindowHeight = 600;

// returns image which is not exceeding kWindowWidth * kWindowWidth and aspect ratio is keeped
static cv::Mat resizedToWindow(cv::Mat img) {
    cv::Size windowSize(kWindowWidth, kWindowHeight);
    const float windowAr = float(kWindowWidth) / kWindowHeight;
    int w = img.cols;
    int h = img.rows;
    float imgAr = (float)w / h;

    cv::Size finalSize;
    if (imgAr > windowAr) {
        // keep width, shrink height
        finalSize.width = kWindowWidth;
        finalSize.height = float(kWindowWidth) / imgAr;
    } else {
        // keep height, shrink width
        finalSize.height = kWindowHeight;
        finalSize.width = float(kWindowHeight) * imgAr;
    }
    cv::Mat result;
    cv::resize(img, result, finalSize);
    return result;
}

// pathToTrainData - path to dir with .txt and .jpg files, pathToDuv - /path/to/compareResults.duv
void cureDataset(const std::string& pathToDuv
               , const std::string& pathToNames) {
    bool backupFolderCreated = createFolderIfDoesntExist(backupFolderPath);
    LOG_IF(!backupFolderCreated, ERROR) << "failed to create " << backupFolderPath << ", backups will be omitted";

    // load
    std::string workPath = extractFileLocationFromFullPath(pathToDuv);
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
    int indexOfToAdd = -1, indexOfToRemove = -1, numToAdd = int(toAdd.size()), numToRemove = int(toRemove.size());
    bool showingToAdd = true;
    static const std::set<char> allowedKeysInAddMode =    {'y', 'n', char(27), 's'}; // accept, no (dont accept), exit, switch
    static const std::set<char> allowedKeysInRemoveMode = {'d', 'k', char(27), 's'}; // delete, keep (dont delete), exit, switch
    int key; // key pressed by user
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, kWindowWidth, kWindowHeight);
    while (true) {
        ++(showingToAdd ? indexOfToAdd : indexOfToRemove);
        if (indexOfToAdd >= numToAdd && indexOfToRemove >= numToRemove) {
            LOG(INFO) << "Finished cure. Toadd index = " << indexOfToAdd << ", ToRemove index = " << indexOfToRemove;
            break;
        }
        if (showingToAdd && indexOfToAdd >= numToAdd) {
            showingToAdd = false;
            LOG(WARNING) << "cureDataset: no more marks to add. Switching to toRemove";
            continue;
        }
        if (!showingToAdd && indexOfToRemove >= numToRemove) {
            showingToAdd = true;
            LOG(WARNING) << "cureDataset: no more marks to remove. Switching to toAdd";
            continue;
        }
        ComparisonResult& cr = showingToAdd ? toAdd[indexOfToAdd] : toRemove[indexOfToRemove];
        auto imgPath = workPath + cr.filename + ".jpg";
        auto detsPath = workPath + cr.filename + ".txt";
        cv::Mat img = imread(imgPath);
        if (nullptr == img.data) {
            LOG(ERROR) << "failed to load image " << imgPath;
            continue;
        }
        cv::Mat imgScaled = resizedToWindow(img);
        LoadedDetections dets = loadedDetectionsFromFile(detsPath);
        const std::string pathToTxtBackup = std::string(backupFolderPath) + "/" + cr.filename + ".txt";

        if (showingToAdd) {
            // see if we've already added this
            int foundDetIndex = findDetection(dets, cr.toLoadedDet());
            if (foundDetIndex >= 0) {
                LOG(ERROR) << "Found detection that has already been added to dataset: \""
                    << cr.toString() << "\" is already in file " << detsPath <<". Please re-generate the .duv file"
                    " by running validate command in darkutils.";
                continue;
            }
            cv::line(imgScaled, cv::Point(0,0), cv::Point(cr.bbox.tl().x * imgScaled.cols, cr.bbox.tl().y * imgScaled.rows)
                        , colorByClass(cr.classId), 2, cv::LINE_8, 0);
            drawBbox(imgScaled, cr.bbox, colorByClass(cr.classId), 1);
            const std::string txt = "Add this " + names.at(cr.classId) + " (" + to_string(int(cr.prob*100)) + "%) to dataset? y/n";
            cv::putText(imgScaled, txt, cv::Point(0,25), cv::FONT_HERSHEY_PLAIN, 1, cvColorWhite, 3);
            cv::putText(imgScaled, txt, cv::Point(0,25), cv::FONT_HERSHEY_PLAIN, 1, cvColorBlack, 1);
            imshow(windowName, imgScaled);
            do {
                key = cv::waitKey(0);
            } while (allowedKeysInAddMode.find(key) == allowedKeysInAddMode.cend());
            if ('y' == key) {
                LOG(INFO) << "appending mark " << cr.toString() << " to " << detsPath;
                // save backup
                if (!ifFileExists(pathToTxtBackup))
                    saveToFile(pathToTxtBackup, to_string(dets));
                else
                    LOG(INFO) << "backup for " << cr.filename << " already exist in " << backupFolderPath << ", dont overwrite.";
                // add this detection and overwrite original file
                dets.push_back(cr.toLoadedDet());
                saveToFile(detsPath, to_string(dets));
            }
        } else {
            // see if we've already removed
            int foundDetIndex = findDetection(dets, cr.toLoadedDet());
            if (foundDetIndex < 0) {
                LOG(ERROR) << "Detection \"" << cr.toString() << "\" from file " << pathToDuv
                    << " was not found in dataset:" << detsPath << ". Please re-generate the .duv file"
                    " by running validate command in darkutils.";
                continue;
            }
            drawBboxCrossed(imgScaled, cr.bbox, colorByClass(cr.classId), 1, 1);
            cv::line(imgScaled, cv::Point(0,0), cv::Point(cr.bbox.tl().x * imgScaled.cols, cr.bbox.tl().y * imgScaled.rows)
                    , colorByClass(cr.classId), 2, cv::LINE_8, 0);
            const std::string txt = "REMOVE this " + names.at(cr.classId) + " from dataset? d/k";
            cv::putText(imgScaled, txt, cv::Point(0,25), cv::FONT_HERSHEY_PLAIN, 1, cvColorWhite, 3);
            cv::putText(imgScaled, txt, cv::Point(0,25), cv::FONT_HERSHEY_PLAIN, 1, cvColorDarkRed, 1);
            imshow(windowName, imgScaled);
            do {
                key = cv::waitKey(0);
            } while (allowedKeysInRemoveMode.find(key) == allowedKeysInRemoveMode.cend());
            if ('d' == key) {
                // save backup
                if (!ifFileExists(pathToTxtBackup))
                    saveToFile(pathToTxtBackup, to_string(dets));
                else
                    LOG(INFO) << "backup for " << cr.filename << " already exist in " << backupFolderPath << ", dont overwrite.";
                LOG(INFO) << "removing detection #" << foundDetIndex << " and saving the remaining "
                    << (dets.size()-1) << " dets to " << detsPath;
                // delete this detection and overwrite original file
                dets.erase(dets.begin() + foundDetIndex);
                saveToFile(detsPath, to_string(dets));
            }
        }
        if (27 == key) {
            return;
        } else if ('s' == key) {
            LOG(INFO) << "switching toAdd/toRemove mode. Status: "
                << (indexOfToAdd+1) << "/" << numToAdd << " to review for adding,"
                << (indexOfToRemove+1) << "/" << numToRemove << " for removal.";
            showingToAdd = !showingToAdd;
        }
    }
}
