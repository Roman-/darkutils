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

// returns index of next ComparisonResult to show - "to add" or "to remove"
// returns -1 if there are no more cmp results to add
int nextCmpToShow(const ComparisonResults& cmpResults, bool toAdd) {
    int index = -1;
    for (size_t i = 0; i < cmpResults.size(); ++i) {
        auto& r = cmpResults[i];
        if (r.treated) {
            continue;
        } else if (index < 0) {
            index = i;
        } else {
            bool isBetter = (toAdd)
                    ? cmpResults[i].prob > cmpResults[index].prob
                    : cmpResults[i].bbox.area() > cmpResults[index].bbox.area();
            index = isBetter ? i : index;
        }
    }
    return index;
};

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

    // TODO stopped here: update .duv and save it (overwrite) as you mark images. Backup on creation.

    // load
    std::string workPath = extractFileLocationFromFullPath(pathToDuv);
    std::vector<ComparisonResult> cmpResults = comparisonResultsFromFile(pathToDuv);
    // to_string(cmpResults) generates thing to save;
    std::vector<std::string> names = getFileContentsAsStringVector(pathToNames);

    // Before we start to cure, backup original .duv file in backups folder
    if (backupFolderCreated) {
        std::string backupDuvFilename = splitString(extractFilenameFromFullPath(pathToDuv), '.')[0]
                + "_backup" + std::to_string(currentTimestamp())
                + ".duv.tsv";
        bool savedDuv = saveToFile(backupFolderPath + backupDuvFilename, to_string(cmpResults));
        LOG_IF(savedDuv, INFO) << "saved backup " << backupDuvFilename;
        LOG_IF(!savedDuv, ERROR) << "failed to save backup " << backupDuvFilename;
    }

    // count toAdd and toRemove indeces; operate with indeces
    size_t numToAdd{0}, numToRemove{0};
    for (auto& r: cmpResults) {
        if (r.isToAdd())
            ++numToAdd;
        else if (r.isToRemove())
            ++numToRemove;
    }

    LOG(INFO) << "loaded " << numToAdd << " marks to add and " << numToRemove << " marks to remove";

    // show things to add interactively
    bool showingToAdd = true;
    static const std::set<char> allowedKeysInAddMode =    {'y', 'n', char(27), 's'}; // accept, no (dont accept), exit, switch
    static const std::set<char> allowedKeysInRemoveMode = {'d', 'k', char(27), 's'}; // delete, keep (dont delete), exit, switch
    int key; // key pressed by user
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, kWindowWidth, kWindowHeight);
    while (true) {
        int index = nextCmpToShow(cmpResults, showingToAdd);
        if (index < 0) {
            int otherIndex = nextCmpToShow(cmpResults, !showingToAdd);
            if (otherIndex < 0) {
                LOG(INFO) << "Cure procedure finished";
                break;
            } else {
                LOG(WARNING) << "No more marks to " << (showingToAdd ? "add":"remove") << ". Switching mode";
                showingToAdd = !showingToAdd;
                continue;
            }
        }

        ComparisonResult& cr = cmpResults[index];
        auto imgPath = workPath + cr.filename + ".jpg";
        auto detsPath = workPath + cr.filename + ".txt";
        LOG(INFO) << "Next to" << (showingToAdd?"add":"remove") << " is #" << index << ": " << cr.toString();
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
            // the line from top-left corner helps to quickly identify the bbox
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
                LOG(INFO) << "appending mark " << cr.toLoadedDet().toHumanString() << " to " << detsPath;
                // save backup
                if (!ifFileExists(pathToTxtBackup))
                    saveToFile(pathToTxtBackup, to_string(dets));
                else
                    LOG(INFO) << "backup for " << cr.filename << " already exist in " << backupFolderPath << ", dont overwrite.";
                // add this detection and overwrite original file
                dets.push_back(cr.toLoadedDet());
                saveToFile(detsPath, to_string(dets));

                // mark detection as treated
                cr.treated = true;
                cr.iou = 1; // maked = detected -> 100% match
                saveToFile(pathToDuv, to_string(cmpResults));
            } else if ('n' == key) {
                // mark detection as treated (ignored)
                LOG(INFO) << "mark ComparisonResult as treated and save .duv";
                cr.treated = true;
                saveToFile(pathToDuv, to_string(cmpResults));
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

                // also eliminate from .duv.tsv
                cmpResults.erase(cmpResults.begin() + index);
                saveToFile(pathToDuv, to_string(cmpResults));
            } else if ('k' == key) {
                // mark as treated
                LOG(INFO) << "mark ComparisonResult as treated and save .duv";
                cr.treated = true;
                saveToFile(pathToDuv, to_string(cmpResults));
            }
        }
        if (27 == key) {
            return;
        } else if ('s' == key) {
            LOG(INFO) << "switching toAdd/toRemove mode."; // TODO stats?
            showingToAdd = !showingToAdd;
        }
    }
}
