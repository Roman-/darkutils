#include "validation.h"
#include "du_common.h"
#include "helpers.h"
#include "easylogging++.h"
#include <DarkHelp.hpp>
#include <string>
using namespace std;
using namespace cv;

// forward decl
ComparisonResults comparePredictions(const DarkHelp::PredictionResults& predictions,
                                     const vector<LoadedDetection>& groundTruthDets);

void validateDataset(std::string pathToDataset, const std::string& configFile, const std::string& weightsFile,
            const std::string& namesFile, const std::string outputFile) {

    pathToDataset += (pathToDataset.back() == '/') ? "" : "/";
    std::vector<std::string> filenames = loadTrainImageFilenames(pathToDataset); // without .ext
    LOG_IF(filenames.empty(), FATAL) << "Train images not found in dir";
    bool writable = saveToFile(outputFile, "", false);
    LOG_IF(!writable, FATAL) << "Can\'t write to file " << outputFile;

    DarkHelp darkhelp(configFile, weightsFile, namesFile);

    darkhelp.threshold                      = 0.15;
    darkhelp.include_all_names              = false;
    darkhelp.names_include_percentage       = true;
    darkhelp.annotation_include_duration    = false;
    darkhelp.annotation_include_timestamp   = false;
    darkhelp.sort_predictions               = DarkHelp::ESort::kAscending;

    std::string result;
    // .duv format: one file for all images&detections, each detection on separate line, sorted by files. Each line:
    // class x y w h percent IoU image name with spaces.jpg
    for (int filesIndex = 0; filesIndex < filenames.size(); ++filesIndex) {
        const string filename = filenames[filesIndex];
        string pathToImage = pathToDataset + filename + ".jpg";
        cv::Mat img = imread(pathToImage);
        if (nullptr == img.data || img.cols < 1 || img.rows < 1) {
            LOG(ERROR) << "failed to load image: " << pathToImage;
            continue;
        }
        vector<LoadedDetection> groundTruthDets = loadDetsFromFile(pathToDataset + filename + ".txt");
        LOG(INFO) << (filesIndex+1) << "/" << filenames.size() << " " << filename
            << ".jpg: " << groundTruthDets.size() << " detections";

        const DarkHelp::PredictionResults predictions = darkhelp.predict(img);
        ComparisonResults results = comparePredictions(predictions, groundTruthDets);
        saveToFile(outputFile, to_string(results), true); // append

    }
    LOG(INFO) << "ValidateDataset finished. Results saved to " << outputFile;
}

ComparisonResults comparePredictions(const DarkHelp::PredictionResults& predictions,
                                     const vector<LoadedDetection>& groundTruthDets) {
    constexpr float thresh = 0.35; // TODO different name
    ComparisonResults results;
    // identifying false negatives: loaded ground truth detections that have no intercections with dets
    for (const LoadedDetection& loadedDet: groundTruthDets) {
        // find detections of the same class, see what intersects.
        float classProbability = -1;
        for (int i = 0; i < predictions.size(); ++i) {
            auto predictionBbox = relativeBbox(predictions[i]);
            float p = getProb(predictions[i], loadedDet.classId);
            float iou = strongIntersection(predictionBbox, loadedDet.bbox);
            if (p > thresh && iou > kStrongIntersectionThresh) {
                classProbability = p;
                results.push_back({loadedDet.classId, predictionBbox, p, iou});
                break;
            }
        }
        // no matching detection for this "ground truth" loadedDet
        if (classProbability < 0) {
            // loadedDet hasn't been detected
            results.push_back({loadedDet.classId, loadedDet.bbox, -1, 0});
        }
    }
    // then look for false positives: find detections from *dets that do not intersect with ground truth
    for (int i = 0; i < predictions.size(); ++i) {
        for (auto it = predictions[i].all_probabilities.cbegin(); it != predictions[i].all_probabilities.cend(); ++it) {
            int classId = it->first;
            float detectionProb = it->second;
            auto predictionBbox = relativeBbox(predictions[i]);
            if (detectionProb > thresh) {
                // see if dets[i] intersect with some of ground truth detections. If not, this is a false positive
                bool hasMatchingGtDet = false;
                for (const LoadedDetection& loadedDet: groundTruthDets) {
                    if (classId == loadedDet.classId && strongIntersection(predictionBbox, loadedDet.bbox)) {
                        hasMatchingGtDet = true; // nothing to add to results, this det was added on the last step
                        break;
                    }
                }
                if (!hasMatchingGtDet) {
                    results.push_back({classId, predictionBbox, detectionProb, 0});
                }
            }
        } // classId
    } // i = index of detection
    return results;
}
