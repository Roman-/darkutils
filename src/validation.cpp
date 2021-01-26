#include "validation.h"
#include "du_common.h"
#include "helpers.h"
#include "easylogging++.h"
#include <DarkHelp.hpp>
#include <string>
using namespace std;
using namespace cv;

static constexpr float kValidationThresh = 0.15;

// forward decl
ComparisonResults comparePredictions(cv::Mat img, const DarkHelp::PredictionResults& predictions,
                                     const vector<LoadedDetection>& groundTruthDets, const std::string& filename);

void validateDataset(std::string pathToDataset, const std::string& configFile, const std::string& weightsFile,
            const std::string& namesFile, const std::string outputFile) {

    pathToDataset += (pathToDataset.back() == '/') ? "" : "/";
    std::vector<std::string> filenames = loadTrainImageFilenames(pathToDataset); // without .ext
    LOG_IF(filenames.empty(), FATAL) << "Train images not found in dir";
    bool writable = saveToFile(outputFile, "", false);
    LOG_IF(!writable, FATAL) << "Can\'t write to file " << outputFile;

    DarkHelp darkhelp(configFile, weightsFile, namesFile);

    darkhelp.threshold                      = kValidationThresh;
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
        const DarkHelp::PredictionResults predictions = darkhelp.predict(img);
        LOG(INFO) << (filesIndex+1) << "/" << filenames.size() << " " << filename
                    << ".jpg: " << groundTruthDets.size() << " marks"
                    << (groundTruthDets.size() == predictions.size() ? " and " : " but ")
                    << predictions.size() << " predictions";
        ComparisonResults results = comparePredictions(img, predictions, groundTruthDets, filename);
        saveToFile(outputFile, to_string(results), true); // append

    }
    LOG(INFO) << "ValidateDataset finished. Results saved to " << outputFile;
}

ComparisonResults comparePredictions(cv::Mat img, const DarkHelp::PredictionResults& predictions,
                                     const vector<LoadedDetection>& groundTruthDets, const std::string& filename) {
    constexpr bool verbose = false;
    ComparisonResults results;

    // identifying false negatives: loaded ground truth detections that have no intercections with dets
    for (const LoadedDetection& loadedDet: groundTruthDets) {
        // find detections of the same class, see what intersects.
        float classProbability = -1;
        // iterate through darknet predictions to see if they match ground truth
        for (int i = 0; i < predictions.size(); ++i) {
            auto predictionBbox = relativeBbox(predictions[i]);
            float p = getProb(predictions[i], loadedDet.classId);
            float iou = intersectionOverUnion(predictionBbox, loadedDet.bbox);
            if (p > kValidationThresh && iou > kStrongIntersectionThresh) {
                classProbability = p;
                results.push_back({loadedDet.classId, predictionBbox, p, iou, filename});
                LOG_IF(verbose, INFO) << "detected OK " << predictions[i];
                break;
            }
        }
        // no matching detection for this "ground truth" loadedDet
        if (classProbability < 0) {
            // loadedDet hasn't been detected
            LOG_IF(verbose, WARNING) << "Darknet doesn\'t see this ground truth detection: " << loadedDet.toString();
            results.push_back({loadedDet.classId, loadedDet.bbox, 0, 0, filename});
        }
    }
    // then look for false positives: find detections from *dets that do not intersect with ground truth
    for (int i = 0; i < predictions.size(); ++i) {
        for (auto it = predictions[i].all_probabilities.cbegin(); it != predictions[i].all_probabilities.cend(); ++it) {
            int classId = it->first;
            float detectionProb = it->second;
            auto predictionBbox = relativeBbox(predictions[i]);
            if (detectionProb > kValidationThresh) {
                // see if dets[i] intersect with some of ground truth detections. If not, this is a false positive
                bool hasMatchingGtDet = false;
                float maxClassIou = 0; // maximum iou between darknet prediction and any of ground_truth with this classId
                for (const LoadedDetection& loadedDet: groundTruthDets) {
                    if (classId == loadedDet.classId) {
                        maxClassIou = std::max(maxClassIou, intersectionOverUnion(predictionBbox, loadedDet.bbox));
                        if (maxClassIou >= kStrongIntersectionThresh) {
                            hasMatchingGtDet = true; // nothing to add to results, this det was added on the last step
                            break;
                        }
                    }
                }
                if (!hasMatchingGtDet) {
                    results.push_back({classId, predictionBbox, detectionProb, maxClassIou, filename});
                    LOG_IF(verbose, WARNING) << "detection predicted by darknet " << results.back().toString()
                        << " does not have corresponding groundTruth mark (max iou = " << maxClassIou << ")";
                }
            }
        } // classId
    } // i = index of detection
    return results;
}
