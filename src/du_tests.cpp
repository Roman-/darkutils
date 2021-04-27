#include <du_tests.h>
#include "du_common.h"
#include "helpers.h"
#include "easylogging++.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct IouTest {
    cv::Rect2f r1;
    cv::Rect2f r2;
    float iou;
};

static const std::string pathToTestDuv = "/invalid_masks/result.duv.tsv";

int runIouTest(const std::string&) {
    constexpr float deltaIou = 1e-6;
    std::vector<IouTest> iouTests = {
        IouTest{cv::Rect2f(0,0,4,4), cv::Rect2f(1,1,4,4), 9./23.},
        IouTest{cv::Rect2f(1,1,3,3), cv::Rect2f(2,2,1,1), 1./9.},
        IouTest{cv::Rect2f(.1,.1,.3,.3), cv::Rect2f(2,2,1,1), 0},
    };
    for (const auto& t: iouTests) {
        float resIou = intersectionOverUnion(t.r1, t.r2);
        if (fabsf(resIou - t.iou) > deltaIou) {
            LOG(ERROR) << "IoU test failed: rects: " << to_human_string(t.r1) << " and " << to_human_string(t.r2)
                << " - expected iou = " << t.iou << ", got " << resIou;
            return -1;
        }
    }
    return 0;
}

int runDsLoadingTests(const std::string& pathToTrainImgs) {
    static const std::string pathToDataset = pathToTrainImgs + "/masks_files";
    auto files = loadTrainImageFilenames(pathToDataset, true); // 1,2,3,4
    if (files.size() != 4) {
        LOG(ERROR) << "runDsLoadingTests failed: number of image files here should be 4 but equals " << files.size();
        return 1;
    }
    static const std::vector<std::string> baseFileNames = {"1", "2", "3", "4"};
    if (files != baseFileNames) {
        LOG(ERROR) << "runDsLoadingTests failed: files != baseFileNames.";
        return 1;
    }
    // empty files
    auto filesWithoutTxt = loadTrainImageFilenames(pathToDataset, false);
    if (!filesWithoutTxt.empty()) {
        LOG(ERROR) << "runDsLoadingTests failed: filesWithoutTxt not empty in " << pathToDataset;
        return 1;
    }
    return 0;
}

int runCmpResultsFromStringTests(const std::string&) {
    const std::string str = "file name with spaces	2	0.5	0.5	0.5	0.5	0.5	0.5	f";
    auto r = ComparisonResult::fromString(str);
    if (r.filename != "file name with spaces") {
        LOG(ERROR) << "runCmpResultsFromStringTests: filename is " << r.filename;
        return -1;
    }
    if (r.classId != 2) {
        LOG(ERROR) << "runCmpResultsFromStringTests: classId is " << r.classId;
        return -1;
    }
    if (!almostEqual(r.bbox.width, 0.5) || !almostEqual(r.bbox.height, 0.5)
            || !almostEqual(r.bbox.x, 0.25) || !almostEqual(r.bbox.y, 0.25)
            || !almostEqual(r.iou, 0.5) || !almostEqual(r.prob, 0.5)) {
        LOG(ERROR) << "runCmpResultsFromStringTests: parsed wrong: is " << r.toString() << "\n but shold be this:\n" << str;
        return -1;
    }
    return 0;
}

int runCmpResultsFromFileTests(const std::string& pathToTestFolder) {
    const std::string pathToDuv = pathToTestFolder + pathToTestDuv;
    auto results = comparisonResultsFromFile(pathToDuv);
    if (results.size() != 12) {
        LOG(ERROR) << "runCmpResultsFromFileTests: expected 12 results in " << pathToDuv << ", got " << results.size();
        return -1;
    }
    auto it = std::find_if(results.begin(), results.end(), [](const ComparisonResult& r) {return !r.isValid();});
    if (it != results.end()) {
        LOG(ERROR) << "runCmpResultsFromFileTests: have invalid result: " << it->toString();
        return -1;
    }

    it = std::find_if(results.begin(), results.end(),
            [](const ComparisonResult& r) {return r.bbox.area() > 1 || almostEqual(r.bbox.area(), 0);});
    if (it != results.end()) {
        LOG(ERROR) << "runCmpResultsFromFileTests: bad bbox: " << it->toString();
        return -1;
    }

    if (results[0].filename != "correct") {
        LOG(ERROR) << "runCmpResultsFromFileTests: result[0] differs from expected: " << results[0].toString();
        return -1;
    }
    return 0;
}

int runDetectionLoadingTest(const std::string& testsDir) {
    std::map<std::string, int> numDetsInFiles = {
        {"1", 0},
        {"2", 1},
        {"3", 4},
        {"4", 1}
    };
    // chicking amount of loaded detections
    for (const auto& p: numDetsInFiles) {
        auto filePath = testsDir + "/masks_files/" + p.first + ".txt";
        int expectedDets = p.second;
        std::vector<LoadedDetection> dets = loadedDetectionsFromFile(filePath);
        if (dets.size() != expectedDets) {
            LOG(ERROR) << "in file " << filePath << ", amount of parsed detections = " << dets.size()
                       << " but expected to be " << expectedDets;
            return -1;
        }
        for (const auto& d: dets) {
            if (!d.isValid()) {
                LOG(ERROR) << "detection loaded from " << filePath << " turned out to be invalid: " << d.toHumanString();
                return -1;
            }
        }
    }

    return 0;
}

int runTxtLoadingTest(const std::string& testsDir) {
    auto txtPath = testsDir + "masks_train.txt";
    auto imgsPaths = loadPathsToImages(txtPath);
    if (imgsPaths.size() != 4) {
        LOG(ERROR) << "Cant load 4 image paths from " << txtPath;
        return 1;
    }
    // chicking amount of loaded detections
    for (const auto& p: imgsPaths) {
        std::vector<std::string> filesToAccess{p+".txt", p+".jpg"};
        for (const auto& f: filesToAccess) {
            if (!ifFileExists(f)) {
                LOG(ERROR) << "Can\'t access to file " << f;
                return 1;
            }
        }
    }

    return 0;
}

int runAllTests(const std::string& testsDataDir) {
    static const std::vector<std::function<int(const std::string&)>> funcsToTest = {
          &runIouTest
        , &runDetectionLoadingTest
        , &runTxtLoadingTest
        , &runCmpResultsFromStringTests
        , &runCmpResultsFromFileTests
        , &runDsLoadingTests
    };

    // check tests dir
    auto dirs = listFilesInDir(testsDataDir);
    if (dirs.end() == std::find(dirs.begin(), dirs.end(), "masks_files")) {
        LOG(FATAL) << "directory " << testsDataDir << " does not contain masks_files dir";
    }
    LOG(INFO) << "Running " << funcsToTest.size() << " tests...";

    for (const auto& f: funcsToTest) {
        if (f(testsDataDir) != 0) {
            return -1;
        }
    }

    LOG(INFO) << "All tests passed successfully";
    return 0;
}
