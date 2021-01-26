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

int runIouTest(const std::string&) {
    LOG(INFO) << "Running iou test";
    constexpr float deltaIou = 1e-6;
    std::vector<IouTest> iouTests = {
        IouTest{cv::Rect2f(0,0,4,4), cv::Rect2f(1,1,4,4), 9./23.},
        IouTest{cv::Rect2f(1,1,3,3), cv::Rect2f(2,2,1,1), 1./9.},
        IouTest{cv::Rect2f(.1,.1,.3,.3), cv::Rect2f(2,2,1,1), 0},
    };
    for (const auto& t: iouTests) {
        float resIou = intersectionOverUnion(t.r1, t.r2);
        if (fabsf(resIou - t.iou) > deltaIou) {
            LOG(ERROR) << "IoU test failed: rects: " << to_string(t.r1) << " and " << to_string(t.r2)
                << " - expected iou = " << t.iou << ", got " << resIou;
            return -1;
        }
    }
    return 0;
}

int runDetectionLoadingTest(const std::string& testsDir) {
    std::map<std::string, int> numDetsInFiles = {
        {"1", 0},
        {"2", 1},
        {"3", 4}
    };
    // chicking amount of loaded detections
    for (const auto& p: numDetsInFiles) {
        auto filePath = testsDir + "/masks_files/" + p.first + ".txt";
        int expectedDets = p.second;
        std::vector<LoadedDetection> dets = loadDetsFromFile(filePath);
        if (dets.size() != expectedDets) {
            LOG(ERROR) << "in file " << filePath << ", amount of parsed detections = " << dets.size()
                       << " but expected to be " << expectedDets;
            return -1;
        }
        for (const auto& d: dets) {
            if (!d.isValid()) {
                LOG(ERROR) << "detection loaded from " << filePath << " turned out to be invalid: " << d.toString();
                return -1;
            }
        }
    }

    // TODO more tests
    return 0;
}

int runAllTests(const std::string& testsDataDir) {
    static const std::vector<std::function<int(const std::string&)>> funcsToTest = {
        &runIouTest,
        &runDetectionLoadingTest
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
