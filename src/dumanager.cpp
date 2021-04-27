#include <dumanager.h>
#include <DarkHelp.hpp>
#include <easylogging++.h>
#include "cv_funcs.h"
#include "helpers.h"

using namespace std;
using namespace cv;

constexpr bool kDrawNames = false;
constexpr bool kDrawPercentage = true;

void configureDarkHelp(DarkHelp& dh) {
    dh.threshold                      = 0.35;
    dh.include_all_names              = false;
    dh.names_include_percentage       = true;
    dh.annotation_include_duration    = false;
    dh.annotation_include_timestamp   = false;
    dh.sort_predictions               = DarkHelp::ESort::kAscending;
}

void markVid(const std::string& configFile, const std::string& weightsFile,
            const std::string& namesFile, const std::string& inputFile) {
    cv::VideoCapture cap(inputFile);
    LOG_IF(!cap.isOpened(), FATAL) << "cant open video " << inputFile;
    float fps = cap.get(CAP_PROP_FPS);
    cv::Size vidSize(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
    const int totalFrames = cap.get(CAP_PROP_FRAME_COUNT);
    LOG(INFO) << "Opened video, " << totalFrames << " total frames, " << fps << " fps, "
        << vidSize.width << "x" << vidSize.height;
    constexpr const char* outFilename = "darkutils_out.mp4";
    auto names = getFileContentsAsStringVector(namesFile);

    DarkHelp darkhelp(configFile, weightsFile, namesFile);
    configureDarkHelp(darkhelp);

    cv::VideoWriter videoWriter(outFilename, cv::VideoWriter::fourcc('M','J','P','G'), fps, vidSize);
    int frameCount = 0;
    // video
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        const auto results = darkhelp.predict(frame);
        LOG(INFO) << (++frameCount) << "/" << totalFrames << ": " << results;

        // cv::Mat output = darkhelp.annotate();
        annotateCustom(frame, results, names, kDrawNames, kDrawPercentage);
        videoWriter << frame;
    }
    cap.release();
    LOG(INFO) << "Annotated file created: " << outFilename;
}

void markImgs(const std::string& configFile, const std::string& weightsFile,
            const std::string& namesFile, std::string pathToImgs) {
    pathToImgs = addSlash(pathToImgs);
    vector<string> imgFiles = listFilesInDir(pathToImgs);
    imgFiles.erase(
            std::remove_if(imgFiles.begin(), imgFiles.end(), [](const string& fn) {return !strEndsWith(fn, ".jpg");} )
          , imgFiles.end());
    if (imgFiles.empty()) {
        LOG(ERROR) << "no .jpg files found in " << pathToImgs;
    }

    constexpr const char* pathToResults = "prediction_results/";
    bool createdOrExists = createFolderIfDoesntExist(pathToResults);
    LOG_IF(!createdOrExists, FATAL) << "failed to create folder: " << pathToResults;
    auto names = getFileContentsAsStringVector(namesFile);

    DarkHelp darkhelp(configFile, weightsFile, namesFile);
    configureDarkHelp(darkhelp);

    int numImgsSaved = 0, imgIndex = 0;
    for (const auto& fn: imgFiles) {
        auto fullPath = pathToImgs + fn;
        cv::Mat img = imread(fullPath);
        if (nullptr == img.data) {
            LOG(ERROR) << "failed to load image " << fullPath;
            continue;
        }
        const auto results = darkhelp.predict(img);
        annotateCustom(img, results, names, kDrawNames, kDrawPercentage);
        std::string outputImgPath = pathToResults + fn;
        LOG(INFO) << (++imgIndex) << "/" << imgFiles.size() << " " << fn << ": " << results;
        bool saved = imwrite(outputImgPath, img);

        if (saved)
            ++numImgsSaved;
        else
            LOG(ERROR) << "failed to save image to";
    }

    LOG(INFO) << "marked and saved " << numImgsSaved << " images to " << pathToResults;
}
