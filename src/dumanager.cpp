#include <dumanager.h>
#include <DarkHelp.hpp>
#include <easylogging++.h>
#include "cv_funcs.h"
#include "helpers.h"

using namespace std;
using namespace cv;

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
    float fps = cap.get(CV_CAP_PROP_FPS);
    cv::Size vidSize(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    LOG(INFO) << "Opened video, " << cap.get(CV_CAP_PROP_FRAME_COUNT) << " total frames, " << fps << " fps, "
        << vidSize.width << "x" << vidSize.height;
    constexpr const char* outFilename = "darkutils_out.mp4";

    DarkHelp darkhelp(configFile, weightsFile, namesFile);
    configureDarkHelp(darkhelp);

    cv::VideoWriter videoWriter(outFilename, cv::VideoWriter::fourcc('M','J','P','G'), fps, vidSize);
    // video
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        const auto result = darkhelp.predict(frame);
        std::cout << result << std::endl;

        cv::Mat output = darkhelp.annotate();
        videoWriter << output;
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

    DarkHelp darkhelp(configFile, weightsFile, namesFile);
    configureDarkHelp(darkhelp);

    int numImgsSaved = 0;
    for (const auto& fn: imgFiles) {
        auto fullPath = pathToImgs + fn;
        cv::Mat img = imread(fullPath);
        if (nullptr == img.data) {
            LOG(ERROR) << "failed to load image " << fullPath;
            continue;
        }
        const auto result = darkhelp.predict(img);
        cv::Mat outputImg = darkhelp.annotate();
        std::string outputImgPath = pathToResults + fn;
        LOG(INFO) << fn << ": " << result;
        bool saved = imwrite(outputImgPath, outputImg);

        if (saved)
            ++numImgsSaved;
        else
            LOG(ERROR) << "failed to save image to";
    }

    LOG(INFO) << "marked and saved " << numImgsSaved << " images to " << pathToResults;
}
