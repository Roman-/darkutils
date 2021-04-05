#include "cv_funcs.h"
#include "extract_frames.h"
#include "helpers.h"
#include <easylogging++.h>
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>
#include <algorithm>


using namespace cv;

void extractFrames(const std::string& pathToVids, double fps, float similarityThresh) {
    constexpr const char* outputDirPath = "extracted_frames/";
    // get videos
    std::vector<std::string> filesList = listFilesInDir(pathToVids);
    static const std::set<std::string> extensions = {"mp4", "avi", "mov", "mpg", "mpeg", "m4v"};
    const int initialFilesCount = filesList.size();
    filesList.erase(filesList.begin(), std::find_if(filesList.begin(), filesList.end(),
                        [&](const std::string& s) {return extensions.end() == extensions.find(s);}));
    LOG_IF(filesList.size() != initialFilesCount, WARNING) << "skipping " << (initialFilesCount - filesList.size())
                                                     << " non-video files in " << pathToVids << ".";
    bool succeedWithFolder = createFolderIfDoesntExist(outputDirPath);
    LOG_IF(!succeedWithFolder, FATAL) << "failed to create folder " << succeedWithFolder;
    cv::Mat prevFrame; // for similarity check (similarity is checked between videos, too)

    for (size_t vidNumber = 0; vidNumber < filesList.size(); ++vidNumber) {
        const std::string& vidFileName = filesList[vidNumber];
        std::string filePath = pathToVids + "/" + vidFileName;
        VideoCapture cap(filePath);
        double captureFps = cap.get(CAP_PROP_FPS);
        int totalFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
        if (!cap.isOpened()) {
            LOG(ERROR) << "can not open stream: " << filePath;
            continue;
        }
        const int framesToSkip = int(captureFps/fps) - 1;
        LOG(INFO) << (vidNumber+1) << "/" << filesList.size() << " extracting frames from "
                  << vidFileName << " (" << captureFps << " fps, saving every " << (framesToSkip+1) << "th frame)";
        int numFramesSaved = 0;
        for (size_t frameIndex = 0; frameIndex < totalFrames && cap.isOpened(); ++frameIndex) {
            cv::Mat m;
            cap >> m;
            if (nullptr == m.data)
                break;
            std::string outFramePath = std::string(outputDirPath)
                    + removeAllChars(vidFileName, '.') + "_fr" + leadingZeros(frameIndex, 4) + ".jpg";
            bool areSimilar = false;
            if (!almostEqual(0, similarityThresh)) {
                areSimilar = (prevFrame.size() == m.size()
                             && imgDiff(prevFrame, m) < similarityThresh);
            }
            if (!areSimilar) {
                bool saved = imwrite(outFramePath, m);
                LOG_IF(!saved, ERROR) << "failed to save image to " << outFramePath;
                numFramesSaved += int(saved);
            }
            prevFrame = m;
            for (size_t fs = 0; fs < framesToSkip && cap.isOpened(); ++fs) {
                cap.grab();
            }
        }
        LOG(INFO) << "Saved " << numFramesSaved << " frames from " << vidFileName;
    }
    LOG(INFO) << "Finished. Frames saved to " << outputDirPath;
}
