#include <dumanager.h>
#include <DarkHelp.hpp>
#include <easylogging++.h>

void dutest(const std::string configFile, const std::string weightsFile,
            const std::string namesFile, const std::string inputFile) {

    cv::VideoCapture cap(inputFile);
    LOG_IF(!cap.isOpened(), FATAL) << "cant open video " << inputFile;
    float fps = cap.get(CV_CAP_PROP_FPS);
    cv::Size vidSize(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    LOG(INFO) << "Opened video, " << cap.get(CV_CAP_PROP_FRAME_COUNT) << " total frames, " << fps << " fps, "
        << vidSize.width << "x" << vidSize.height;
    constexpr const char* outFilename = "darkutils_out.mp4";

    DarkHelp darkhelp(configFile, weightsFile, namesFile);

    darkhelp.threshold                      = 0.35;
    darkhelp.include_all_names              = false;
    darkhelp.names_include_percentage       = true;
    darkhelp.annotation_include_duration    = true;
    darkhelp.annotation_include_timestamp   = false;
    darkhelp.sort_predictions               = DarkHelp::ESort::kAscending;

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
