#include <cv_funcs.h>
#include <DarkHelp.hpp>
#include <easylogging++.h>

void drawBbox(cv::Mat& img, const cv::Rect& bbox, const cv::Scalar& color, int width) {
    if (nullptr == img.data)
        return;
    cv::rectangle(img, bbox, color, width, cv::LINE_8, 0);
}

void drawBbox(cv::Mat& img, const cv::Rect2f& bbox, const cv::Scalar& color, int width) {
    if (nullptr == img.data)
        return;
    cv::Rect absBbox(bbox.x * img.cols, bbox.y * img.rows, bbox.width * img.cols, bbox.height * img.rows);
    cv::rectangle(img, absBbox, color, width, cv::LINE_8, 0);
}
