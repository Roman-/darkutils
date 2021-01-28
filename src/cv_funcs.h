#ifndef CV_FUNCS_H
#define CV_FUNCS_H

#include <DarkHelp.hpp>
#include <opencv2/opencv.hpp>

void customAnnotateImage(cv::Mat img);

// absolute bbox
void drawBbox(cv::Mat& img, const cv::Rect& bbox, const cv::Scalar& color, int width = 1);
// relative bbox
void drawBbox(cv::Mat& img, const cv::Rect2f& bbox, const cv::Scalar& color, int width = 1);
// absolute bbox & cross inside
void drawBboxCrossed(cv::Mat& img, const cv::Rect& bbox, const cv::Scalar& color, int rectWidth = 1, int crossWidth = 1);
// relative bbox & cross inside
void drawBboxCrossed(cv::Mat& img, const cv::Rect2f& bbox, const cv::Scalar& color, int rectWidth = 1, int crossWidth = 1);

namespace cvColors {
// returns color by class index (0 is always pink, 1 is greenish etc.)
cv::Scalar colorByClass(int classId);

static const cv::Scalar cvColorWhite =  CV_RGB(255, 255, 255);
static const cv::Scalar cvColorBlack =  CV_RGB(0,   0  , 0);
static const cv::Scalar cvColorRed =    CV_RGB(255, 0  , 0);
static const cv::Scalar cvColorDarkRed= CV_RGB(128, 0  , 0);
static const cv::Scalar cvColorOrange = CV_RGB(255, 69 , 0);
static const cv::Scalar cvColorYellow = CV_RGB(255, 255, 0);
static const cv::Scalar cvColorGreen =  CV_RGB(0  , 255, 0);
} // namespace svColors

#endif // CV_FUNCS_H
