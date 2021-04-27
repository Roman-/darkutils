#include <cv_funcs.h>
#include <DarkHelp.hpp>
#include <easylogging++.h>
#include <string>

using namespace cvColors;
using std::to_string;

void drawBbox(cv::Mat& img, const cv::Rect& bbox, const cv::Scalar& color, int width) {
    if (nullptr == img.data)
        return;
    cv::rectangle(img, bbox, color, width, cv::LINE_8, 0);
}

void drawBbox(cv::Mat& img, const cv::Rect2d& bbox, const cv::Scalar& color, int width) {
    cv::Rect absBbox(bbox.x * img.cols, bbox.y * img.rows, bbox.width * img.cols, bbox.height * img.rows);
    drawBbox(img, absBbox, color, width);
}

void drawBboxCrossed(cv::Mat& img, const cv::Rect& bbox, const cv::Scalar& color, int rectThickness, int crossThickness) {
    if (nullptr == img.data)
        return;
    cv::rectangle(img, bbox, color, rectThickness, cv::LINE_8, 0);
    cv::line(img, bbox.tl(), bbox.br(), color, crossThickness, cv::LINE_8, 0);
    cv::line(img, cv::Point(bbox.x, bbox.br().y), cv::Point(bbox.br().x, bbox.y), color, crossThickness, cv::LINE_8, 0);
}

void drawBboxCrossed(cv::Mat& img, const cv::Rect2d& bbox, const cv::Scalar& color, int rectThickness, int crossThickness) {
    cv::Rect absBbox(bbox.x * img.cols, bbox.y * img.rows, bbox.width * img.cols, bbox.height * img.rows);
    drawBboxCrossed(img, absBbox, color, rectThickness, crossThickness);
}

// draw bboxes and percentage
void annotateCustom(cv::Mat& img, const DarkHelp::PredictionResults& results
                    , const std::vector<std::string>& names, bool drawNames, bool drawPercentage) {
    constexpr const int fontFace = cv::FONT_HERSHEY_DUPLEX;
    constexpr const float fontScale = 0.5;
    constexpr const int thickness = 1;
    for (const auto& r: results) {
        auto color = colorByClass(r.best_class);
        drawBbox(img, r.rect, color, 2);
        if (drawNames || drawPercentage) {
            std::string txt = drawNames ? names[r.best_class] : "";
            txt += (drawNames && drawPercentage) ? " " : "";
            txt += drawPercentage ? (to_string(int(r.best_probability * 100)) + "%") : "";
            auto textSize = cv::getTextSize(txt, fontFace, fontScale, thickness, 0);
            cv::Rect textRect(r.rect.x, r.rect.y - textSize.height, textSize.width, textSize.height);
            cv::rectangle(img, textRect, color, cv::FILLED);
            cv::putText(img, txt, r.rect.tl(), fontFace, fontScale, contrastTextColor(color), thickness);
        }
    }
}

namespace cvColors {
cv::Scalar colorByClass(int classId) {
    // a not-so-elegant method but it works
    static std::vector<cv::Scalar> palette = {
        CV_RGB(242, 31, 112), // pinkish
        CV_RGB(59, 242, 31), // greenish
        CV_RGB(242, 172, 31), // orange
        CV_RGB(31, 242, 221), // aqua
        CV_RGB(242, 214, 31), // yellowish
        CV_RGB(31, 56, 242), // blue
        CV_RGB(137, 31, 242) // purple
    };
    while (classId >= palette.size()) {
        int r = int(fabsf(sin(classId)*12345678)) % 256;
        int g = int(fabsf(sin(classId)*1234567)) % 256;
        int b = int(fabsf(sin(classId)*123456)) % 256;
        palette.push_back(CV_RGB(r,g,b));
    }

    return palette[classId];
}

}

float imgDiff(cv::Mat img1, cv::Mat img2) {
    int w = img1.cols;
    int h = img1.rows;
    if (img1.size() != img2.size()) {
        LOG(ERROR) << "image sizes mismatch in imgDiff: img1 " << w << "x" << h << ", img2 " << img2.cols << "x" << img2.rows;
        return 1;
    }
    cv::Mat diffImage;
    cv::absdiff(img1, img2, diffImage);

    float result = 0;

    for(int j=0; j<diffImage.rows; ++j) {
        for(int i=0; i<diffImage.cols; ++i) {
            cv::Vec3b pix = diffImage.at<cv::Vec3b>(j,i);
            result += float(pix[0] + pix[1] + pix[2]) / (255*3);
    }   }
    return result / (diffImage.rows * diffImage.cols);
}
