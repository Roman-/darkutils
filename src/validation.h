#ifndef VALIDATION_H
#define VALIDATION_H

#include <string>

// checks all dataset images with trained model, output info about detections and IoUs to file
// pathToDataset - path to .jpg and .txt files.
// param outputFile - /path/to/output.duv - path to darkUtilsValidation-format file
// .duv format: one file for all images&detections, each detection on separate line, sorted by files. Each line:
// class x y w h percent IoU image name with spaces.jpg
void validateDataset(std::string pathToDataset, const std::string& configFile, const std::string& weightsFile,
            const std::string& namesFile, const std::string outputFile);


#endif // VALIDATION_H
