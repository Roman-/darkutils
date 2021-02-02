#include <fstream>
#include "du_utilities.h"
#include "easylogging++.h"
#include "helpers.h"
#include "du_common.h"

int createEmptyTxtFiles(const std::string& pathToDataset) {
    auto path = addSlash(pathToDataset);
    auto filenames = loadTrainImageFilenames(path, false);
    LOG(INFO) << "found " << filenames.size() << " unlabelled files in " << path;
    if (filenames.empty())
        return 0;
    int numFilesCreated = 0;
    for (const auto& f: filenames) {
        std::string emptyFilePath = path + f + ".txt";
        std::ofstream output(emptyFilePath);
        if (output.is_open()) {
            ++numFilesCreated;
        } else {
            LOG_N_TIMES(1, ERROR) << "can not create file " << emptyFilePath << ", omitting next error messages";
        }
    }
    LOG_IF(filenames.size() == numFilesCreated, INFO) << "Successfully created " << numFilesCreated << " .txt files";
    LOG_IF(filenames.size() != numFilesCreated, WARNING) << "Created only " << numFilesCreated << " .txt files in " << path;

    return (filenames.size() == numFilesCreated) ? 0 : -1;
}
