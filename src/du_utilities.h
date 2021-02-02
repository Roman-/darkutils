#ifndef DU_UTILITIES_H
#define DU_UTILITIES_H

#include <vector>
#include <string>

// for each .jpg image in \param pathToDataset, create empty .txt file with the same name
// as if the image was marked as empty.
int createEmptyTxtFiles(const std::string& pathToDataset);

#endif // DU_UTILITIES_H
