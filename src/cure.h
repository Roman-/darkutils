#ifndef CURE_H
#define CURE_H

#include <string>

// "cure" dataset by interactively showing apparently wrong marks from .duv file
// @param pathToDuv path to results.duv.tsv, with image paths being either absolute or relative to .duv.tsv
void cureDataset(const std::string& pathToDuv
               , const std::string& pathToNames);


#endif // CURE_H
