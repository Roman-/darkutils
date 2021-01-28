#ifndef CURE_H
#define CURE_H

#include <string>

// "cure" dataset by interactively showing apparently wrong marks from .duv file
void cureDataset(const std::string& pathToTrainData
               , const std::string& pathToDuv
               , const std::string& pathToNames);


#endif // CURE_H
