#ifndef DUMANAGER_H
#define DUMANAGER_H

#include <string>
using std::string;

void markVid(const std::string& configFile, const std::string& weightsFile,
            const std::string& namesFile, const std::string& inputFile);

void markImgs(const std::string& configFile, const std::string& weightsFile,
              const std::string& namesFile, std::string pathToImgs);

#endif // DUMANAGER_H
