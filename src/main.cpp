// std
#include <iostream>
#include <string>
#include <map>

// 3rd-party
#include <easylogging++.h>
#include <ini_parser.h>
#include <DarkHelp.hpp>

// dark utils project
#include "helpers.h"
#include "dumanager.h"
#include "du_tests.h"
#include "validation.h"
#include "cure.h"
#include "cv_funcs.h"
#include "extract_frames.h"
#include "du_utilities.h"

INITIALIZE_EASYLOGGINGPP

using std::string;
using std::cout;
using std::cerr;
using std::endl;

static int showUsage(std::string name) {
    cerr << "Usage: " << endl
           //        0          1        2           3           4          5           6
         << "\t" << name << " markvid yoloCfgFile weightsFile namesFile inputVideo" << endl
         << "\t" << name << " markimgs yoloCfgFile weightsFile namesFile /path/to/imgs/" << endl
         << "\t" << name << " extractframes /path/to/videos/ fps similarityThresh=0" << endl
         << "\t" << name << " addemptytxt /path/to/dataset/" << endl
         << "\t" << name << " test /path/to/darkutils/data/tests/"  << endl
         << "\t" << name << " validate yoloCfgFile weightsFile namesFile /path/to/dataset/ outputFile.duv"  << endl
         << "\t" << name << " cure /path/to/dataset/ duvFile namesFile" << endl;
    return -1;
}
int main(int argc, char **argv) {
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%level %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);

    if (argc < 2)
        return showUsage(argv[0]);

    std::string command(argv[1]);

    // check number of args
    std::map<std::string, int> commandNumArgs = {
        {"test", 3},
        {"markvid", 6},
        {"markimgs", 6},
        {"addemptytxt", 3},
        {"extractframes", 5},
        {"validate", 7},
        {"cure", 5},
        // TODO introduce "sanitycheck" command: checks that each img has .txt, no extra files, no broken images
    };
    if (commandNumArgs.end() == commandNumArgs.find(command) || argc != commandNumArgs.at(command))
        return showUsage(argv[0]);

    if (command == "markvid") {
        markVid(argv[2], argv[3], argv[4], argv[5]);
        return 0;
    }

    if (command == "markimgs") {
        markImgs(argv[2], argv[3], argv[4], argv[5]);
        return 0;
    }

    if (command == "addemptytxt") {
        return createEmptyTxtFiles(argv[2]);
    }

    if (command == "extractframes") {
        double fps = std::stod(argv[3]);
        float similarityThresh = std::stof(argv[4]);
        extractFrames(argv[2], fps, similarityThresh);
        return 0;
    }

    if (command == "test")
        return runAllTests(argv[2]);

    if (command == "validate") {
        validateDataset(argv[5], argv[2], argv[3], argv[4], argv[6]);
        return 0;
    }

    if (command == "cure") {
        cureDataset(argv[2], argv[3], argv[4]);
        return 0;
    }

    return -1;
}

