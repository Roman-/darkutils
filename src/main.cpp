// std
#include <iostream>
#include <string>
#include <map>

// 3rd-party
#include <easylogging++.h>
#include <ini_parser.h>
#include <DarkHelp.hpp>

// dark utils project
#include <helpers.h>
#include <dumanager.h>
#include <du_tests.h>
#include <validation.h>

INITIALIZE_EASYLOGGINGPP

using std::string;
using std::cout;
using std::cerr;
using std::endl;

static int showUsage(std::string name) {
    cerr << "Usage: " << endl
           //        0          1        2           3           4          5           6
         << "\t" << name << " makevid yoloCfgFile weightsFile namesFile inputVideo" << endl
         << "\t" << name << " test "<<"/path/to/darkutils/data/tests"  << endl
         << "\t" << name << " validate yoloCfgFile weightsFile namesFile /path/to/dataset outputFile.duv"  << endl;
    return -1;
}
int main(int argc, char **argv) {
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level (%fbase:%line) %msg");
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::ToFile, "false"); // < TODO this does not work
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);

    if (argc < 2)
        return showUsage(argv[0]);

    std::string command(argv[1]);

    // check number of args
    std::map<std::string, int> commandNumArgs = {
        {"test", 3},
        {"makevid", 6},
        {"validate", 7},
    };
    if (commandNumArgs.end() == commandNumArgs.find(command) || argc != commandNumArgs.at(command))
        return showUsage(argv[0]);

    if (command == "test")
        return runAllTests(argv[2]);

    if (command == "validate") {
        validateDataset(argv[5], argv[2], argv[3], argv[4], argv[6]);
        return 0;
    }

    std::string cfgFile(argv[2]);
    std::string weightsFile(argv[3]);
    std::string namesFile(argv[4]);
    std::string inputFile(argv[5]);

    dutest(cfgFile, weightsFile, namesFile, inputFile);

    return 0;
}

