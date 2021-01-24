// std
#include <iostream>
#include <string>

// 3rd-party
#include <easylogging++.h>
#include <ini_parser.h>
#include <DarkHelp.hpp>

// dark utils project
#include <helpers.h>
#include <dumanager.h>

INITIALIZE_EASYLOGGINGPP

using std::string;

static int showUsage(std::string name) {
    //                         0           1         2          3          4          5
    std::cerr << "Usage: " << name << " command yoloCfgFile weightsFile namesFile inputFile/dir\n"
              << std::endl;
    return -1;
}
int main(int argc, char **argv) {
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level (%fbase:%line) %msg");
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::ToFile, "false");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);

    if (argc != 6)
        return showUsage(argv[0]);

    std::string command(argv[1]);
    std::string cfgFile(argv[2]);
    std::string weightsFile(argv[3]);
    std::string namesFile(argv[4]);
    std::string inputFile(argv[5]);

    dutest(cfgFile, weightsFile, namesFile, inputFile);

    return 0;
}

