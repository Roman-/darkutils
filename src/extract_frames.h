#ifndef EXTRACT_FRAMES_H
#define EXTRACT_FRAMES_H

#include <string>

// for each video in pathWithVids, open and extract frames to output folder
// similarityThresh: if >0, consecutive frames will be checked for similarity and too similar frames will not be saved
// Recommended value for similarityThresh = 0.002
void extractFrames(const std::string& pathToVids, double fps, float similarityThresh = 0);

#endif // EXTRACT_FRAMES_H
