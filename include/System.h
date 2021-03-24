#ifndef SYSTEM_H
#define SYSTEM_H


#include <string>
#include <thread>
#include <vector>
#include <opencv2/core/core.hpp>

#include "Tracking.h"
#include "Map.h"

namespace WeiSLAM{
    using namespace std;

    class Tracking;
    class Map;

    class System{
        public:
            enum eSensor{
                MONOCULAR = 0,
                STEREO = 1,
                RGBD = 2
            };

            System(const string &strSettingsFile, const eSensor sensor);

            cv::Mat TrackRGBD(const cv::Mat &im, cv::Mat &depthmap, const cv::Mat &flowmap, const cv::Mat &masksem,
                              const cv::Mat &mTcw_gt, const vector<vector<float>> &vObjectPose_gt, const double &timestamp,
                              cv::Mat &imTraj, const int &nImage);
            
            void SaveResultsIJRR2021(const string & filename);


            cv::Mat TrackMono(const cv::Mat &im, const cv::Mat &flowmap, const cv::Mat &maskSem, const cv::Mat mTcw_gt,
                              const vector<vector<float>> &vObjectPose_gt, const double &timestamp, cv::Mat &imTraj,
                              const int &nImage);
        private:
            eSensor mSensor;

            Map * map;
            Tracking * tracker;
    };
}

#endif