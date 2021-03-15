#ifndef SYSTEM_H
#define SYSYEM_H

#include <string>
#include <thread>
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

        private:
            eSensor mSensor;

            Map * map;
            Tracking * tracker;
    };
}

#endif