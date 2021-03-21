#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Map.h"
#include "Frame.h"
#include "../Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

namespace WeiSLAM{
    using namespace std;

    class Optimizer{
        public:
            int static PoseOptimizationNew(Frame *pCurFrame, Frame *pLastFrame, vector<int> &TemperalMatch);
            int static PoseOptimizationFlow2Cam(Frame *pCurFrame, Frame *pLastFrame, vector<int> &TemperalMatch);
            cv::Mat static PoseOptimizationObjMot(Frame *pCurFrame, Frame *pLastFrame, const vector<int> &OnjId, vector<int> &InlierID);
            cv::Mat static PoseOptimizationFlow2(Frame *pCurFrame, Frame *pLastFrame, const vector<int> &ObjId, vector<int> &InlierID);
            void static FullBatchOptimization(Map* pMap, const cv::Mat Calib_K);
            void static PartialBatchOptimization(Map* pMap, const cv::Mat Calib_k, const int WINDOW_SIZE);
            cv::Mat static Get3DinWorld(const cv::KeyPoint &vKey1, const cv::KeyPoint &vKey2, const cv::Mat &Calib_K, const cv::Mat &CameraPose);
            cv::Mat static Get3DinCamera(const cv::KeyPoint &vKey2, const cv::KeyPoint &vKey2, const cv::Mat &Calib_K);
    };
}

#endif