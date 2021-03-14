#ifndef MAP_H
#define MAP_H

#include <opencv2/core/core.hpp>

#include <set>

namespace WeiSLAM{
    using namespace std;
    class Map{
        public:
            Map();

            //static features and depths detected in image plane
            vector<vector<cv::KeyPoint>> vpFeatSta;
            vector<vector<float>> vfDepSta;
            vector<vector<cv::Mat>> vp3DPointSta;

            //index of temporal matching
            vector<vector<int>> vnAssoSta;

            //feature tracklet pair.first = frameID; pair.second = featureID
            vector<vector<pair<int, int>>> TrackletSta;

            //dynamic feature correspondences and depths detected in image plane
            vector<vector<cv::KeyPoint>> vpFeatDyn;
            vector<vector<float>> vfDepDyn;
            vector<vector<cv::Mat>> vp3DPointDyn;
            //index of temporal matching
            vector<vector<int>> vnAssoDyn;
            //label indicating which objec the feature belongs to
            vector<vector<int>> vnFeatLabel;
            //feature tracklets pair.first = frameID; pair.second = featureID
            vector<vector<pair<int, int>>> TrackletDyn;
            vector<int> nObjID;

            //absolute camera pose of each frame, starting from 1st frame
            vector<cv::Mat> vmCameraPose;
            vector<cv::Mat> vmCameraPose_RF;    //refine result
            vector<cv::Mat> vmCameraPose_GT;    //ground truth result

            //rigid motion of camera and dynamic points
            vector<vector<cv::Mat>> vmRigidCentre;      //ground truth object center
            vector<vector<cv::Mat>> vmRigidMotion;     
            vector<vector<cv::Mat>> vmObjPosePre;       
            vector<vector<cv::Mat>> vmRigidMotion_RF;   //refine result
            vector<vector<cv::Mat>> vmRigidMotion_GT;   //ground truth result
            vector<vector<float>> vfAllSpeed_GT;        //camera and object speeds
    
            //rigid motion label in each frame
            //0 stands for camera motion; 1, ...., l stands for rigid motion.
            vector<vector<int>> vnRMLabel;      //tracking label;
            vector<vector<int>> vnSMLabel;      //semantic label;
            vector<vector<int>> vnSMLabelGT;
            //object status
            vector<vector<bool>> vbObjStat;
            //object tracking times
            vector<vector<int>> vnObjTraTime;
            vector<int> nObjTraCount;
            vector<int> nObjTraCountGT;
            vector<int> nObjTraSemLab;

            //time analysis
            vector<float> fLBA_time;
            //(0) frame updating (1) camera estimation (2)object tracking, (3)object estimation (4)map updating
            vector<vector<float>> vfAll_time;


        protected:
            long unsigned int mnMaxKFid;
            //index related to a big change in the map
            int mnBigChangeIdx;    
    };

}
#endif