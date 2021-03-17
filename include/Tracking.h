#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <Eigen/Core>

#include "Map.h"
#include "Frame.h"
#include "ORBextractor.h"
#include "System.h"

#include <mutex>

namespace WeiSLAM{
    using namespace std;

    class Map;
    class System;
    
    class Tracking{
        struct BirdEyeVizProperties{
            float birdeye_scale_factor_;
            int birdeye_far_plane_;
            int birdeye_left_plane_;
            int birdeye_right_plane_;

            BirdEyeVizProperties(){
                birdeye_scale_factor_ = 13.0;
                birdeye_far_plane_ = 50;
                birdeye_left_plane_ = -20;
                birdeye_right_plane_ = 20;
            }
        };

        struct LessPoint2f{
            bool operator()(const cv::Point2f &lhs, const cv::Point2f &rhs) const{
                return (lhs.x == rhs.x) ? (lhs.y < rhs.y) : (lhs.x < rhs.x);
            }
        };

        public:
            Tracking(System* pSys, Map *map, const string &strSettingPath, const int sensor);

            cv::Mat GrabImageRGB(const cv::Mat &imRGB, cv::Mat &imD, const cv::Mat &imFlow, const cv::Mat &maskSEM,
                                 const cv::Mat &mTcw_gt, const vector<vector<float>> &vObjPose_gt, const double &timestamp,
                                 cv::Mat &imTraj, const int &nImage);

            void GetSceneFlowObj();

            vector<vector<int>> DynObjTracking();

            void DrawLine(cv::KeyPoint &keys, cv::Point2f &flow, cv::Mat &ref_image, const cv::Scalar &color,
                          int thickness=2, int line_type=1, const cv::Point2i &offset=cv::Point2i(0, 0));
            void DrawTransparentSquare(cv::Point center, cv::Vec3b color, int raduis, double alpha, cv::Mat &ref_image);
            void DrawGridBirdeye(double res_x, double res_z, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image);
            void DrawSparseFlowBirdeye(
                const vector<Eigen::Vector3d> &pts, const vector<Eigen::Vector3d> & vel,
                const cv::Mat &camera, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image
            );
            void TransformPointToScaledFrustum(double &pose_x, double &pose_z, const BirdEyeVizProperties &viz_props);

            cv::Mat ObjPoseParsingKT(const vector<float> &vObjPose_gt);
            cv::Mat ObjPoseParsingOx(const vector<float> &vObjPose_gt);

            cv::Mat GetInitModelCam(const vector<int> &MatchId, vector<int> &MatchId_sub);
            cv::Mat GetInitModelObj(const vector<int> &ObjId, vector<int> &ObjId_sub, const int objid);

            void StackObjInfo(vector<cv::KeyPoint> &FeatDynObj, vector<float> &DepDynObj,
                              vector<int> &FeatLabObj);

            vector<vector<pair<int, int>>> GetStaticTrack();
            vector<vector<pair<int, int>>> GetDynamicTrack();
            vector<vector<pair<int, int>>> GetDynamicTrackNew();
            vector<vector<int>> GetObjTrackTime(vector<vector<int>> &ObjTrackLab, vector<vector<int>> &ObjSemanticLab,
                                                vector<vector<int>> &vnSMLabGT);
            
            void GetMetricError(const vector<cv::Mat> &CamPose, vector<vector<cv::Mat>> &RigMot, const vector<vector<cv::Mat>> &ObjPosePre,
                                const vector<cv::Mat> &CamPose_gt, const vector<vector<cv::Mat>> &RigMot_gt,
                                const vector<vector<bool>> &ObjStat);
            void PlotMetricError(const vector<cv::Mat> &CamPose, const vector<vector<cv::Mat>> &RigMot, const vector<vector<cv::Mat>> &ObjPosePre,
                                 const vector<cv::Mat> &CamPse_gt, const vector<vector<cv::Mat>> &RigMot_gt,
                                 const vector<vector<bool>> &ObjStat);
            void GetVelocityError(const vector<vector<cv::Mat>> &RigMot, const vector<vector<cv::Mat>> &PointDyn,
                                  const vector<vector<int>> &FeaLab, const vector<vector<int>> &RMLab,
                                  const vector<vector<float>> &Velo_gt, const vector<vector<int>> &TmpMatch,
                                  const vector<vector<bool>> &ObjStat);

            void RenewFrameInfo(const std::vector<int> &TM_sta);
            void UpdateMask();

        public:
            enum eTrackingState{
                NO_IMAGES_YET=0,
                NOT_INITIALIZED=1,
                OK=2,
            };
            eTrackingState mState;
            eTrackingState mLastProcessedState;

            enum eDataState{
                OMD=1,
                KITTI=2,
                VirtualKITTI=3,
            };

            eDataState mTestData;

            //input sensor
            int mSensor;

            //current Frame
            Frame currentFrame;
            cv::Mat mImGray;

            cv::Mat mImGrayLast;

            cv::Mat mFlowMap, mFlowMapLast;
            cv::Mat mDepthMap;
            cv::Mat mSegMap, mSegMapLast;

            //transfer the ground truth to use identity matrix as origin
            cv::Mat mOriginInv;
            
            //Store temperal matching feature index
            bool bFrame2Frame, bFirstFrame;
            vector<int> TemeralMatch, TemperalMatch_subset;
            vector<cv::KeyPoint> mvKeysLastFrame, mvKeysCurrentFrame;

            vector<cv::KeyPoint> mvTmpObjKeys;
            vector<float> mvTmpObjectDepth;
            vector<int> mvTmpSemObjLabel;
            vector<cv::Point2f> mvTmpObjFlowNext;
            vector<cv::KeyPoint> mvTmpObjCorres;

            //re-projection error
            vector<float> repro_e;

            //save current frame ID
            int f_id;

            //save the global Tracking ID;
            int max_id;

            //save stop frame
            int stopFrame;
            
            //save optimization decision
            bool bLocalBatch;
            bool bGlobalBatch;
            //whether use joint optic-flow formulation
            bool bJoint;

            //window size and overlapping size for local batch optimization
            int nWINDOW_SIZE, nOVERLAP_SIZE;

            //Max Tracking points on background and object in each frame
            int nMaxTrackPointBG, nMaxTrackPointOBJ;

            //Scene Flow magnitude and distribution threshold
            float fsFMgThres, fsFDsThres;
            
            //save timing values
            vector<float> all_timing;

            //use sampled feature or detected feature for background
            int nUseSampleFea;

        protected:
            void Track();

            void Initialization();

            ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

            System* mpSystem;

            Map* mpMap;

            cv::Mat mK;
            cv::Mat mDistCoef;
            float mbf;

            //threshold close/far points
            //points seen as close by the stereo/RGBD sensor are considered reliable
            float mThDepth;
            float mThDepthObj;

            //the depth map scale factor
            float mDepthMapFactor;

            //current matches in frame
            int mnMatchesInliers;
            
            //Last frame info
            Frame mLastFrame;

            //motion model
            cv::Mat mVelocity;

            bool mbRGB;
    };
}
#endif