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

            void DrawLine(cv::KeyPoint &keys, cv::Point2f &flow, cv::Mat &ref_image, const cv::Scalar * color,
                          int thickness=2, int line_type=1, const cv::Point2i &offset=cv::Point2i(0, 0));
            void DrawTransparentSquare(cv::Point center, cv::Vec3b color, int raduis, double alpha, cv::Mat &ref_image);
            void DrawGridBirdeye(double res_x, double res_z, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image);
            void DrawSparseFlowBirdeye(
                const vector<Eigen::Vector3d> &pts, const vector<Eigen::Vector3d> & vel,
                const cv::Mat &camera, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image
            );
            void TransformPointToScaledFrustum(double &pose_x, double &pose_z, const )
    };
}
#endif