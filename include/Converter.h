#ifndef CONVERTER_H
#define CONVERTER_H

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

#include "dependencies/g2o/g2o/types/types_six_dof_expmap.h"
#include "dependencies/g2o/g2o/types/types_seven_dof_expmap.h"
#include <string>
#include <vector>

namespace WeiSLAM{
    class Converter{
        public:
            static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptor);
            
            static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
            static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &Sim3);

            static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
            static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
            static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4> &m);
            static cv::Mat toCvMat(const Eigen::Matrix3d &m);
            static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1> &m);
            static cv::Mat toCvMat(const Eigen::MatrixXd &m);
            static cv::Mat toCvMat(const cv::Point3f &point3F);

            static cv::Point3f toPoint3f(const cv::Mat & cvMat);

            static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t);
            static cv::Mat tocvSkewMatrix(const cv::Mat &m);

            static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat &cvVector);
            static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Point3f &cvPoint);
            static Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat &cvMat3);
            static Eigen::Matrix<double, 4, 4> toMatrix4d(const cv::Mat &cvMat4);
            static std::vector<float> toQuaternion(const cv::Mat &M);

            static bool isRotationMatrix(const cv::Mat &R);
            static std::vector<float> toEuler(const cv::Mat &R);

            static cv::Mat toInvMatrix(const cv::Mat &T);
    };
}
#endif