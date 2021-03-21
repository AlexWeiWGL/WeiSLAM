//
// Created by alexwei on 2021/3/21.
//

#ifndef WEISLAM_PINHOLE_H
#define WEISLAM_PINHOLE_H


#include <assert.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/assume_abstract.hpp>

#include "TwoViewReconstruction.h"
#include "GeometricCamera.h"

namespace WeiSLAM{
    class Pinhole : public GeometricCamera{
        friend class boost::serialization::access;

        template<class Archive> void serialize(Archive& ar, const unsigned int version)
        {
            ar & boost::serialization::base_object<GeometricCamera>*(this);
        }

    public:
        Pinhole(){
            mvParameters.reserve(4);
            mnId = nNextId++;
            mnType = CAM_PINHOLE;
        }

        Pinhole(const std::vector<float> _vParameters) : GeometricCamera(_vParameters),tvr(nullptr){
            assert(mvParameters.size()==4);
            mnId = nNextId++;
            mnType = CAM_PINHOLE;
        }

        Pinhole(Pinhole* pPinhole):GeometricCamera(pPinhole->mvParameters), tvr(nullptr){
            assert(mvParameters.size()==4);
            mnId = nNextId++;
            mnType = CAM_PINHOLE;
        }

        ~Pinhole(){
            if(tvr) delete tvr;
        }

        cv::Point2f project(const cv::Point2f &p3D);
        cv::Point2f project(const cv::Mat &m3D);
        Eigen::Vector2d project(const Eigen::Vector3d & v3D);
        cv::Mat projectMat(const cv::Point3f & p3D);

        float uncertainty2(const Eigen::Matrix<double, 2, 1> &p2D);

        cv::Point3f unproject(const cv::Point2f &p2D);
        cv::Mat unprojectMat(const cv::Point3f &p2D);

        cv::Mat projectJac(const cv::Point3f &p3D);
        Eigen::Matrix<double, 2, 3> projectJac(const Eigen::Vector3d& v3D);

        cv::Mat unprojectJac(const cv::Point2f &p2D);

        bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint> vKeys2, const std::vector<int> &vMatches12,
                                     cv::Mat R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);
        cv::Mat toK();

        bool epipolarConstrain(GeometricCamera* pCamera2, const cv::KeyPoint& k1, const cv::KeyPoint& kp2, const cv::Mat& R12, const cv::Mat& t12, const float sigmaLevel, const float unc);
        bool matchAndtriangulate(const cv::KeyPoint& kp1, const cv::KeyPoint& jp2, GeometricCamera* pOther,
                                 cv::Mat & Tcw1, cv::Mat& Tcw2,
                                 const float sigmaLevel1, const float sigmaLevel2,
                                 cv::Mat & x3Dtriangulated){return false;}

        friend std::ostream& operator<<(std::ostream &os, const Pinhole& ph);
        friend std::istream& operator>>(std::istream &os, Pinhole& ph);

    private:
        cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

        TwoViewReconstruction* tvr;
    };
}

#endif //WEISLAM_PINHOLE_H
