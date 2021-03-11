#include "../include/Converter.h"

namespace WeiSLAM{
    std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat &Descriptors){
        std::vector<cv::Mat> vDesc;
        vDesc.reserve(Descriptors.rows);
        for(int i=0; i<Descriptors.rows; i++){
            vDesc.push_back(Descriptors.row(i));
        }

        return vDesc;
    }

    g2o::SE3Quat Converter::toSE3Quat(const cv::Mat &mat){
        Eigen::Matrix<double, 3, 3> R;
        R << mat.at<float>(0, 0), mat.at<float>(0, 1), mat.at<float>(0, 2),
             mat.at<float>(1, 0), mat.at<float>(1, 1), mat.at<float>(1, 2),
             mat.at<float>(2, 0), mat.at<float>(2, 1), mat.at<float>(2, 2);
        
        Eigen::Matrix<double, 3, 1> t(mat.at<float>(0,3), mat.at<float>(1, 3), mat.at<float>(2, 3));

        return g2o::SE3Quat(R, t);
    }

    cv::Mat Converter::toCvMat(const g2o::SE3Quat &SE3){
        Eigen::Matrix<double, 4, 4> eigenMat = SE3.to_homogeneous_matrix();
        return toCvMat(eigenMat);
    }

    cv::Mat Converter::toCvMat(const g2o::Sim3 &Sim3){
        Eigen::Matrix3d eigenR = Sim3.rotation().toRotationMatrix();
        Eigen::Vector3d eigent = Sim3.translation();
        double s = Sim3.scale();
        return toCvSE3(s*eigenR, eigent);
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 4, 4> & m){
        cv::Mat cvMat(4, 4, CV_32F);
        for(int i=0; i<4; i++){
            for(int j=0; j<4; j++){
                cvMat.at<float>(i, j) = m(i, j);
            }
        }

        return cvMat.clone();
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix3d &m){
        cv::Mat cvMat(3, 3, CV_32F);
        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++){
                cvMat.at<float>(i, j) = m(i, j);
            }
        }
        
        return cvMat.clone();
    }

    cv::Mat Converter::toCvMat(const Eigen::MatrixXd &m){
        cv::Mat cvMat(m.rows(), m.cols(), CV_32F);
        for(int i=0; i<m.rows(); i++){
            for (int j=0; j<m.cols(); j++){
                cvMat.at<float>(i, j) = m(i, j);
            }
        }

        return cvMat.clone();
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 3, 1> &m){
        cv::Mat cvMat(3, 1, CV_32F);
        for(int i=0; i<3; i++){
            cvMat.at<float>(i) = m(i);
        }

        return cvMat.clone();
    }

    cv::Mat Converter::toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t){
        cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);
        for(int i=0; i<3; i++){
            for(int j = 0; j<3; j++){
                cvMat.at<float>(i, j) = R(i, j);
            }
        }
        for(int i=0; i<3; i++){
            cvMat.at<float>(i, 3) = t(i);
        }

        return cvMat.clone();
    }

    Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Mat &cvVector){
        Eigen::Matrix<double, 3, 1> v;
        v <<cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

        return v;
    }

    Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Point3f &cvPoint){
        Eigen::Matrix<double, 3, 1> v;
        v << cvPoint.x, cvPoint.y, cvPoint.z;

        return v;
    }

    Eigen::Matrix<double, 3, 3> Converter::toMatrix3d(const cv::Mat &cvMat3){
        Eigen::Matrix<double, 3, 3> M;
        M << cvMat3.at<float>(0, 0), cvMat3.at<float>(0, 1), cvMat3.at<float>(0, 2),
             cvMat3.at<float>(1, 0), cvMat3.at<float>(1, 1), cvMat3.at<float>(1, 2),
             cvMat3.at<float>(2, 0), cvMat3.at<float>(2, 1), cvMat3.at<float>(2, 2);

        return M;
    }

    Eigen::Matrix<double, 4, 4> Converter::toMatrix4d(const cv::Mat &cvMat4){
        Eigen::Matrix<double, 4, 4> M;
        M <<cvMat4.at<float>(0, 0), cvMat4.at<float>(0, 1), cvMat4.at<float>(0, 2), cvMat4.at<float>(0, 3),
            cvMat4.at<float>(1, 0), cvMat4.at<float>(1, 1), cvMat4.at<float>(1, 2), cvMat4.at<float>(1, 3),
            cvMat4.at<float>(2, 0), cvMat4.at<float>(2, 1), cvMat4.at<float>(2, 2), cvMat4.at<float>(2, 3),
            cvMat4.at<float>(3, 0), cvMat4.at<float>(3, 1), cvMat4.at<float>(3, 2), cvMat4.at<float>(3, 3);

        return M;
    }

    cv::Mat Converter::tocvSkewMatrix(const cv::Mat & v){
        return (cv::Mat_<float>(3, 3)<< 0, -v.at<float>(2), v.at<float>(1),
                                       v.at<float>(2), 0, -v.at<float>(0), 
                                       -v.at<float>(1), v.at<float>(0), 0);
    }

    bool Converter::isRotationMatrix(const cv::Mat &R){
        cv::Mat Rt;
        cv::transpose(R, Rt);
        cv::Mat shouldBeIdentity = Rt * R;
        cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

        return cv::norm(I, shouldBeIdentity) < 1e-6;
    }

    std::vector<float> Converter::toEuler(const cv::Mat &R){
        assert(isRotationMatrix(R));
        float sy = sqrt(R.at<float>(0, 0) * R.at<float>(0, 0) + R.at<float>(1, 0) * R.at<float>(1, 0));

        bool singular = sy < 1e-6;

        float x, y, z;
        if(!singular){
            x = atan2(R.at<float>(2, 1), R.at<float>(2, 2));
            y = atan2(-R.at<float>(2, 0), sy);
            z = atan2(R.at<float>(1, 0), R.at<float>(0, 0));
        }else{
            x = atan2(-R.at<float>(1, 2), R.at<float>(1, 1));
            y = atan2(-R.at<float>(2, 0), sy);
            z = 0;
        }

        std::vector<float> v_eular(3);
        v_eular[0] = x;
        v_eular[1] = y;
        v_eular[2] = z;

        return v_eular;
    }
}