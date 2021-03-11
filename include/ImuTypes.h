
#ifndef IMUTYPES_H
#define IMUTYPES_H

#include <vector>
#include <utility>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <mutex>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

namespace WeiSLAM{
    namespace IMU{
        const float GRAVITY_VALUE=9.81;

        class Point{
            public:
                Point(const float &acc_x, const float &acc_y, const float &acc_z,
                        const float &ang_vel_x, const float &ang_vel_y, const float &ang_vel_z,
                        const double &timestamp): a(acc_x, acc_y, acc_z), w(ang_vel_x, ang_vel_y, ang_vel_z), t(timestamp){}
                Point(const cv::Point3f Acc, const cv::Point3f Gyro, const double &timestamp):
                    a(Acc.x, Acc.y, Acc.z), w(Gyro.x, Gyro.y, Gyro.z), t(timestamp){}
            
            public:
                cv::Point3f a;
                cv::Point3f w;
                double t;
        };

        class Bias{
            friend class boost::serialization::access;
            template<class Archive> void serialize(Archive &ar, const unsigned int version){
                ar & bax;
                ar & bay;
                ar & baz;

                ar & bwx;
                ar & bwy;
                ar & bwz;
            }

            public:
                Bias():bax(0), bay(0), baz(0), bwx(0), bwy(0), bwz(0){}
                Bias(const float &b_acc_x, const float &b_acc_y, const float &b_acc_z,
                        const float &b_ang_vel_x, const float &b_ang_vel_y, const float &b_ang_vel_z):
                        bax(b_acc_x), bay(b_acc_y), baz(b_acc_z), bwx(b_ang_vel_x), bwy(b_ang_vel_y), bwz(b_ang_vel_z){}
                void CopyFrom(Bias &b);
                friend std::ostream & operator << (std::ostream &out, const Bias &b);
            
            public:
                float bax, bay, baz;
                float bwx, bwy, bwz;

        };

        class Calib{
            template<class Archive>
            void serializeMatrix(Archive &ar, cv::Mat &mat, const unsigned int version){
                int cols, rows, type;
                bool continuous;

                if(Archive::is_saving::value){
                    cols = mat.cols; rows = mat.rows; type = mat.type();
                    continuous = mat.isContinuous();
                }

                ar & cols & rows & type & continuous;
                if(Archive::is_Loading::value)
                    mat.create(rows, cols, type);
                
                if(continuous){
                    const unsigned int data_size = rows * cols * mat.elemSize();
                    ar & boost::serialization::make_array(mat.ptr(), data_size);
                }else{
                    const unsigned int row_size = cols * mat.elemSize();
                    for (int i=0; i<rows; i++){
                        ar & boost::serialization::make_array(mat.ptr(i), row_size);
                    }
                }
            }

            friend class boost::serialization::access;
            template<class Archive>
            void serialize(Archive & ar, const unsigned int version){
                serializeMatrix(ar, Tcb, version);
                serializeMatrix(ar, Tbc, version);
                serializeMatrix(ar, Cov, version);
                serializeMatrix(ar, CovWalk, version);
            }

            public:
                Calib (const cv::Mat &Tbc_, const float &ng, const float &na, const float &ngw, const float &naw){
                    Set(Tbc_, ng, na, ngw, naw);
                }
                Calib(const Calib &calib);
                Calib(){}

                void Set(const cv::Mat &Tbc_, const float &ng, const float &na, const float &ngw, const float &naw);

            public:
                cv::Mat Tcb;
                cv::Mat Tbc;
                cv::Mat Cov, CovWalk;
        };

        class Preintegrated{
            template<class Archive>
            void serializeMatrix(Archive &ar, cv::Mat & mat, const unsigned int version){
                int cols, rows, type;
                bool continunous;
            
                if(Archive::is_saving::value){
                    cols = mat.cols; rows = mat.rows; type = mat.type();
                    continunous = mat.isContinuous();
                }

                ar & cols &rows & type & continunous;
                if(Archive::is_loading::value)
                    mat.create(rows, cols, type);
                
                if(continunous){
                    const unsigned int data_size = rows * cols * mat.elemSize();
                    ar & boost::serialization::make_array(mat.ptr(), data_size);
                }else{
                    const unsigned int row_size = cols*mat.elemSize();
                    for(int i=0; i<rows; i++){
                        ar & boost::serialization::make_array(mat.ptr(i), row_size);
                    }
                }
            }

            friend class boost::serialization::access;
            template<class Archive>
            void serialize(Archive &ar, const unsigned int version){
                ar & dT;
                serializeMatrix(ar, C, version);
                serializeMatrix(ar, Info, version);
                serializeMatrix(ar, Nga, version);
                serializeMatrix(ar, NgaWalk, version);
                ar & b;
                serializeMatrix(ar, dR, version);
                serializeMatrix(ar, dV, version);
                serializeMatrix(ar, dP, version);
                serializeMatrix(ar, JRg, version);
                serializeMatrix(ar, JVg, version);
                serializeMatrix(ar, JVa, version);
                serializeMatrix(ar, JPg, version);
                serializeMatrix(ar, JPa, version);
                serializeMatrix(ar, avgA, version);
                serializeMatrix(ar, avgW, version);

                ar & bu;
                serializeMatrix(ar, db, version);
                ar & mvMeasurements;
            }

            public:
                Preintegrated(const Bias &b_, const Calib *calib);
                Preintegrated(Preintegrated* pImuPre);
                Preintegrated(){}
                ~Preintegrated(){}
                void CopyFrom(Preintegrated* pImuPre);
                void Initialize(const Bias &b_);
                void IntegratedNewMeasurement(const cv::Point3f &acceleration, const cv::Point3f &angVel, const float &dt);
                void Reintegrate();
                void MergePrevious(Preintegrated* pPrev);
                void SetNewBias(const Bias &bu_);
                IMU::Bias GetDeltaBias(const Bias &b_);
                cv::Mat GetDeltaRotation(const Bias &b_);
                cv::Mat GetDeltaVelocity(const Bias &b_);
                cv::Mat GetDeltaPosition(const Bias &b_);
                cv::Mat GetUpdatedDeltaRotation();
                cv::Mat GetUpdatedDletaVelocity();
                cv::Mat GetUpdatedDeltaPosition();
                cv::Mat GetOriginalDeltaRotation();
                cv::Mat GetOriginalDeltaVelocity();
                cv::Mat GetOriginalDeltaPosition();
                Eigen::Matrix<double, 15, 15> GetInformationMatrix();
                cv::Mat GetDeltaBias();
                Bias GetOriginalBias();
                Bias GetUpdatedBias();
            
            public:
                float dT;
                cv::Mat C;
                cv::Mat Info;
                cv::Mat Nga, NgaWalk;

                Bias b;
                cv::Mat dR, dV, dP;
                cv::Mat JRg, JVg, JVa, JPg, JPa;
                cv::Mat avgA;
                cv::Mat avgW;
            
            private:
                Bias bu;
                cv::Mat db;

                struct integrable{
                    integrable(const cv::Point3f &a_, const cv::Point3f &w_, const float &t_):a(a_), w(w_), t(t_){}
                    cv::Point3f a;
                    cv::Point3f w;
                    float t;
                };

                std::vector<integrable> mvMeasurements;
                std::mutex mMutex;

        };

        cv::Mat ExpSO3(const float &x, const float &y, const float &z);
        Eigen::Matrix<double, 3, 3> ExpSO3(const double &x, const double &y, const double &z);
        cv::Mat ExpSO3(const cv::Mat &v);
        cv::Mat LogSO3(const cv::Mat &R);
        cv::Mat RightJacobianSO3(const float &x, const float &y, const float &z);
        cv::Mat RightJacobianSO3(const cv::Mat &v);
        cv::Mat InverseRightJacobianSO3(const float &x, const float &y, const float &z);
        cv::Mat Skew(const cv::Mat &v);
        cv::Mat NormalizeRotation(const cv::Mat &R);

    }
}

#endif