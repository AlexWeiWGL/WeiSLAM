#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

namespace WeiSLAM{
    using namespace std;

    #define FRAME_GRID_ROWS 48
    #define FRAME_GRID_COLS 64

    class Frame{
        public:
            //Feature extrator
            ORBextractor* imORBextractorLeft, *imORBextractorRight;
            
            double timeStamp;

            cv::Mat K;
            static float fx;
            static float fy;
            static float cx;
            static float cy;
            static float invfx;
            static float invfy;
            cv::Mat distCoef;
            
            //threshold close/ far points
            float thDepth;
            float thDepthObj;

            //number of keypoints
            int N;

            //Stereo baseline multiplied by fx
            float mbf;

            //stereo basline in meters
            float mb;

            //vector of keypoints and undistorted
            vector<cv::KeyPoint> mvKeys, mvKeysRight;
            vector<cv::KeyPoint> mvKeysUh;

            //corresponding stereo coordinate and depth for each keypoint
            vector<float> mvuRight;
            vector<float> mvDepth;

            //ORB descriptor, each row associated to a keypoint
            cv::Mat descriptor, descriptorsRight;

            //number of keypoints
            int N_s;

            // store keypoints and descriptors
            vector<cv::KeyPoint> mvStatKeys, mvStatKeysRight;

            //store dense key points and depth on objects
            vector<cv::KeyPoint> mvObjKeys;
            vector<float> mvObjDepth;
            vector<cv::Mat> mvObj3DPoint;

            //correnspondence for the objects
            vector<cv::KeyPoint> mvObjCorres;

            //optical flow for the obejcts
            vector<cv::Point2f> mvObjFlowGT, mvObjFlowNext;

            //semantic object label of all the foreground features
            vector<int> semObjLabel;
            
            //save the object status(false for outlier, true for inlier)
            vector<bool> bObjStat;

            //depth for each keypoint
            vector<float> mvStatDepth;

            //store the label index for each features: -1 (outlier/unknown), o(static), 1....n(object label)
            vector<int> objLabel;

            //store the 3D flow vector and the 2D re-projection error vector
            vector<cv::Point3f> flow_3d;
            vector<cv::Point2f> flow_2d;

            //store the motion of objects
            vector<cv::Mat> objMod;
            vector<cv::Mat> objPosePre;
            vector<cv::Point2f> speed;
            vector<int> nModLabel;
            vector<int> nSemPosition;
            vector<int> vObjBoxID;  //bounding rectangle for every object
            vector<vector<int>> vnObjID;    //object id in current frame
            vector<vector<int>> vnObjInlierID;  //object od in current frame
            vector<cv::Mat> vObjCentre3D;       //3D in the world coordinate frame
            vector<cv::Mat> vObjCentre2D;   // 2D in the image plane

            cv::Mat mInitModel;     //for initializing motion


            vector<cv::KeyPoint> mvCorres; //correspondence
            vector<cv::Point2f> mvFlow, mvFlowNext;    //optical flow
            //temoral saved
            vector<float> mvStatDepthTmp;
            vector<cv::KeyPoint> mvStatKeysTmp;
            vector<cv::Mat> mvStat3DPointTmp;
            vector<int> vSemLabelTmp;
            vector<int> vObjLabel_gtTmp;
            int N_s_tmp;

            //inlier ID generated in this frame
            vector<int> nStatInlierID;
            vector<int> nDynInlierID;

            //ground truth
            vector<cv::Mat> vObjPose_gt;
            vector<int> nSemPosi_gt;
            vector<cv::Mat> vObjMod_gt;
            vector<float> vObjSpeed_gt;

            cv::Mat mTcw_gt;
            vector<int> vObjLabel_gt;//0 background, 1....n instance label

            vector<bool> mvbOutlier; // flag to identify outlier associations

            //keypoints are assigned to cells in a grid to reduce the matching complexity
            static float mfGridElementWidthInv;
            static float mfGridElementHeightInv;
            vector<size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
            
            //Camera pose
            cv::Mat camPose;

            //current and next frame id
            static long unsigned int nNextId;
            long unsigned int mnId;

            //Scale pyramid info
            int scaleLevels;
            float scaleFactor;
            float logScaleFactor;
            vector<float> scaleFactors;
            vector<float> invScaleFactors;
            vector<float> levelSigma2;
            vector<float> invLevelSigma2;

            //Undistorted Image Bounds
            static float minX;
            static float maxX;
            static float minY;
            static float maxy;

            static bool initialComputations;

        private:
            void UndistortKeyPoints();

            void ComputeImageBounds(const cv::Mat &imLeft);

            void AssignFeaturesToGrid();

            cv::Mat mRcw;       //rotation
            cv::Mat mtcw;    //translation
            cv::Mat mRwc;   //rotation reverse
            cv::Mat mOw;    //center

             

        public:
            Frame();

            Frame(const Frame & frame);
            Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imFlow, const cv::Mat &maskSEM, const double &timeStamp, ORBextractor* extractor,
                  cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, const float &theDepthObj, const int &UseSampleFea);

            void ExtractORB(int flag, const cv::Mat &im);
            void SetPose(cv::Mat Tcw);

            void UpdatePoseMatrices();

            inline cv::Mat GetCameraCenter(){
                return mOw.clone();
            }

            inline cv::Mat GetRotationInverse(){
                return mRwc.clone();
            }

            //compute the cell of a KeyPoint(return false if outside the grid)
            bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

            vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel = -1, const int maxLevel = -1)const;

            void ComputeStereoFromRGBD(const cv::Mat &imgDepth);

            cv::Mat UnprojectStereo(const int &i);
            cv::Mat UnprojectStereoStat(const int &i, const bool &addnoise);
            cv::Mat UnprojectStereoObject(const int &i, const bool &addnoise);
            cv::Mat UnprojectStereoObjectCamera(const int &i, const bool &addnoise);
            cv::Mat UnprojectStereoObjectNoise(const int &i, const bool &addnoise);
            cv::Mat ObtainFlowDepthObject(const int &i, const bool &addnoise);
            cv::Mat ObtainFlowDepthCamera(const int &i, const bool &addnoise);

            vector<cv::KeyPoint> SampleKeyPoints(const int &rows, const int &cols);
    };
}

#endif