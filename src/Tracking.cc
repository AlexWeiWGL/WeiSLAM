#include "../include/Tracking.h"

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "../include/cvplot/cvplot.h"

#include "../include/Converter.h"
#include "../include/Map.h"
#include "../include/Optimizer.h"

#include <iostream>
#include <string>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include <mutex>
#include <unistd.h>
#include <numeric>
#include <algorithm>
#include <map>
#include <random>

using namespace std;

bool SortPairInt(const pair<int, int> &a, const pair<int, int> &b){
    return (a.second > b.second);
}

namespace WeiSLAM{
    Tracking::Tracking(System *pSys, Map *pMap, const string &strSettingPath, const int sensor):
        mState(NO_IMAGES_YET), mSensor(sensor), mpSystem(pSys), mpMap(pMap)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if(k3 != 0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if(fps==0)
            fps = 30;
        
        cout << endl << "Camera Parameters: " << endl << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- fps: " << fps << endl;

        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if(mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        int nFeatures = fSettings["ORBextractor.scaleFactor"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nlevel = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nlevel, fIniThFAST, fMinThFAST);

        if(sensor==System::STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nlevel, fIniThFAST, fMinThFAST);

        cout << endl << "System Parameters: " << endl << endl;

        int DataCode = fSettings["ChooseData"];
        switch(DataCode)
        {
            case 1:
                mTestData = OMD;
                cout << "- tested dataset: OMD" << endl;
                break;
            case 2:
                mTestData = KITTI;
                cout << "- tested dataset: KITTI" << endl;
                break;
        }

        if(sensor==System::STEREO || sensor==System::RGBD)
        {
            mThDepth = (float)fSettings["ThDepthBG"];
            mThDepthObj = (float)fSettings["ThDepthOBJ"];
            cout << "- depth threshold (background/object): " << mThDepth << "/" << mThDepthObj << endl;
        }

        if(sensor == System::RGBD)
        {
            mDepthMapFactor = fSettings["DepthMapFactor"];
            cout << "- depth map factor: " << mDepthMapFactor << endl;
        }

        nMaxTrackPointBG = fSettings["MaxTrackPointBG"];
        nMaxTrackPointOBJ = fSettings["MaxTrackPointOBJ"];
        cout << "- max tracking points: " << "(1) background: " << nMaxTrackPointBG << " (2) object: " << nMaxTrackPointOBJ << endl;

        fsFMgThres = fSettings["SFMgThres"];
        fsFDsThres = fSettings["SFDsThres"];
        cout << "- scene flow paras: " << "(1) magnitude: " << fsFMgThres << "(2) percentage: " << fsFDsThres << endl;

        nWINDOW_SIZE = fSettings["WINDOW_SIZE"];
        nOVERLAP_SIZE = fSettings["OVERLAP_SIZE"];
        cout << "- local batch paras: " << "(1) window: " << nWINDOW_SIZE << "(2) overlap: " << nOVERLAP_SIZE << endl;

        nUseSampleFea = fSettings["UseSampleFeature"];
        if(nUseSampleFea == 1)
            cout << "- used sampled feature for background scene..." << endl;
        else
            cout << "- used detected feature for background scene..." << endl;
        
    }

    cv::Mat Tracking::GrabImageRGB(const cv::Mat &imRGB, cv::Mat &imD, const cv::Mat &imFlow, 
                                   const cv::Mat &maskSEM, const cv::Mat &mTcw_gt, const vector<vector<float>> &vObjPose_gt,
                                   const double &timestamp, cv::Mat &imTraj, const int &nImage)
    {
        stopFrame = nImage - 1;
        bJoint = true;
        cv::RNG rng((unsigned)time(NULL));

        if(mState == NO_IMAGES_YET)
            f_id = 0;
        
        mImGray = imRGB;

        for(int i=0; i<imD.rows; i++){
            for(int j=0; j<imD.cols; j++){
                if(imD.at<float>(i, j) < 0)
                    imD.at<float>(i, j) = 0;
                else{
                    if(mTestData == OMD)
                        imD.at<float>(i, j) = imD.at<float>(i, j)/mDepthMapFactor;
                    else if(mTestData == KITTI)
                    {
                        imD.at<float>(i, j) = mbf/(imD.at<float>(i, j) / mDepthMapFactor);
                    }
                }
            }
        }

        cv::Mat imDepth = imD;

        if(mImGray.channels() == 3)
        {
            if(mbRGB)
                cv::cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cv::cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        }
        else if(mImGray.channels() == 4)
        {
            if(mbRGB)
                cv::cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cv::cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        //save map in the tracking head
        mDepthMap = imD;
        mFlowMap = imFlow;
        mSegMap = maskSEM;

        //initialize timing vector
        all_timing.resize(5, 0);

        if(mState != NO_IMAGES_YET)
        {
            clock_t s_0, e_0;
            double mask_upd_time;
            s_0 = clock();

            //update mask information
            UpdateMask();
            e_0 = clock();
            mask_upd_time = (double)(e_0-s_0)/CLOCKS_PER_SEC*1000;
            all_timing[0] = mask_upd_time;

        }

        currentFrame = Frame(mImGray, imDepth, imFlow, maskSEM, timestamp, mpORBextractorLeft, mK, mDistCoef, mbf, mThDepth, mThDepthObj, nUseSampleFea);


        if(mState != NO_IMAGES_YET)
        {
            cout << "Update Current Frame From Last ....." << endl;

            currentFrame.mvStatKeys = mLastFrame.mvCorres;
            currentFrame.N_s = currentFrame.mvStatKeys.size();

            currentFrame.mvStatDepth = vector<float>(currentFrame.N_s, -1);
            for(int i=0; i<currentFrame.N_s; i++){
                const cv::KeyPoint &kp = currentFrame.mvStatKeys[i];

                const int v = kp.pt.y;
                const int u = kp.pt.x;

                if(u<(mImGray.cols-1) && u>0 && v<(mImGray.rows-1) && v > 0)
                {
                    float d = imDepth.at<float>(v, u);

                    if(d>0)
                        currentFrame.mvStatDepth[i] = d;
                } 
            }
        }
    }
}