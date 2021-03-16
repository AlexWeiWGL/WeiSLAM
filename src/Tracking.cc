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


            mvTmpObjKeys = currentFrame.mvObjKeys;
            mvTmpObjectDepth = currentFrame.mvObjDepth;
            mvTmpSemObjLabel = currentFrame.semObjLabel;
            mvTmpObjFlowNext = currentFrame.mvObjFlowNext;
            mvTmpObjCorres = currentFrame.mvObjCorres;

            currentFrame.mvObjKeys = mLastFrame.mvObjCorres;
            currentFrame.mvObjDepth.resize(currentFrame.mvObjKeys.size(), -1);
            currentFrame.mvObjDepth.resize(currentFrame.mvObjKeys.size(), -1);
            for(int i=0; i<currentFrame.mvObjKeys.size(); ++i){
                const int u = currentFrame.mvObjKeys[i].pt.x;
                const int v = currentFrame.mvObjKeys[i].pt.y;
                if(u<(mImGray.cols - 1) && u>0 && v<(mImGray.rows - 1) && v>0 && imDepth.at<float>(v, u)<mThDepthObj && imDepth.at<float>(v, u) > 0)
                {
                    currentFrame.mvObjDepth[i] = imDepth.at<float>(v, u);
                    currentFrame.semObjLabel[i] = maskSEM.at<int>(v, u);
                }
                else{
                    currentFrame.mvObjDepth[i] = 0.1;
                    currentFrame.semObjLabel[i] = 0;
                }
            }

            cout << "Update Current Frame, Done!" << endl;
        }

        //assign pose ground truth
        if(mState==NO_IMAGES_YET)
        {
            currentFrame.mTcw_gt = Converter::toInvMatrix(mTcw_gt);
            mOriginInv = mTcw_gt;
        }
        else{
            currentFrame.mTcw_gt = Converter::toInvMatrix(mTcw_gt)*mOriginInv;
        }

        //assign object pose ground truth
        currentFrame.nSemPosi_gt.resize(vObjPose_gt.size());
        currentFrame.vObjPose_gt.resize(vObjPose_gt.size());

        for(int i=0; i<vObjPose_gt.size(); ++i){
            //label
            currentFrame.nSemPosi_gt[i] = vObjPose_gt[i][1];
            //pose
            if(mTestData == OMD)
                currentFrame.vObjPose_gt[i] = ObjPoseParsingOx(vObjPose_gt[i]);
            else if(mTestData==KITTI)
                currentFrame.vObjPose_gt[i] = ObjPoseParsingKT(vObjPose_gt[i]);
        }
        //save temperal matches for visualization
        TemeralMatch = vector<int>(currentFrame.N_s, -1);
        //initialize object label
        currentFrame.objLabel.resize(currentFrame.mvObjKeys.size(), -2);

        cout << "Start Tracking ...... " << endl;
        Track();
        cout << "End Tracking ...... " << endl;

        //update global id
        f_id = f_id + 1;

        // display label on the image
        if(timestamp != 0 && bFrame2Frame == true)
        {
            vector<cv::KeyPoint> KeyPoints_tmp(1);

            for(int i=0; i<TemperalMatch_subset.size(); i=i+1)
            {
                if(TemperalMatch_subset[i] >= currentFrame.mvStatKeys.size())
                    continue;
                KeyPoints_tmp[0] = currentFrame.mvStatKeys[TemperalMatch_subset[i]];
                if(KeyPoints_tmp[0].pt.x>=(mImGray.cols-1) || KeyPoints_tmp[0].pt.x<=0 || KeyPoints_tmp[0].pt.y>=(mImGray.rows - 1) || KeyPoints_tmp[0].pt.y<=0)
                    continue;
                if(maskSEM.at<int>(KeyPoints_tmp[0].pt.y, KeyPoints_tmp[0].pt.x)!=0)
                    continue;
                cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0, 0, 0), 1); //red
            }

            //static and dynamic objects
            for(int i=0; i<currentFrame.objLabel.size(); ++i)
            {
                if(currentFrame.objLabel[i] == -1 || currentFrame.objLabel[i] == -2)
                    continue;
                int l = currentFrame.objLabel[i];
                if(l > 25)
                    l = l/2;
                
                KeyPoints_tmp[0] = currentFrame.mvObjKeys[i];
                if(KeyPoints_tmp[0].pt.x >= (mImGray.cols - 1) || KeyPoints_tmp[0].pt.x <= 0 || KeyPoints_tmp[0].pt.y >= (mImGray.rows - 1) || KeyPoints_tmp[0].pt.y <=0)
                    continue;
                switch(l)
                {
                    case 0:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0, 0, 255), 1);
                        break;
                    case 1:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(128, 0, 128), 1);
                        break;
                    case 2:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255, 255, 0), 1);
                        break;
                    case 3:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0, 255, 0), 1);
                        break;
                    case 4:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255, 0, 0), 1);
                        break;
                    case 5:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0, 255, 255), 1);
                        break;
                    case 6:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(64, 0, 64), 1);
                        break;
                    case 7:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255, 255, 255), 1);
                        break;
                    case 8:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(255, 228, 128), 1);
                        break;
                    case 9:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(180, 105, 255), 1);
                        break;
                    case 10:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(128, 42, 42), 1);
                        break;
                    case 11:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(128, 0, 42), 1);
                        break;
                    case 12:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0, 20, 40), 1);
                        break;
                    case 13:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(60, 0, 128), 1);
                        break;
                    case 14:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(128, 60, 128), 1);
                        break;
                    case 15:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(60, 20, 100), 1);
                        break;
                    case 16:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(100, 50, 0), 1);
                        break;
                    case 17:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(35, 142, 107), 1);
                        break;
                    case 18:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(142, 107, 35), 1);
                        break;
                    case 19:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(45, 82, 160), 1);
                        break;
                    case 20:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(200, 100, 100), 1);
                        break;
                    case 21:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(0, 20, 200), 1);
                        break;
                    case 22:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(20, 107, 177), 1);
                        break;
                    case 23:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(180, 105, 255), 1);
                        break;
                    case 24:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(100, 100, 100), 1);
                        break;
                    case 25:
                        cv::drawKeypoints(imRGB, KeyPoints_tmp, imRGB, cv::Scalar(90, 0, 155), 1);
                        break;
                }
            }

            cv::imshow("Static Background and Object Points", imRGB);
            if(f_id<4)
                cv::waitKey(1);
            else
                cv::waitKey(1);
        }

        //show bounding box with speed
        if(timestamp != 0 && bFrame2Frame == true && mTestData == KITTI)
        {
            cv::Mat mImBGR(mImGray.size(), CV_8UC3);
            cv::cvtColor(mImGray, mImBGR, CV_GRAY2RGB);
            for(int i=0; i<currentFrame.vObjBoxID.size(); ++i)
            {
                if(currentFrame.speed[i].x == 0)
                    continue;
                
                cv::Point pt1(vObjPose_gt[currentFrame.vObjBoxID[i]][2], vObjPose_gt[currentFrame.vObjBoxID[i]][3]);
                cv::Point pt2(vObjPose_gt[currentFrame.vObjBoxID[i]][4], vObjPose_gt[currentFrame.vObjBoxID[i]][5]);

                cv::rectangle(mImBGR, pt1, pt2, cv::Scalar(0, 255, 0), 2);

                string sp_est = to_string(currentFrame.speed[i].x/36);
                sp_est.resize(5);
                string output_est = sp_est + "km/h";
                cv::putText(mImBGR, output_est, cv::Point(pt1.x, pt1.y-10), cv::FONT_HERSHEY_DUPLEX, 0.9, CV_RGB(0, 255, 0), 2);

            }

            cv::imshow("Object Speed", mImBGR);
            cv::waitKey(1);
        }

        //show trajectory result
        if(mTestData == KITTI)
        {
            int sta_x = 300, sta_y = 100, radi = 2, thic = 5;
            float scale = 6;
            cv::Mat CamPos = Converter::toInvMatrix(currentFrame.camPose);
            int x = int(CamPos.at<float>(0, 3)*scale) + sta_x;
            int y = int(CamPos.at<float>(2, 3)*scale) + sta_y;

            cv::rectangle(imTraj, cv::Point(x, y), cv::Point(x+10, y+10), cv::Scalar(0, 0, 255), 1);
            cv::rectangle(imTraj, cv::Point(10, 30), cv::Point(550, 60), CV_RGB(0, 0, 0), CV_FILLED);
            cv::putText(imTraj, "Camera Trajectory (RED SQUARE)", cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 0.6, CV_RGB(255, 255, 255), 1);
            char text[100];
            sprintf(text, "x= %02fm y = %02fm z = %02fm", CamPos.at<float>(0, 3), CamPos.at<float>(1, 3), CamPos.at<float>(2, 3));
            cv::putText(imTraj, text, cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar::all(255), 1);
            cv::putText(imTraj, "Object Trajectories (COLORED CIRCLES)", cv::Point(10, 70), cv::FONT_HERSHEY_COMPLEX, 0.6, CV_RGB(255, 255, 255), 1);

            for(int i=0; i<currentFrame.vObjCentre3D.size(); ++i)
            {
                if(currentFrame.vObjCentre3D[i].at<float>(0, 0) == 0 && currentFrame.vObjCentre3D[i].at<float>(0, 2)==0)
                    continue;
                int x = int(currentFrame.vObjCentre3D[i].at<float>(0, 0)*scale) + sta_x;
                int y = int(currentFrame.vObjCentre3D[i].at<float>(0, 2)*scale) + sta_y;

                int l = currentFrame.nModLabel[i];
                switch(l)
                {
                    case 1:
                        cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(128, 0, 128), thic);
                        break;
                    case 2:
                        cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(0, 255, 255), thic);
                        break;
                    case 3:
                        cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(0, 255, 0), thic);
                        break;
                    case 4:
                        cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(0, 0, 255), thic);
                        break;
                    case 5:
                        cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(255, 255, 0), thic);
                        break;
                    case 6:
                        cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(128, 0, 128), thic);
                        break;
                    case 7:
                        cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(255, 255, 255), thic);
                        break;
                    case 8:
                        cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(196, 228, 255), thic);
                        break;
                    case 9:
                        cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(180, 105, 255), thic);
                        break;
                    case 10:
                        cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(42, 42, 165), thic);
                        break;
                    case 11:
                        cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(35, 142, 107), thic);
                        break;
                    case 12:
                        cv::circle(imTraj, cv::Point(x, y), radi, CV_RGB(45, 82, 160), thic);
                        break;
                }
            }
            
            cv::imshow("Camera and Object Trajectories", imTraj);
            if(f_id < 3)
                cv::waitKey(1);
            else
                cv::waitKey(1);
            
        }
        
        
        if(timestamp != 0 && bFrame2Frame == true && mTestData == OMD)
        {
            PlotMetricError(mpMap->vmCameraPose, mpMap->vmRigidMotion, mpMap->vmObjPosePre,
                            mpMap->vmCameraPose_GT, mpMap->vmRigidMotion_GT, mpMap->vbObjStat);
        }

        mImGrayLast = mImGray;
        TemeralMatch.clear();
        mSegMapLast = mSegMap;
        mFlowMapLast = mFlowMap;

        return currentFrame.camPose.clone();
        
    }

    void Tracking::Track()
    {
        if(mState == NO_IMAGES_YET)
            mState = NOT_INITIALIZED;
        
        mLastProcessedState = mState;

        if(mState == NOT_INITIALIZED)
        {
            bFirstFrame = true;
            bFrame2Frame = false;

            if(mSensor == System::RGBD)
                Initialization();
            
            if(mState != OK)
                return;
        }
        else
        {
            bFrame2Frame = true;

            cout << "---------------------------------------------------" << endl;
            cout << ".............Dealing with Camera Pose.............." << endl;
            cout << "---------------------------------------------------" << endl;
            
            //update temperalMatch
            for(int i=0; i<currentFrame.N_s; ++i){
                TemeralMatch[i] = i;
            }

            clock_t s_1_1, s_1_2, e_1_1, e_1_2;
            double cam_pos_time;
            s_1_1 = clock();
            //get initial estimate using p3p plus RanSac
            cv::Mat iniTcw = GetInitModelCam(TemeralMatch, TemperalMatch_subset);
            e_1_1 = clock();

            s_1_2 = clock();

            //compute the pose with new matching
            currentFrame.SetPose(iniTcw);
            if(bJoint)
                Optimizer::PoseOptimizationFlow2Cam(&currentFrame, &mLastFrame, TemperalMatch_subset);
            else
                Optimizer::PoseOptimizationNew(&currentFrame, &mLastFrame, TemperalMatch_subset);
            
            e_1_2 = clock();
            cam_pos_time = (double)(e_1_1 - s_1_1)/CLOCKS_PER_SEC*1000 + (double)(e_1_2 - s_1_2)/CLOCKS_PER_SEC*1000;
            all_timing[1] = cam_pos_time;

            //update motion model
            if(!mLastFrame.camPose.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                mVelocity = currentFrame.camPose*LastTwc;
            }

            cv::Mat T_lc_inv = currentFrame.camPose*Converter::toInvMatrix(mLastFrame.camPose);
            cv::Mat T_lc_gt = mLastFrame.mTcw_gt*Converter::toInvMatrix(currentFrame.mTcw_gt);
            cv::Mat RePoEr_cam = T_lc_inv * T_lc_gt;

            float t_rpe_cam = sqrt(RePoEr_cam.at<float>(0, 3) * RePoEr_cam.at<float>(0, 3) + RePoEr_cam.at<float>(1, 3)*RePoEr_cam.at<float>(1, 3)+ RePoEr_cam.at<float>(2, 3)*RePoEr_cam.at<float>(2, 3));
            float trace_rpe_cam = 0;
            for(int i=0; i<3; ++i)
            {
                if(RePoEr_cam.at<float>(i,i)>1.0)
                    trace_rpe_cam = trace_rpe_cam + 1.0 - (RePoEr_cam.at<float>(i, i)-1.0);
                else
                    trace_rpe_cam = trace_rpe_cam + RePoEr_cam.at<float>(i, i);
            }
            cout << fixed << setprecision(6);
            float r_rpe_cam = acos((trace_rpe_cam - 1.0)/2.0) * 180.0 / CV_PI;
            
            cout << "the relative pose error of estimated camera pose, " << "t: " << t_rpe_cam << "R: " << r_rpe_cam << endl;



            cout << "--------------------------------------------------------" << endl;
            cout << "..............Dealing with Objects Now.................." << endl;
            cout << "--------------------------------------------------------" << endl;


            //compute sparse scene flow to the found matches
            GetSceneFlowObj();

            //dynamic object tracking
            cout << "Object Tracking ...... " << endl;
            vector<vector<int>> objIdNew = DynObjTracking();
            cout << "Object Tracking, Done !" << endl;
        
            //Object motion estimation

            clock_t s_3_1, s_3_2, e_3_1, e_3_2;
            double obj_mot_time = 0, t_con = 0;

            currentFrame.bObjStat.resize(objIdNew.size(), true);
            currentFrame.objMod.resize(objIdNew.size());
            currentFrame.objPosePre.resize(objIdNew.size());
            currentFrame.vObjMod_gt.resize(objIdNew.size());
            currentFrame.vObjSpeed_gt.resize(objIdNew.size());
            currentFrame.speed.resize(objIdNew.size());
            currentFrame.vObjBoxID.resize(objIdNew.size());
            currentFrame.vObjCentre3D.resize(objIdNew.size());
            currentFrame.vnObjID.resize(objIdNew.size());
            currentFrame.vnObjInlierID.resize(objIdNew.size());
            repro_e.resize(objIdNew.size(), 0.0);
            cv::Mat Last_Twc_gt = Converter::toInvMatrix(mLastFrame.mTcw_gt);
            cv::Mat Curr_Twc_gt = Converter::toInvMatrix(currentFrame.mTcw_gt);

            //main loop
            for(int i=0; i<objIdNew.size(); ++i){
                cout << endl << "Processing Object No.[" << currentFrame.nModLabel[i] << "]:" << endl;
                
                //get the ground truth object motion
                cv::Mat L_p, L_c, L_w_p, L_w_c, H_p_c, H_p_c_body;
                bool bCheckGT1 = false, bCheckGT2 = false;
                for(int k=0; k<mLastFrame.nSemPosi_gt.size(); ++k)
                {
                    if(mLastFrame.nSemPosi_gt[k] == currentFrame.nSemPosition[i]){
                        if(mTestData==OMD)
                        {
                            L_w_p = mLastFrame.vObjPose_gt[k];
                        }
                        else if(mTestData==KITTI)
                        {
                            L_p = mLastFrame.vObjPose_gt[k];
                            L_w_p = Last_Twc_gt * L_p;
                        }
                        bCheckGT1 = true;
                        break;
                    }
                }

                for(int k=0; k<currentFrame.nSemPosi_gt.size(); ++k)
                {
                    if(currentFrame.nSemPosi_gt[k]==currentFrame.nSemPosition[i]){
                        if(mTestData==OMD)
                        {
                            L_w_c = currentFrame.vObjPose_gt[k];
                        }
                        else if(mTestData  == KITTI)
                        {
                            L_c = currentFrame.vObjPose_gt[k];
                            L_w_c = Curr_Twc_gt * L_c;
                        }
                        currentFrame.vObjBoxID[i] = k;
                        bCheckGT2 = true;
                        break;
                    }
                }

                if(!bCheckGT1 || !bCheckGT2)
                {
                    cout << "Found a detected object woth no ground truth motion !!!" << endl;
                    currentFrame.bObjStat[i] = false;
                    currentFrame.vObjMod_gt[i] = cv::Mat::eye(4, 4, CV_32F);
                    currentFrame.objMod[i] = cv::Mat::eye(4, 4, CV_32F);
                    currentFrame.vObjCentre3D[i] = (cv::Mat_<float>(3, 1) << 0.f, 0.f, 0.f);
                    currentFrame.vObjSpeed_gt[i] = 0.0;
                    currentFrame.vnObjInlierID[i] = objIdNew[i];
                    continue;
                }

                cv::Mat L_w_p_inv = Converter::toInvMatrix(L_w_p);
                H_p_c = L_w_c * L_w_p_inv;
                currentFrame.vObjMod_gt[i] = H_p_c_body;
                currentFrame.objPosePre[i] = L_w_p;
                
                cv::Mat objCentre3D_pre = (cv::Mat_<float>(3, 1) << 0.f, 0.f, 0.f);
                for(int j=0; j<objIdNew[i].size(); ++j)
                {
                    cv::Mat x3D_p = mLastFrame.UnprojectStereoObject(objIdNew[i][j], 0);
                    objCentre3D_pre = objCentre3D_pre + x3D_p;
                }
                objCentre3D_pre = objCentre3D_pre / objIdNew[i].size();
                currentFrame.vObjCentre3D[i] = objCentre3D_pre;

                s_3_1 = clock();

                //get initial model and inlier set using p3p RanSac
                vector<int> objIdTest = objIdNew[i];
                currentFrame.vnObjID[i] = objIdTest;
                vector<int> objIdTest_in;
                currentFrame.mInitModel = GetInitModelObj(objIdTest, objIdTest_in, i);

                e_3_1 = clock();

                if(objIdTest_in.size()<50)
                {
                    cout << "Object Initialization Fail !!!" << endl;
                    currentFrame.bObjStat[i] = false;
                    currentFrame.vObjMod_gt[i] = cv::Mat::eye(4, 4, CV_32F);
                    currentFrame.objMod[i] = cv::Mat::eye(4, 4, CV_32F);
                    currentFrame.vObjCentre3D[i] = (cv::Mat_<float>(3, 1) << 0.f, 0.f, 0.f);
                    currentFrame.vObjSpeed_gt[i] = 0.0;
                    currentFrame.speed[i] = cv::Point2f(0.f, 0.f);
                    currentFrame.vnObjInlierID[i] = objIdTest_in;
                    continue;
                }

                s_3_2 = clock();
                //save object motion and label
                vector<int> inlierID;
                if(bJoint)
                {
                    cv::Mat obj_X_tmp = Optimizer::PoseOptimizationFlow2(&currentFrame, &mLastFrame, objIdTest_in, inlierID);
                    currentFrame.objMod[i] = Converter::toInvMatrix(currentFrame.camPose)*obj_X_tmp;
                }
                else
                    currentFrame.objMod[i] = Optimizer::PoseOptimizationObjMot(&currentFrame, &mLastFrame, objIdTest_in, inlierID);
                e_3_2 = clock();
                t_con = t_con + 1;
                obj_mot_time = obj_mot_time + (double)(e_3_1-s_3_1)/CLOCKS_PER_SEC*1000 + (double)(e_3_2-s_3_2)/CLOCKS_PER_SEC*1000;
                
                currentFrame.vnObjInlierID[i] = inlierID;



                //get the ground truth object speed here
                cv::Mat sp_gt_v, sp_gt_v2;
                sp_gt_v = H_p_c.rowRange(0,3).col(3) - (cv::Mat::eye(3, 3, CV_32F)- H_p_c.rowRange(0, 3).colRange(0,3))*objCentre3D_pre;
                sp_gt_v2 = L_w_p.rowRange(0,3).col(3) - L_w_c.rowRange(0,3).col(3);
                float sp_gt_norm = sqrt(sp_gt_v.at<float>(0)*sp_gt_v.at<float>(0)+sp_gt_v.at<float>(1) + sp_gt_v.at<float>(2)*sp_gt_v.at<float>(2)) * 36;

                currentFrame.vObjSpeed_gt[i] = sp_gt_norm;

                //Calculate the estimated object speed
                cv::Mat sp_est_v;
                sp_est_v = currentFrame.objMod[i].rowRange(0, 3).col(3) - (cv::Mat::eye(3, 3, CV_32F)-currentFrame.objMod[i].rowRange(0, 3))*objCentre3D_pre;
                float sp_est_norm = sqrt(sp_est_v.at<float>(0)*sp_est_v.at<float>(0) + sp_est_v.at<float>(1)*sp_est_v.at<float>(1) + sp_est_v.at<float>(2)*sp_est_v.at<float>(2)) *36;

                cout << "estimated and gound truth object speed: " << sp_est_norm << "km/h " << sp_gt_norm << "km/h" << endl;

                currentFrame.speed[i].x = sp_est_norm * 36;
                currentFrame.speed[i].y = sp_gt_norm * 36;

                //calculate the relative pose error

                cv::Mat H_p_c_body_est = L_w_p_inv * currentFrame.objMod[i] * L_w_p;
                cv::Mat RePoEr = Converter::toInvMatrix(H_p_c_body_est)*H_p_c_body;

                float t_rpe = sqrt(pow(RePoEr.at<float>(0,3), 2) + pow(RePoEr.at<float>(1,3), 2) + pow(RePoEr.at<float>(2, 3), 2));
                float trace_rpe = 0;
                for(int i=0; i<3; ++i)
                {
                    if(RePoEr.at<float>(i, i)>1.0)
                        trace_rpe = trace_rpe + 1.0 - (RePoEr.at<float>(i, i)- 1.0);
                    else    
                        trace_rpe = trace_rpe + RePoEr.at<float>(i, i);
                }
                float r_rpe = acos((trace_rpe - 1.0)/2.0)*180.0/CV_PI;
                cout << "the relative pose error of the object, " << "t: " << t_rpe << "R: " << r_rpe << endl;

            }

            if(t_con!=0)
            {
                obj_mot_time = obj_mot_time/t_con;
                all_timing[3] = obj_mot_time;
            }
            else
                all_timing[3] = 0;

            clock_t s_4, e_4;
            double map_upd_time;
            s_4 = clock();
            RenewFrameInfo(TemperalMatch_subset);
            e_4 = clock();
            map_upd_time = (double) (e_4-s_4)/CLOCKS_PER_SEC*1000;

            //save timing analysis to the map
            mpMap->vfAll_time.push_back(all_timing);

            cout << "Assign tp LastFrame ......" << endl;

            mvKeysLastFrame = mLastFrame.mvStatKeys;
            mvKeysCurrentFrame = currentFrame.mvStatKeys;

            mLastFrame = Frame(currentFrame);
            mLastFrame.mvObjKeys = currentFrame.mvObjKeys;
            mLastFrame.mvObjDepth = currentFrame.mvObjDepth;
            mLastFrame.semObjLabel = currentFrame.semObjLabel;

            mLastFrame.mvStatKeys = currentFrame.mvStatKeysTmp;
            mLastFrame.mvStatDepth = currentFrame.mvStatDepthTmp;


            cout << "Assign to lastframe, Done !" << endl;

            //save some stuffs for graph structure

            cout << "Save Graph Structure ......" << endl;

            //detected static features, corresponding depth and associations
            mpMap->vpFeatSta.push_back(currentFrame.mvStatKeysTmp);
            mpMap->vfDepSta.push_back(currentFrame.mvStatDepthTmp);
            mpMap->vp3DPointSta.push_back(currentFrame.mvStat3DPointTmp);
            mpMap->vnAssoSta.push_back(currentFrame.nStatInlierID);

            //detected dynamic object features, corresponding depth and associations
            mpMap->vpFeatDyn.push_back(currentFrame.mvObjKeys);
            mpMap->vfDepDyn.push_back(currentFrame.mvObjDepth);
            mpMap->vp3DPointDyn.push_back(currentFrame.mvObj3DPoint);
            mpMap->vnAssoDyn.push_back(currentFrame.nDynInlierID);
            mpMap->vnFeatLabel.push_back(currentFrame.objLabel);

            if(f_id == stopFrame || bLocalBatch)
            {
                //save static feature tracklets
                mpMap->TrackletSta = GetStaticTrack();
                //save dynamic feature tracklets
                mpMap->TrackletDyn = GetDynamicTrackNew();
            }

            cv::Mat CameraPoseTmp = Converter::toInvMatrix(currentFrame.camPose);
            mpMap->vmCameraPose.push_back(CameraPoseTmp);
            mpMap->vmCameraPose_RF.push_back(CameraPoseTmp);
            //rigid motions and labels including camera(lable=0) and objects(lable>0)
            vector<cv::Mat> Mot_Tmp, ObjPose_Tmp;
            vector<int> Mot_Lab_Tmp, Sem_Lab_Tmp;
            vector<bool> Obj_Stat_Tmp;

            cv::Mat CameraMotionTmp = Converter::toInvMatrix(mVelocity);
            Mot_Tmp.push_back(CameraMotionTmp);
            ObjPose_Tmp.push_back(CameraMotionTmp);
            Mot_Lab_Tmp.push_back(0);
            Sem_Lab_Tmp.push_back(0);
            Obj_Stat_Tmp.push_back(true);

            //save object motions and label
            for(int i=0; i<currentFrame.objMod.size(); ++i)
            {
                if(!currentFrame.bObjStat[i])
                    continue;
                Obj_Stat_Tmp.push_back(currentFrame.bObjStat[i]);
                Mot_Tmp.push_back(currentFrame.objMod[i]);
                ObjPose_Tmp.push_back(currentFrame.objPosePre[i]);
                Mot_Lab_Tmp.push_back(currentFrame.nModLabel[i]);
                Sem_Lab_Tmp.push_back(currentFrame.nSemPosition[i]);
            }

            //save to the map
            mpMap->vmRigidMotion.push_back(Mot_Tmp);
            mpMap->vmObjPosePre.push_back(ObjPose_Tmp);
            mpMap->vmRigidMotion_RF.push_back(Mot_Tmp);
            mpMap->vnRMLabel.push_back(Mot_Lab_Tmp);
            mpMap->vnSMLabel.push_back(Sem_Lab_Tmp);
            mpMap->vbObjStat.push_back(Obj_Stat_Tmp);

            //count the tracking times fo each unique object
            if(max_id > 1)
                mpMap->vnObjTraTime = GetObjTrackTime(mpMap->vnRMLabel, mpMap->vnSMLabel, mpMap->vnSMLabelGT);
            
            //------------------------Ground Truth-----------------------------

            //ground truth camera pose
            cv::Mat CameraPoseTmpGT = Converter::toInvMatrix(currentFrame.mTcw_gt);
            mpMap->vmCameraPose_GT.push_back(CameraPoseTmpGT);

            //ground truth rigid Motions
            vector<cv::Mat> Mot_Tmp_gt;
            //Save Camera Motion
            cv::Mat CameraMotionTmp_gt = mLastFrame.mTcw_gt*Converter::toInvMatrix(currentFrame.mTcw_gt);
            Mot_Tmp_gt.push_back(CameraMotionTmp_gt);
            //save object motions
            for(int i=0; i<currentFrame.vObjMod_gt.size(); ++i)
            {
                if(!currentFrame.bObjStat[i])
                    continue;
                Mot_Tmp_gt.push_back(currentFrame.vObjMod_gt[i]);
            }
            //save to the map    
            mpMap->vmRigidMotion_GT.push_back(Mot_Tmp_gt);

            //ground truth camera and object speeds
            vector<float> Speed_Tmp_gt;
            //Save camera speed
            Speed_Tmp_gt.push_back(1.0);
            //save object motions
            for(int i=0; i<currentFrame.vObjSpeed_gt.size(); ++i)
            {
                if(!currentFrame.bObjStat[i])
                    continue;
                Speed_Tmp_gt.push_back(currentFrame.vObjSpeed_gt[i]);
            }
            //save to the map
            mpMap->vfAllSpeed_GT.push_back(Speed_Tmp_gt);

            cout << "Save graph structure, Done !" << endl;
        }

        // partial batch optimize on all the measurements(Local optimization)
        bLocalBatch = false;
        if((f_id-nOVERLAP_SIZE + 1) % (nWINDOW_SIZE-nOVERLAP_SIZE)==0 && f_id>=nWINDOW_SIZE-1 && bLocalBatch)
        {
            cout << "--------------------------------------------" <<  endl;
            cout << "! ! ! ! Partial Batch Optimization ! ! ! ! " << endl;
            cout << "--------------------------------------------" << endl;
            clock_t s_5, e_5;
            double loc_ba_time;
            s_5 = clock();
            //get partial batch optimization
            Optimizer::PartialBatchOptimization(mpMap, mK, nWINDOW_SIZE);
            e_5 = clock();
            loc_ba_time = (double)(e_5 - s_5)/CLOCKS_PER_SEC*1000;
            mpMap->fLBA_time.push_back(loc_ba_time);
        }

        //Full batch optimize on all the measurements(global optimization)

        bGlobalBatch = true;
        if(f_id == stopFrame)
        {
            //metric error before optimization
            GetMetricError(mpMap->vmCameraPose, mpMap->vmRigidMotion, mpMap->vmObjPosePre,
                           mpMap->vmCameraPose_GT, mpMap->vmRigidMotion_GT, mpMap->vbObjStat);
            
            if(bGlobalBatch && mTestData==KITTI)
            {
                //get full batch optimization
                Optimizer::FullBatchOptimization(mpMap, mK);

                //Metric error after optimization
                GetMetricError(mpMap->vmCameraPose_RF, mpMap->vmRigidMotion_RF, mpMap->vmObjPosePre,
                               mpMap->vmCameraPose_GT, mpMap->vmRigidMotion_GT, mpMap->vbObjStat);
                
            }
        }

        mState = OK;
    }

    void Tracking::Initialization()
    {
        cout << "Initialization ......." << endl;

        //initialize the 3d points
        {
            //static
            vector<cv::Mat> mv3DPointTmp;
            for(int i=0; i<currentFrame.mvStatKeysTmp.size(); ++i)
            {
                mv3DPointTmp.push_back(Optimizer::Get3DinCamera(currentFrame.mvStatKeysTmp[i], currentFrame.mvStatDepthTmp[i], mK));
            }
            currentFrame.mvStat3DPointTmp = mv3DPointTmp;

            vector<cv::Mat> mvObj3DPointTmp;
            for(int i=0; i<currentFrame.mvObjKeys.size(); ++i)
            {
                mvObj3DPointTmp.push_back(Optimizer::Get3DinCamera(currentFrame.mvObjKeys[i], currentFrame.mvObjDepth[i], mK));
            }
            currentFrame.mvObj3DPoint = mv3DPointTmp;

        }

        //save detected static features and corresponding depth
        mpMap->vpFeatSta.push_back(currentFrame.mvStatKeysTmp);
        mpMap->vfDepSta.push_back(currentFrame.mvStatDepthTmp);
        mpMap->vp3DPointSta.push_back(currentFrame.mvStat3DPointTmp);

        //save detected dynamic object features and corresponding depth
        mpMap->vpFeatDyn.push_back(currentFrame.mvObjKeys);
        mpMap->vfDepDyn.push_back(currentFrame.mvObjDepth);
        mpMap->vp3DPointDyn.push_back(currentFrame.mvObj3DPoint);

        //save camera pose
        mpMap->vmCameraPose.push_back(cv::Mat::eye(4, 4, CV_32F));
        mpMap->vmCameraPose_RF.push_back(cv::Mat::eye(4, 4, CV_32F));
        mpMap->vmCameraPose_GT.push_back(cv::Mat::eye(4, 4, CV_32F));

        //set frame pose to the origin
        currentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
        currentFrame.mTcw_gt = cv::Mat::eye(4, 4, CV_32F);

        mLastFrame = Frame(currentFrame);
        mLastFrame.mvObjKeys = currentFrame.mvObjKeys;
        mLastFrame.mvObjDepth = currentFrame.mvObjDepth;
        mLastFrame.semObjLabel = currentFrame.semObjLabel;

        mLastFrame.mvStatKeys = currentFrame.mvStatKeysTmp;
        mLastFrame.mvStatDepth = currentFrame.mvStatDepthTmp;
        mLastFrame.N_s = currentFrame.N_s_tmp;
        mvKeysLastFrame = mLastFrame.mvStatKeys;

        mState = OK;

        cout << "Initialization, Done !" << endl;
    }

    void Tracking::GetSceneFlowObj()
    {   
        //initialzation
        int N = currentFrame.mvObjKeys.size();
        currentFrame.flow_3d.resize(N);

        vector<Eigen::Vector3d> pts_p3d(N, Eigen::Vector3d(-1, -1, -1)), pts_vel(N, Eigen::Vector3d(-1, -1, -1));

        const cv::Mat Rcw = currentFrame.camPose.rowRange(0,3).colRange(0,3);
        const cv::Mat tcw = currentFrame.camPose.rowRange(0,3).col(3);

        //Main loop
        for(int i=0; i<N; ++i)
        {
            if(currentFrame.semObjLabel[i]<=0 || mLastFrame.semObjLabel[i]<= 0)
            {
                currentFrame.objLabel[i] = -1;
                continue;
            }

            //get the 3d flow
            cv::Mat x3D_p = mLastFrame.UnprojectStereoObject(i, 0);
            cv::Mat x3D_c = currentFrame.UnprojectStereoObject(i, 0);

            pts_p3d[i] << x3D_p.at<float>(0), x3D_p.at<float>(1), x3D_p.at<float>(2);

            cv::Point3f flow3d;
            flow3d.x = x3D_c.at<float>(0)- x3D_p.at<float>(0);
            flow3d.y = x3D_c.at<float>(1) - x3D_p.at<float>(1);
            flow3d.z = x3D_c.at<float>(2) - x3D_p.at<float>(2);

            pts_vel[i] << flow3d.x, flow3d.y, flow3d.z;

            currentFrame.flow_3d[i] = flow3d;
        }
    }

    vector<vector<int>> Tracking::DynObjTracking()
    {
        clock_t s_2, e_2;
        double obj_tra_time;
        s_2 = clock();

        //find the unique labels in semantic label
        auto UniLab = currentFrame.semObjLabel;
        sort(UniLab.begin(), UniLab.end());
        UniLab.erase(unique(UniLab.begin(), UniLab.end()), UniLab.end());

        //collect the predicted labels and semantic labels in vector
        vector<vector<int>> Posi(UniLab.size());
        for(int i=0; i<currentFrame.semObjLabel.size(); ++i)
        {
            //skip outliers
            if(currentFrame.objLabel[i] ==  -1)
                continue;
            
            //save object label
            for(int j=0; j <UniLab.size(); ++j)
            {
                if(currentFrame.semObjLabel[i]==UniLab[j]){
                    Posi[j].push_back(i);
                    break;
                }
            }
        }

        
    }
}