#include "../include/Tracking.h"

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cvplot/cvplot.h>

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

        reconstruction = new TwoViewReconstruction(K);
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

    cv::Mat Tracking::GrabImageMono(const cv::Mat &imRGB, const cv::Mat &imFlow,
                                   const cv::Mat &maskSEM, const cv::Mat &mTcw_gt, const vector<vector<float>> &vObjPose_gt,
                                   const double &timestamp, cv::Mat &imTraj, const int &nImage)
    {
        stopFrame = nImage - 1;
        bJoint = true;
        cv::RNG rng((unsigned)time(NULL));

        if(mState == NO_IMAGES_YET)
            f_id = 0;
        
        mImGray = imRGB;

//        for(int i=0; i<imD.rows; i++){
//            for(int j=0; j<imD.cols; j++){
//                if(imD.at<float>(i, j) < 0)
//                    imD.at<float>(i, j) = 0;
//                else{
//                    if(mTestData == OMD)
//                        imD.at<float>(i, j) = imD.at<float>(i, j)/mDepthMapFactor;
//                    else if(mTestData == KITTI)
//                    {
//                        imD.at<float>(i, j) = mbf/(imD.at<float>(i, j) / mDepthMapFactor);
//                    }
//                }
//            }
//        }

        //cv::Mat imDepth = imD;

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
        //mDepthMap = imD;
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

        currentFrame = Frame(mImGray, imFlow, maskSEM, timestamp, mpORBextractorLeft, mK, mDistCoef, mbf, mThDepth, mThDepthObj, nUseSampleFea);


        if(mState != NO_IMAGES_YET)
        {
            cout << "Update Current Frame From Last ....." << endl;

            currentFrame.mvStatKeys = mLastFrame.mvCorres;
            currentFrame.N_s = currentFrame.mvStatKeys.size();

            //currentFrame.mvStatDepth = vector<float>(currentFrame.N_s, -1);
//            for(int i=0; i<currentFrame.N_s; i++){
//                const cv::KeyPoint &kp = currentFrame.mvStatKeys[i];
//
//                const int v = kp.pt.y;
//                const int u = kp.pt.x;
//
////                if(u<(mImGray.cols-1) && u>0 && v<(mImGray.rows-1) && v > 0)
////                {
////                    float d = imDepth.at<float>(v, u);
////
////                    if(d>0)
////                        currentFrame.mvStatDepth[i] = d;
////                }
//            }


            mvTmpObjKeys = currentFrame.mvObjKeys;
           // mvTmpObjectDepth = currentFrame.mvObjDepth;
            mvTmpSemObjLabel = currentFrame.semObjLabel;
            mvTmpObjFlowNext = currentFrame.mvObjFlowNext;
            mvTmpObjCorres = currentFrame.mvObjCorres;

            currentFrame.mvObjKeys = mLastFrame.mvObjCorres;
            //currentFrame.mvObjDepth.resize(currentFrame.mvObjKeys.size(), -1);
            //currentFrame.mvObjDepth.resize(currentFrame.mvObjKeys.size(), -1);
            for(int i=0; i<currentFrame.mvObjKeys.size(); ++i){
                const int u1 = currentFrame.mvObjKeys[i].pt.x;
                const int v1 = currentFrame.mvObjKeys[i].pt.y;
                if(u1<(mImGray.cols - 1) && u1>0 && v1<(mImGray.rows - 1) && v1>0){
                    //currentFrame.mvObjDepth[i] = imDepth.at<float>(v, u);
                    currentFrame.semObjLabel[i] = maskSEM.at<int>(v1, u1);
                }
                else{
                    //currentFrame.mvObjDepth[i] = 0.1;
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

            if(mSensor == System::MONOCULAR)
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
            reconstruction->Reconstruct(mLastFrame.mvStatKeys, currentFrame.mvStatKeys, TemeralMatch,RTmp, tTmp, v3DTmp, vbTriangulatedTmp);
            e_1_1 = clock();

            s_1_2 = clock();
            currentFrame.camPose.rowRange(0, 3).colRange(0,3) = RTmp;
            currentFrame.camPose.rowRange(0, 3).col(3) = tTmp;

            for(int i=0; i<currentFrame.N_s; ++i)
            {
                currentFrame.mvStat3DPointTmp[i] = Converter::toPoint3f(currentFrame.Calculate3D(currentFrame.mvCorres[i], currentFrame.mvStatKeysTmp[i], currentFrame.camPose, mK));
            }
            //compute the pose with new matching
            if(bJoint)
                Optimizer::PoseOptimizationFlow2Cam(&currentFrame, &mLastFrame, TemeralMatch); //可能会出现问题TemeralMatch
            else
                Optimizer::PoseOptimizationNew(&currentFrame, &mLastFrame, TemeralMatch);
            
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
                    //尝试计算三维坐标
                    cv::Point3f x3D_p =  Converter::toPoint3f(currentFrame.Calculate3D(currentFrame.mvObjCorres[objIdNew[i][j]], currentFrame.mvObjKeys[objIdNew[i][j]], currentFrame.camPose, mK));
                    currentFrame.mvObj3DPoint[objIdNew[i][j]] = x3D_p;
                    objCentre3D_pre = objCentre3D_pre + (cv::Mat_<float>(3, 1)<< x3D_p.x, x3D_p.y, x3D_p.z); /// new add mono
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
            fill(matches12Sta.begin(), matches12Sta.end(), -1);
            vector<bool> vbTriangulatedSta;
            vector<cv::Point3f> mv3DPointTmp;
            reconstruction->Reconstruct(currentFrame.mvStatKeysTmp, currentFrame.mvStatKeysTmp, matches12Sta,
                                        currentFrame.camPose.rowRange(0, 3).colRange(0, 3), currentFrame.camPose.rowRange(0,3).col(3),
                                        mv3DPointTmp, vbTriangulatedSta);
            currentFrame.mvStat3DPointTmp = mv3DPointTmp;

            fill(matches12Dyn.begin(), matches12Dyn.end(), -1);
            vector<bool> vbTriangulatedDyn;
            vector<cv::Point3f> mvObj3DPointTmp;
            reconstruction->Reconstruct(currentFrame.mvObjKeys, currentFrame.mvObjKeys, matches12Dyn,
                                            currentFrame.camPose.rowRange(0, 3).colRange(0, 3), currentFrame.camPose.rowRange(0, 3).col(3),
                                            mvObj3DPointTmp, vbTriangulatedDyn);

            currentFrame.mvObj3DPoint = mvObj3DPointTmp;

        }

        //save detected static features and corresponding depth
        mpMap->vpFeatSta.push_back(currentFrame.mvStatKeysTmp);
        //mpMap->vfDepSta.push_back(currentFrame.mvStatDepthTmp);
        mpMap->vp3DPointSta.push_back(currentFrame.mvStat3DPointTmp);

        //save detected dynamic object features and corresponding depth
        mpMap->vpFeatDyn.push_back(currentFrame.mvObjKeys);
        //mpMap->vfDepDyn.push_back(currentFrame.mvObjDepth);
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
        //mLastFrame.mvObjDepth = currentFrame.mvObjDepth;
        mLastFrame.semObjLabel = currentFrame.semObjLabel;

        mLastFrame.mvStatKeys = currentFrame.mvStatKeysTmp;
        //mLastFrame.mvStatDepth = currentFrame.mvStatDepthTmp;
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
            cv::Point3f x3D_p = mLastFrame.mvObj3DPoint[i];  //考虑在这用重新用triangulate方法  暂时完成
            cv::Point3f x3D_c = currentFrame.mvObj3DPoint[i];

            pts_p3d[i] << x3D_p.x, x3D_p.y, x3D_p.z;

            cv::Point3f flow3d;
            flow3d.x = x3D_c.x- x3D_p.y;
            flow3d.y = x3D_c.y - x3D_p.y;
            flow3d.z = x3D_c.z - x3D_p.z;

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

        //save object only from Posi()
        vector<vector<int>> objId;
        vector<int> sem_posi;   //semantaic label psotision for the objects
        int shrin_thr_row = 0, shrin_thr_col = 0;
        if(mTestData == KITTI)
        {
            shrin_thr_row = 25;
            shrin_thr_col = 50;
        }
        for(int i=0; i < Posi.size(); ++i)
        {
            //shrink the image to get rid of object parts on the boundary
            float count = 0, count_thres = 0.5;
            for(int j=0; j<Posi[i].size(); ++j)
            {
                const float u = currentFrame.mvObjKeys[Posi[i][j]].pt.x;
                const float v = currentFrame.mvObjKeys[Posi[i][j]].pt.y;
                if(v < shrin_thr_row || v>(mImGray.rows-shrin_thr_row) || u < shrin_thr_col ||u>(mImGray.cols-shrin_thr_col))
                    count = count + 1;
            }
            if(count/Posi[i].size()>count_thres)
            {
                for(int k=0; k<Posi[i].size(); ++k)
                    currentFrame.objLabel[Posi[i][k]] = -1;
                continue;
            }
            else
            {
                objId.push_back(Posi[i]);
                sem_posi.push_back(UniLab[i]);
            }
        }

        //check scene flow distribution of each object and keep the dynamic object
        vector<vector<int>> ObjIdNew;
        vector<int> SemPosNew, obj_dis_tres(sem_posi.size(), 0);
        for(int i=0; i<objId.size(); ++i)
        {
            float sf_min = 100, sf_max=0, sf_mean=0, sf_count=0;
            vector<int>sf_range(10, 0);
            for(int j=0; j<objId[i].size(); ++j)
            {
                float sf_norm = sqrt(pow(currentFrame.flow_3d[objId[i][j]].x, 2) + pow(currentFrame.flow_3d[objId[i][j]].z, 2));
                if(sf_norm < fsFMgThres)
                    sf_count = sf_count + 1;
                if(sf_norm < sf_min)
                    sf_min = sf_norm;
                if(sf_norm > sf_max)
                    sf_max = sf_norm;
                sf_mean = sf_mean + sf_norm;
                {
                    if(sf_norm>=0.0 && sf_norm < 0.05)
                        sf_range[0] = sf_range[0] + 1;
                    else if(sf_norm >= 0.05 && sf_norm < 0.1)
                        sf_range[1] = sf_range[1] + 1;
                    else if(sf_norm >= 0.1 && sf_norm < 0.2)
                        sf_range[2] = sf_range[2] + 1;
                    else if(sf_norm >= 0.2 && sf_norm <0.4)
                        sf_range[3] = sf_range[3] + 1;
                    else if(sf_norm >= 0.4 && sf_norm <0.8)
                        sf_range[4] = sf_range[4] + 1;
                    else if(sf_norm >= 0.8 && sf_norm <1.6)
                        sf_range[5] = sf_range[5] + 1;
                    else if(sf_norm >= 1.6 && sf_norm <3.2)
                        sf_range[6] = sf_range[6] + 1;
                    else if(sf_norm >= 3.2 && sf_norm <6.4)
                        sf_range[7] = sf_range[7] + 1;
                    else if(sf_norm >= 6.4 && sf_norm <12.8)
                        sf_range[8] = sf_range[8] + 1;
                    else if(sf_norm >= 12.8 && sf_norm <25.8)
                        sf_range[9] = sf_range[9] + 1;
                }
            }

            if(sf_count/objId[i].size() > fsFDsThres)
            {
                //label this object as static background
                for(int k=0; k <objId[i].size(); ++k)
                    currentFrame.objLabel[objId[i][k]] = 0;
                continue;
            }
            else if(objId[i].size()<150)
            {
                obj_dis_tres[i] = -1;
                //label this object as far away object
                for(int k = 0; k < objId[i].size(); ++k)
                    currentFrame.objLabel[objId[i][k]] = -1;
                continue;
            }
            else
            {
                ObjIdNew.push_back(objId[i]);
                SemPosNew.push_back(sem_posi[i]);
            }
        }

        //add ground truth tracks

        vector<int> nSemPosi_gt_tmp = currentFrame.nSemPosi_gt;
        for(int i=0; i<sem_posi.size(); ++i)
        {
            for(int j=0; j<nSemPosi_gt_tmp.size(); ++j)
            {
                if(sem_posi[i] == nSemPosi_gt_tmp[j] && obj_dis_tres[i]==-1)
                {
                    nSemPosi_gt_tmp[j] = -1;
                }
            }
        }

        mpMap->vnSMLabelGT.push_back(nSemPosi_gt_tmp);


        //initialize global object id
        vector<int> LabId(ObjIdNew.size());
        for(int i=0; i<ObjIdNew.size(); ++i)
        {
            //save semantic labels in last frame
            vector<int>lb_last;
            for(int k=0; k < ObjIdNew[i].size(); ++i)
                lb_last.push_back(mLastFrame.semObjLabel[ObjIdNew[i][k]]);
            
            //find label that appears most in Lb_last
            //count duplcates
            map<int, int> dups;
            for(int k:lb_last)
                ++dups[k];
            
            // sort them by descending order
            vector<pair<int, int>> sorted;
            for(auto k : dups)
                sorted.push_back(make_pair(k.first, k.second));
            sort(sorted.begin(), sorted.end(), SortPairInt);

            //label the object in current frame
            int New_lab = sorted[0].first;

            if(max_id == 1)
            {
                LabId[i] = max_id;
                for(int k=0; k<ObjIdNew[i].size(); ++ k)
                    currentFrame.objLabel[ObjIdNew[i][k]] = max_id;
                max_id = max_id + 1;
            }
            else{
                bool exist = false;
                for(int k=0; k<mLastFrame.nSemPosition.size(); ++k)
                {
                    if(mLastFrame.nSemPosition[k]==New_lab && mLastFrame.bObjStat[k])
                    {
                        LabId[i] = mLastFrame.nModLabel[k];
                        for(int k=0; k<ObjIdNew[i].size(); ++k)
                            currentFrame.objLabel[ObjIdNew[i][k]] = LabId[i];
                        exist = true;
                        break;
                    }
                }
                if(exist == false)
                {
                    LabId[i] = max_id;
                    for(int k=0; k<ObjIdNew[i].size(); ++k)
                        currentFrame.objLabel[ObjIdNew[i][k]] = max_id;
                    max_id = max_id + 1;
                }
            }
        }

        //assign the model label in current frame
        currentFrame.nModLabel = LabId;
        currentFrame.nSemPosition = SemPosNew;

        e_2 = clock();
        obj_tra_time = (double)(e_2 - s_2)/CLOCKS_PER_SEC*1000;
        all_timing[2] = obj_tra_time;

        return ObjIdNew;
    }

    cv::Mat Tracking::GetInitModelCam(const vector<int> &MatchId, vector<int> &MatchId_sub)
    {
        cv::Mat Mod = cv::Mat::eye(4, 4, CV_32F);
        int N = MatchId.size();

        //construct input
        vector<cv::Point2f> cur_2d(N);
        vector<cv::Point3f> pre_3d(N);

        for(int i=0; i<N; ++i)
        {
            cv::Point2f tmp_2d;
            tmp_2d.x = currentFrame.mvStatKeys[MatchId[i]].pt.x;
            tmp_2d.y = currentFrame.mvStatKeys[MatchId[i]].pt.y;
            cur_2d[i] = tmp_2d;
            cv::Point3f tmp_3d;
            cv::Mat x3D_p;
            vector<bool> vbTriagulated;
            //TwoViewReconstruction::Reconstruct(currentFrame.mvKeys[i], currentFrame.mvCorres[i], currentFrame.camPose.rowRange(0, 3).colRange(0, 3)
             //                                  , currentFrame.camPose.rowRange(0, 3).col(3), x3D_p, vbTriagulated);
            tmp_3d.x = x3D_p.at<float>(0);
            tmp_3d.y = x3D_p.at<float>(1);
            tmp_3d.z = x3D_p.at<float>(2);
            pre_3d[i] = tmp_3d;
        }

        //camera matrix & distortion coefficients
        cv::Mat camera_mat(3, 3, CV_64FC1);
        cv::Mat disCoeffs = cv::Mat::zeros(1, 4, CV_64FC1);
        camera_mat.at<double>(0,0) = mK.at<float>(0,0);
        camera_mat.at<double>(1, 1) = mK.at<float>(1,1);
        camera_mat.at<double>(0, 2) = mK.at<float>(0, 2);
        camera_mat.at<double>(1, 2) = mK.at<float>(1, 2);
        camera_mat.at<double>(2, 2) = 1.0;

        //output
        cv::Mat Rvec(3, 1, CV_64FC1);
        cv::Mat Tvec(3, 1, CV_64FC1);
        cv::Mat d(3, 3, CV_64FC1);
        cv::Mat inliers;

        //solve
        int iter_num = 500;
        double reprojectionError = 0.4, confidence = 0.98;
        cv::solvePnPRansac(pre_3d, cur_2d, camera_mat, disCoeffs, Rvec, Tvec, false,
                    iter_num, reprojectionError, confidence, inliers, cv::SOLVEPNP_AP3P);
        cv::Rodrigues(Rvec, d);

        //assign the result to current pose
        Mod.at<float>(0, 0) = d.at<double>(0, 0); Mod.at<float>(0,1) = d.at<double>(0,1); Mod.at<float>(0, 2) = d.at<double>(0, 2); Mod.at<float>(0, 3) = Tvec.at<double>(0, 0);
        Mod.at<float>(1, 0) = d.at<double>(1, 0); Mod.at<float>(1,1) = d.at<double>(1,1); Mod.at<float>(1, 2) = d.at<double>(1, 2); Mod.at<float>(1, 3) = Tvec.at<double>(1, 0);
        Mod.at<float>(2, 0) = d.at<double>(2, 0); Mod.at<float>(2,1) = d.at<double>(2,1); Mod.at<float>(2, 2) = d.at<double>(2, 2); Mod.at<float>(2, 3) = Tvec.at<double>(2, 0);

        //calculate the re-projection error
        vector<int> MM_inlier;
        cv::Mat MotionModel;

        if(mVelocity.empty())
            MotionModel = cv::Mat::eye(4, 4, CV_32F)*mLastFrame.camPose;
        else
            MotionModel = mVelocity*mLastFrame.camPose;
        for(int i=0; i<N; ++i)
        {
            const cv::Mat x3D = (cv::Mat_<float>(3, 1) << pre_3d[i].x, pre_3d[i].y, pre_3d[i].z);
            const cv::Mat x3D_c = MotionModel.rowRange(0, 3).colRange(0, 3)*x3D + MotionModel.rowRange(0, 3).col(3);
        
            const float xc = x3D_c.at<float>(0);
            const float yc = x3D_c.at<float>(1);
            const float invzc = 1.0/x3D_c.at<float>(2);
            const float u = currentFrame.fx*xc*invzc+currentFrame.cx;
            const float v = currentFrame.fy*yc*invzc+currentFrame.cy;
            const float u_ = cur_2d[i].x - u;
            const float v_ = cur_2d[i].y - v;
            const float Rpe = sqrt(u_*u_ + v_ * v_);
            if(Rpe<reprojectionError){
                MM_inlier.push_back(i);
            }
        }

        cv::Mat output;

        if(inliers.rows>MM_inlier.size())
        {   
            //save the inliers IDs
            output = Mod;
            MatchId_sub.resize(inliers.rows);
            for(int i=0; i<MatchId_sub.size(); ++i){
                MatchId_sub[i] = MatchId[inliers.at<int>(i)];
            }
        }
        else{
            output = MotionModel;
            MatchId_sub.resize(MM_inlier.size());
            for(int i=0; i<MatchId_sub.size(); ++i){
                MatchId_sub[i] = MatchId[MM_inlier[i]];
            }
        }
        
        return output;
    }

    cv::Mat Tracking::GetInitModelObj(const vector<int> &ObjId, vector<int> &ObjId_sub, const int objid)
    {
        cv::Mat Mod = cv::Mat::eye(4, 4, CV_32F);
        int N = ObjId.size();

        //construct input
        vector<cv::Point2f> cur_2d(N);
        vector<cv::Point3f> pre_3d(N);
        for(int i=0; i<N; ++i)
        {
            cv::Point2f tmp_2d;
            tmp_2d.x = currentFrame.mvObjKeys[ObjId[i]].pt.x;
            tmp_2d.y = currentFrame.mvObjKeys[ObjId[i]].pt.y;
            cur_2d[i] = tmp_2d;
            cv::Point3f tmp_3d;
            cv::Mat x3D_p = mLastFrame.UnprojectStereoObject(ObjId[i], 0);
            tmp_3d.x = x3D_p.at<float>(0);
            tmp_3d.y = x3D_p.at<float>(1);
            tmp_3d.z = x3D_p.at<float>(2);
            pre_3d[i] = tmp_3d;
        }

        //camera matrix & distortion coefficients
        cv::Mat camera_mat(3, 3, CV_64FC1);
        cv::Mat distCoeffs = cv::Mat::zeros(1, 4, CV_64FC1);
        camera_mat.at<double>(0, 0) = mK.at<float>(0, 0);
        camera_mat.at<double>(1, 1) = mK.at<float>(1, 1);
        camera_mat.at<double>(1, 2) = mK.at<float>(1, 2);
        camera_mat.at<double>(0, 2) = mK.at<float>(0, 2);
        camera_mat.at<double>(2, 2) = 1.0;

        //output
        cv::Mat Rvec(3, 1, CV_64FC1);
        cv::Mat Tvec(3, 1, CV_64FC1);
        cv::Mat d(3, 3, CV_64FC1);
        cv::Mat inliers;

        //solve
        int iter_num = 500;
        double reprojectionError = 0.4, confidence = 0.98;
        cv::solvePnPRansac(pre_3d, cur_2d, camera_mat, distCoeffs, Rvec, Tvec, false,
                    iter_num, reprojectionError, confidence, inliers, cv::SOLVEPNP_AP3P);
        cv::Rodrigues(Rvec, d);

        Mod.at<float>(0, 0) = d.at<double>(0, 0); Mod.at<float>(0,1) = d.at<double>(0,1); Mod.at<float>(0, 2) = d.at<double>(0, 2); Mod.at<float>(0, 3) = Tvec.at<double>(0, 0);
        Mod.at<float>(1, 0) = d.at<double>(1, 0); Mod.at<float>(1,1) = d.at<double>(1,1); Mod.at<float>(1, 2) = d.at<double>(1, 2); Mod.at<float>(1, 3) = Tvec.at<double>(1, 0);
        Mod.at<float>(2, 0) = d.at<double>(2, 0); Mod.at<float>(2,1) = d.at<double>(2,1); Mod.at<float>(2, 2) = d.at<double>(2, 2); Mod.at<float>(2, 3) = Tvec.at<double>(2, 0);

        //generate motion model if it does exist from previous frame
        int CurObjLab = currentFrame.nModLabel[objid];
        int PreObjID = -1;

        for(int i=0; i<mLastFrame.nModLabel.size(); ++i)
        {
            if(mLastFrame.nModLabel[i]==CurObjLab)
            {
                PreObjID = i;
                break;
            }
        }

        cv::Mat MotionModel, output;
        vector<int> ObjId_tmp(N, -1);
        if(PreObjID != -1)
        {
            vector<int> MM_inlier;
            MotionModel = currentFrame.camPose*mLastFrame.objMod[PreObjID];
            for(int i=0; i<N; ++i)
            {
                const cv::Mat x3D = (cv::Mat_<float>(3, 1) << pre_3d[i].x, pre_3d[i].y, pre_3d[i].z);
                const cv::Mat x3D_c = MotionModel.rowRange(0, 3).colRange(0, 3)*x3D + MotionModel.rowRange(0, 3).col(3);
        
                const float xc = x3D_c.at<float>(0);
                const float yc = x3D_c.at<float>(1);
                const float invzc = 1.0/x3D_c.at<float>(2);
                const float u = currentFrame.fx*xc*invzc+currentFrame.cx;
                const float v = currentFrame.fy*yc*invzc+currentFrame.cy;
                const float u_ = cur_2d[i].x - u;
                const float v_ = cur_2d[i].y - v;
                const float Rpe = sqrt(u_*u_ + v_ * v_);
                if(Rpe<reprojectionError){
                    MM_inlier.push_back(i);
                }
            }

            if(inliers.rows>MM_inlier.size())
            {
                output = Mod;
                ObjId_sub.resize(inliers.rows);
                for(int i=0; i<ObjId_sub.size(); ++i)
                {
                    ObjId_sub[i] = ObjId[inliers.at<int>(i)];
                    ObjId_tmp[inliers.at<int>(i)] = ObjId[inliers.at<int>(i)];
                }
            }
            else{
                output = MotionModel;
                ObjId_sub.resize(MM_inlier.size());
                for(int i=0; i<ObjId_sub.size(); ++i){
                    ObjId_sub[i] = ObjId[MM_inlier[i]];
                    ObjId_tmp[MM_inlier[i]] = ObjId[MM_inlier[i]];
                }
            }
        }
        else{
            output = Mod;
            ObjId_sub.resize(inliers.rows);
            for(int i=0; i<ObjId_sub.size(); ++i){
                ObjId_sub[i] = ObjId[inliers.at<int>(i)];
                ObjId_tmp[inliers.at<int>(i)] = ObjId[inliers.at<int>(i)];
            }
        }

        for(int i=0; i<ObjId_tmp.size(); ++i){
            if(ObjId_tmp[i] == -1)
                currentFrame.objLabel[ObjId[i]] = -1;
        }

        return output;
    }

    void Tracking::DrawLine(cv::KeyPoint &keys, cv::Point2f &flow, cv::Mat &ref_image, const cv::Scalar &color, int thickness, int line_type, const cv::Point2i &offset)
    {
        auto cv_p1 = cv::Point2i(keys.pt.x, keys.pt.y);
        auto cv_p2 = cv::Point2i(keys.pt.x + flow.x, keys.pt.y + flow.y);

        bool p1_in_bounds = true;
        bool p2_in_bounds = true;
        if((cv_p1.x < 0) && cv_p1.y < 0 && cv_p1.x > ref_image.cols && cv_p1.y > ref_image.rows)
        {
            p1_in_bounds = false;
        }
        if(cv_p1.x<0 && cv_p2.y<0 &&cv_p2.x >ref_image.cols && cv_p2.y > ref_image.rows)
        {
            p2_in_bounds = false;
        }

        if(p1_in_bounds || p2_in_bounds){
            auto p1_offs = offset + cv_p1;
            auto p2_offs = offset + cv_p2;
            if(cv::clipLine(cv::Size(ref_image.cols, ref_image.rows), p1_offs, p2_offs)){
                cv::arrowedLine(ref_image, p1_offs, p2_offs, color, thickness, line_type);
            }
        }
    }

    void Tracking::DrawTransparentSquare(cv::Point center, cv::Vec3b color, int radius, double alpha, cv::Mat &ref_image)
    {
        for(int i=-radius; i<radius; i++){
            for(int j = -radius; j<radius; j++){
                int coord_y = center.y + i;
                int coord_x = center.x + j;

                if(coord_x > 0 && coord_x <ref_image.cols && coord_y <ref_image.rows){
                    ref_image.at<cv::Vec3b>(cv::Point(coord_x, coord_y)) = (1.0-alpha)*ref_image.at<cv::Vec3b>(cv::Point(coord_x, coord_y)) + alpha*color;
                }
            }
        }
    }

    void Tracking::DrawGridBirdeye(double res_x, double res_z, const BirdEyeVizProperties & viz_props, cv::Mat &ref_image)
    {
        auto color = cv::Scalar(0.0, 0.0, 0.0);
        //draw horizontal lines
        for(double i=0; i<viz_props.birdeye_far_plane_; i+=res_z){
            double x_1 = viz_props.birdeye_left_plane_;
            double y_1 = i;
            double x_2 = viz_props.birdeye_right_plane_;
            double y_2 = i;
            TransformPointToScaledFrustum(x_1, y_1, viz_props);
            TransformPointToScaledFrustum(x_2, y_2, viz_props);
            auto p1 = cv::Point(x_1, y_1), p2 = cv::Point(x_2, y_2);
            cv::line(ref_image, p1, p2, color);
        }

        //draw vertical lines
        for(double i=viz_props.birdeye_left_plane_; i<viz_props.birdeye_right_plane_; i+=res_x)
        {
            double x_1 = i;
            double y_1 = 0;
            double x_2 = i;
            double y_2 = viz_props.birdeye_far_plane_;
            TransformPointToScaledFrustum(x_1, y_1, viz_props);
            TransformPointToScaledFrustum(x_2, y_2, viz_props);
            auto p1 = cv::Point(x_1, y_1), p2 = cv::Point(x_2, y_2);
            cv::line(ref_image, p1, p2, color);
        }
    }


    void Tracking::DrawSparseFlowBirdeye(
            const vector<Eigen::Vector3d> &pts, const vector<Eigen::Vector3d> &vel,
            const cv::Mat &camera, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image)
    {
        //for scaling / flipping conv.matrics
        Eigen::Matrix2d flip_mat;
        flip_mat << viz_props.birdeye_scale_factor_*1.0 ,0, 0, viz_props.birdeye_scale_factor_*1.0;
        Eigen::Matrix2d world_to_cam_mat;
        const Eigen::Matrix4d &ref_to_rt_inv = Converter::toMatrix4d(camera);
        world_to_cam_mat << ref_to_rt_inv(0, 0), ref_to_rt_inv(2, 0), ref_to_rt_inv(0, 2), ref_to_rt_inv(2, 2);
        flip_mat = flip_mat*world_to_cam_mat;

        //Parameters
        ref_image = cv::Mat(viz_props.birdeye_scale_factor_*viz_props.birdeye_far_plane_,
                                (-viz_props.birdeye_left_plane_+viz_props.birdeye_right_plane_)*viz_props.birdeye_scale_factor_, CV_32FC3);
        ref_image.setTo(cv::Scalar(1.0, 1.0, 1.0));
        Tracking::DrawGridBirdeye(1.0, 1.0, viz_props, ref_image);

        for(int i=0; i<pts.size(); i++)
        {
            Eigen::Vector3d p_3d = pts[i];
            Eigen::Vector3d p_vel = vel[i];

            if(p_3d[0] == -1 || p_3d[1]==  -1 || p_3d[2] < 0)
                continue;
            if(p_vel[0] > 0.1 || p_vel[2] > 0.1)
                continue;
            
            const Eigen::Vector2d velocity = Eigen::Vector2d(p_vel[0], p_vel[2]);
            Eigen::Vector3d dir(velocity[0], 0.0, velocity[1]);

            double x_1 = p_3d[0];
            double z_1 = p_3d[2];

            double x_2 = x_1 + dir[0];
            double z_2 = z_1 + dir[2];

            if(x_1 > viz_props.birdeye_left_plane_ && x_2 > viz_props.birdeye_left_plane_ &&
               x_1 < viz_props.birdeye_right_plane_ && x_2 <viz_props.birdeye_right_plane_ &&
               z_1 >0 && z_2 > 0 &&
               z_1 < viz_props.birdeye_far_plane_ && z_2 < viz_props.birdeye_far_plane_){

                   TransformPointToScaledFrustum(x_1, z_1, viz_props);
                   TransformPointToScaledFrustum(x_2, z_2, viz_props);

                   cv::arrowedLine(ref_image, cv::Point(x_1, z_1), cv::Point(x_2, z_2), cv::Scalar(1.0, 0.0, 0.0), 1);
                   cv::circle(ref_image, cv::Point(x_1, z_1), 3.0, cv::Scalar(0.0, 0.0, 1.0), -1.0);
            }
        }

        //coord. sys.
        int arrow_len = 60;
        int offset_y = 10;
        cv::arrowedLine(ref_image, cv::Point(ref_image.cols/2, offset_y),
                        cv::Point(ref_image.cols/2 + arrow_len, offset_y),
                        cv::Scalar(1.0, 0, 0), 2);
        cv::arrowedLine(ref_image, cv::Point(ref_image.cols/2, offset_y),
                        cv::Point(ref_image.cols/2, offset_y+arrow_len),
                        cv::Scalar(0.0, 1.0, 0), 2);
        
        //flip image, because it is more intuitive to have ref. point at the bottom of the image
        cv::Mat dst;
        cv::flip(ref_image, dst, 0);
        ref_image = dst;

    }

    void Tracking::TransformPointToScaledFrustum(double &pose_x, double &pose_z, const BirdEyeVizProperties &viz_props)
    {
        pose_x += (-viz_props.birdeye_left_plane_);
        pose_x *= viz_props.birdeye_scale_factor_;
        pose_z *= viz_props.birdeye_scale_factor_;
    }

    cv::Mat Tracking::ObjPoseParsingKT(const vector<float> &vObjPose_gt){
        //assign t vector
        cv::Mat t(3, 1, CV_32FC1);
        t.at<float>(0) = vObjPose_gt[6];
        t.at<float>(1) = vObjPose_gt[7];
        t.at<float>(2) = vObjPose_gt[8];

        //from Euler to Rotation matrix
        cv::Mat R(3, 3, CV_32FC1);

        //assign r vector
        float y = vObjPose_gt[9] + (CV_PI/2);
        float x = 0.0;
        float z = 0.0;

        //the angles are in radians
        float cy = cos(y);
        float sy = sin(y);
        float cx = cos(x);
        float sx = sin(x);
        float cz = cos(z);
        float sz = sin(z);

        float m00, m01, m02, m10, m11, m12, m20, m21, m22;
        //R = Ry* Rx * Rz

        m00 = cy*cz + sy*sz*sx;
        m01 = -cy*sz + sy*sx*cz;
        m02 = sy*cx;
        m10 = cx*sz;
        m11 = cx*cz;
        m12 = -sx;
        m20 = -sy*cz + cy*sx*sz;
        m21 = sy*sz + cy*sx*cz;
        m22 = cy*cx;

        R.at<float>(0, 0) = m00;
        R.at<float>(0, 1) = m01;
        R.at<float>(1, 1) = m11;
        R.at<float>(0, 2) = m02;
        R.at<float>(1, 0) = m10;
        R.at<float>(1, 2) = m12;
        R.at<float>(2, 0) = m20;
        R.at<float>(2, 1) = m21;
        R.at<float>(2, 2) = m22;

        cv::Mat Pose = cv::Mat::eye(4, 4, CV_32F);
        Pose.at<float>(0,0) = R.at<float>(0,0); Pose.at<float>(0,1) = R.at<float>(0,1); Pose.at<float>(0,2) = R.at<float>(0,2); Pose.at<float>(0,3) = t.at<float>(0);
        Pose.at<float>(1,0) = R.at<float>(1,0); Pose.at<float>(1,1) = R.at<float>(1,1); Pose.at<float>(1,2) = R.at<float>(1,2); Pose.at<float>(1,3) = t.at<float>(1);
        Pose.at<float>(2,0) = R.at<float>(2,0); Pose.at<float>(2,1) = R.at<float>(2,1); Pose.at<float>(2,2) = R.at<float>(2,2); Pose.at<float>(2,3) = t.at<float>(2);


        return Pose;
    }

    cv::Mat Tracking::ObjPoseParsingOx(const vector<float> &vObjPose_gt)
    {
        //assign t vector
        cv::Mat t(3, 1, CV_32FC1);
        t.at<float>(0) = vObjPose_gt[2];
        t.at<float>(1) = vObjPose_gt[3];
        t.at<float>(2) = vObjPose_gt[4];

        //from axis-angle to Rotation Matrix
        cv::Mat R(3, 3, CV_32FC1);
        cv::Mat Rvec(3, 1, CV_32FC1);

        //assign r vector
        Rvec.at<float>(0, 0) = vObjPose_gt[5];
        Rvec.at<float>(0, 1) = vObjPose_gt[6];
        Rvec.at<float>(0, 2) = vObjPose_gt[7];

        const float angle = sqrt(pow(vObjPose_gt[5], 2) + pow(vObjPose_gt[6], 2) + pow(vObjPose_gt[7], 2));
        if(angle > 0)
        {
            Rvec.at<float>(0, 0) = Rvec.at<float>(0, 0)/angle;
            Rvec.at<float>(0, 1) = Rvec.at<float>(0, 1)/angle;
            Rvec.at<float>(0, 2) = Rvec.at<float>(0, 2)/angle;
        }

        const float s = sin(angle);
        const float c = cos(angle);

        const float v = 1-c;
        const float x = Rvec.at<float>(0, 0);
        const float y = Rvec.at<float>(0, 1);
        const float z = Rvec.at<float>(0, 2);
        const float xyv = x*y*v;
        const float yzv = y*z*v;
        const float xzv = x*z*v;

        R.at<float>(0, 0) = x*x*v + c;
        R.at<float>(0, 1) = xyv - z*s;
        R.at<float>(0, 2) = xzv + y*s;
        R.at<float>(1, 0) = xyv + z*s;
        R.at<float>(1, 1) = y*y*v + c;
        R.at<float>(1, 2) = yzv - x*s;
        R.at<float>(2, 0) = xzv - y*s;
        R.at<float>(2, 1) = yzv + x*s;
        R.at<float>(2, 2) = z*z*v + c;

        cv::Mat Pose = cv::Mat::eye(4, 4, CV_32F);
        Pose.at<float>(0,0) = R.at<float>(0,0); Pose.at<float>(0,1) = R.at<float>(0,1); Pose.at<float>(0,2) = R.at<float>(0,2); Pose.at<float>(0,3) = t.at<float>(0);
        Pose.at<float>(1,0) = R.at<float>(1,0); Pose.at<float>(1,1) = R.at<float>(1,1); Pose.at<float>(1,2) = R.at<float>(1,2); Pose.at<float>(1,3) = t.at<float>(1);
        Pose.at<float>(2,0) = R.at<float>(2,0); Pose.at<float>(2,1) = R.at<float>(2,1); Pose.at<float>(2,2) = R.at<float>(2,2); Pose.at<float>(2,3) = t.at<float>(2);

        return Converter::toInvMatrix(mOriginInv*Pose);
    }

    void Tracking::StackObjInfo(vector<cv::KeyPoint> &FeatDynObj, vector<float> &DepDynObj, vector<int> &FeatLabObj)
    {
        for(int i=0; i<currentFrame.vnObjID.size(); ++i)
        {
            for(int j=0; j<currentFrame.vnObjID[i].size(); ++j)
            {
                FeatDynObj.push_back(mLastFrame.mvObjKeys[currentFrame.vnObjID[i][j]]);
                FeatDynObj.push_back(currentFrame.mvObjKeys[currentFrame.vnObjID[i][j]]);
                DepDynObj.push_back(mLastFrame.mvObjDepth[currentFrame.vnObjID[i][j]]);
                DepDynObj.push_back(currentFrame.mvObjDepth[currentFrame.vnObjID[i][j]]);
                FeatLabObj.push_back(currentFrame.objLabel[currentFrame.vnObjID[i][j]]);
            }
        }
    }

    vector<vector<pair<int, int>>> Tracking::GetStaticTrack()
    {
        //Get temporal match from map
        vector<vector<int>> TemporalMatch = mpMap->vnAssoSta;
        int N = TemporalMatch.size();
        //save the track id in tracklets for previous frame and current frame
        vector<int> TrackCheck_pre;
        vector<vector<pair<int, int>>> TrackLets;

        int IDsofar = 0;
        for(int i=0; i<N; ++i) {
            //initialize trackCheck
            vector<int> TrackCheck_cur(TemporalMatch[i].size(), -1);

            //check each feature
            for (int j = 0; j < TemporalMatch[i].size(); ++j) {
                //first pair of frames
                if (i == 0) {
                    //check if there's association
                    if (TemporalMatch[i][j] != -1) {
                        //first , save on tracklet consist of two featureID
                        vector<pair<int, int>> TraLet(2);
                        TraLet[0] = make_pair(i, TemporalMatch[i][j]);
                        TraLet[1] = make_pair(i + 1, j);
                        //then, save to the main tracklets list
                        TrackLets.push_back(TraLet);

                        //save tracklet id
                        TrackCheck_cur[j] = IDsofar;
                        IDsofar = IDsofar + 1;
                    } else
                        continue;
                }
                    //frame i and i+1
                else {
                    //check if there is association
                    if (TemporalMatch[i][j] != -1)
                    {
                        //check the TrackID in previous frame
                        //if it is associated before, then add to existing tracklets
                        if (TrackCheck_pre[TemporalMatch[i][j]] != -1) {
                            TrackLets[TrackCheck_pre[TemporalMatch[i][j]]].push_back(make_pair(i + 1, j));
                            TrackCheck_cur[j] = TrackCheck_pre[TemporalMatch[i][j]];
                        }
                            //if not , insert new tracklets
                        else {
                            //first save one tracklet consisting of two featureID
                            vector<pair<int, int>> TraLet(2);
                            TraLet[0] = make_pair(i, TemporalMatch[i][j]);
                            TraLet[1] = make_pair(i + 1, j);
                            //then save to the main tracklets list
                            TrackLets.push_back(TraLet);

                            //save tracklet ID
                            TrackCheck_cur[j] = IDsofar;
                            IDsofar = IDsofar + 1;
                        }
                    }
                    else
                        continue;
                }

            }

            TrackCheck_pre = TrackCheck_cur;
        }

        //display info
        cout << endl;
        cout << "===========================================" << endl;
        cout << "the number of static feature tracklets: " << TrackLets.size() << endl;
        cout << "===========================================" << endl;
        cout << endl;

        vector<int> TrackLength(N, 0);
        for(int i=0; i<TrackLets.size(); ++i)
            TrackLength[TrackLets.size()-2]++;

        int LengthOver_5 = 0;
        ofstream save_track_distri;
        string save_td = "track_distribution_static.txt";
        save_track_distri.open(save_td.c_str(), ios::trunc);
        for(int i=0; i<N; ++i){
            if(TrackLength[i] != 0)
                save_track_distri << TrackLength[i] << endl;
            if(i+2 >= 5)
                LengthOver_5 = LengthOver_5 + TrackLength[i];
        }
        save_track_distri.close();

        return TrackLets;
    }

    vector<vector<pair<int, int>>> Tracking::GetDynamicTrackNew() {
        //Get temporal match from map
        vector<vector<int>> TemporalMatch = mpMap->vnAssoDyn;
        vector<vector<int>> ObjLab = mpMap->vnFeatLabel;
        int N = TemporalMatch.size();
        //save the track id in TrackLets for previous frame and current frame
        vector<int> TrackCheck_pre;
        vector<vector<pair<int, int>>> TrackLets;
        vector<int> ObjectID;

        int IDsofar = 0;
        for(int i=0; i<N; ++i)
        {
            //initialize Trackcheck
            vector<int> TrackCheck_cur(TemporalMatch[i].size(), -1);

            //check each feature
            for(int j=0; j<TemporalMatch[i].size(); ++j)
            {
                //first pair of frames
                if(i==0)
                {
                    if(TemporalMatch[i][j] != -1)
                    {
                        vector<pair<int, int>> TraLet(2);
                        TraLet[0] = make_pair(i, TemporalMatch[i][j]);
                        TraLet[1] = make_pair(i+1, j);

                        TrackLets.push_back(TraLet);
                        ObjectID.push_back(ObjLab[i][j]);

                        TrackCheck_cur[i] = IDsofar;
                        IDsofar = IDsofar + 1;
                    }
                }
                else
                {
                    if(TemporalMatch[i][j] != -1)
                    {
                        if(TrackCheck_pre[TemporalMatch[i][j]] != -1)
                        {
                            TrackLets[TrackCheck_pre[TemporalMatch[i][j]]].push_back(make_pair(i+1, j));
                            TrackCheck_cur[j] = TrackCheck_pre[TemporalMatch[i][j]];
                        }
                        else
                        {
                            vector<pair<int , int>> TraLet(2);
                            TraLet[0] = make_pair(i, TemporalMatch[i][j]);
                            TraLet[1] = make_pair(i+1, j);

                            TrackLets.push_back(TraLet);
                            ObjectID.push_back(ObjLab[i][j]);

                            TrackCheck_cur[j] = IDsofar;
                            IDsofar = IDsofar + 1;
                        }
                    }
                }
            }

            TrackCheck_pre = TrackCheck_cur;
        }

        mpMap->nObjID = ObjectID;

        cout << endl;
        cout << "==========================================" << endl;
        cout << "the number of dynamic feature tracklets: " << TrackLets.size() << endl;
        cout << "==========================================" << endl;
        cout << endl;

        vector<int> TrackLength(N, 0);
        for(int i=0; i<TrackLets.size(); ++i){
            TrackLength[TrackLets[i].size()-2]++;
        }

        int LengtjOver_5 = 0;
        ofstream save_track_distri;
        string save_td = "track_distribution.txt";
        save_track_distri.open(save_td.c_str(), ios::trunc);
        for(int i=0; i<N; ++i)
        {
            if(TrackLength[i] != 0)
                save_track_distri << TrackLength[i] << endl;
            if(i+2 > 5)
                LengtjOver_5 = LengtjOver_5 + TrackLength[i];
        }
        save_track_distri.close();
        cout << "Length over 5 (DYNAMIC):::::::::::::::: " << LengtjOver_5 << endl;

        return TrackLets;
    }

    vector<vector<int>> Tracking::GetObjTrackTime(vector<vector<int>> &ObjTrackLab, vector<vector<int>> &ObjSemanticLab,
                                                  vector<vector<int>> &vnSMLabGT)
    {
        vector<int> TrackCount(max_id-1, 0);
        vector<int> TrackCountGT(max_id-1, 0);
        vector<int> SemanticLabel(max_id-1, 0);
        vector<vector<int>> ObjTrackTime;

        //count each object track
        for(int i=0; i<ObjTrackLab.size(); ++i)
        {
            if(ObjTrackLab[i].size()<2)
                continue;

            for(int j=1; j<ObjTrackLab[i].size(); ++j)
            {
                TrackCount[ObjTrackLab[i][j]-1] = TrackCount[ObjTrackLab[i][j]-1] + 1;
                SemanticLabel[ObjTrackLab[i][j] - 1] = ObjSemanticLab[i][j];
            }
        }

        //count each object track in ground truth
        for(int i=0; i<vnSMLabGT.size(); ++i)
        {
            for(int j=0; j<vnSMLabGT.size(); ++j)
            {
                for(int k=0; k<SemanticLabel.size(); ++k)
                {
                    if(SemanticLabel[k] == vnSMLabGT[i][j])
                    {
                        TrackCountGT[k] = TrackCountGT[k] + 1;
                        break;
                    }
                }
            }
        }

        mpMap->nObjTraCount = TrackCountGT;
        mpMap->nObjTraCountGT = TrackCountGT;
        mpMap->nObjTraSemLab = SemanticLabel;

        //save to each frame the count number
        for(int i=0; i<ObjTrackLab.size(); ++i)
        {
            vector<int> TrackTimeTmp(ObjTrackLab[i].size(), 0);

            if(TrackTimeTmp.size() < 2)
            {
                ObjTrackTime.push_back(TrackTimeTmp);
                continue;
            }

            for(int j=1; j<TrackTimeTmp.size(); ++j)
            {
                TrackTimeTmp[j] = TrackCount[ObjTrackLab[i][j] - 1];
            }
            ObjTrackTime.push_back(TrackTimeTmp);
        }

        return ObjTrackTime;
    }

    vector<vector<pair<int, int>>> Tracking::GetDynamicTrack()
    {
        vector<vector<cv::KeyPoint>> Feats = mpMap->vpFeatDyn;
        vector<vector<int>> ObjLab = mpMap->vnFeatLabel;
        int N = Feats.size();

        vector<vector<pair<int, int>>> TrackLets;
        //save object id of each tracklets
        vector<int> ObjectID;
        //save the track id in TrackLEts for precious frame and current frame
        vector<int> TrackCheck_pre;

        int IDsofer = 0;
        for(int i=0; i < N; ++i)
        {
            vector<int> TrackCheck_cur(Feats[i].size(), -1);

            if(Feats[i].empty())
            {
                TrackCheck_pre = TrackCheck_cur;
                continue;
            }

            if(i==0)
            {
                int M = Feats[i].size();
                for(int j=0; j<N; j=j+2)
                {
                    vector<pair<int, int>> TraLet(2);
                    TraLet[0] = make_pair(i, j);
                    TraLet[1] = make_pair(i, j+1);
                    TrackLets.push_back(TraLet);
                    ObjectID.push_back(ObjLab[i][j/2]);

                    TrackCheck_cur[j+1] = IDsofer;
                    IDsofer = IDsofer + 1;
                }
            }
            else
            {
                int M_pre = TrackCheck_pre.size();
                int M_cur = Feats[i].size();

                if(M_pre == 0)
                {
                    for(int j=0; j<M_cur; j=j+2)
                    {
                        vector<pair<int, int>> TraLet(2);
                        TraLet[0] = make_pair(i, j);
                        TraLet[1] = make_pair(i, j+1);
                        TrackLets.push_back(TraLet);
                        ObjectID.push_back(ObjLab[i][j/2]);

                        TrackCheck_cur[j+1] = IDsofer;
                        IDsofer = IDsofer + 1;
                    }
                }
                else
                {
                    vector<int> TM(M_cur, -1);
                    vector<float> MinDist(M_cur, -1);
                    int nmatches = 0;
                    for(int k=1; k<M_pre; k=k+2)
                    {
                        float x_ = Feats[i-1][k].pt.x;
                        float y_ = Feats[i-1][k].pt.y;
                        float min_dist = 10;
                        int candi = -1;
                        for(int j=0; j<M_cur; j=j+2)
                        {
                            if(ObjLab[i-1][(k-1)/2] != ObjLab[i][j/2])
                            {
                                continue;
                            }
                            float x = Feats[i][j].pt.x;
                            float y = Feats[i][j].pt.y;
                            float dist = sqrt(pow(x_-x, 2) + pow(y_-y, 2));

                            if(dist < min_dist)
                            {
                                min_dist = dist;
                                candi = j;
                            }
                        }

                        if(min_dist < 1.0)
                        {
                            if(TM[candi] == -1 || (TM[candi] != -1 * min_dist < MinDist[candi]))
                            {
                                TM[candi] = k;
                                MinDist[candi] = min_dist;
                                nmatches = nmatches + 1;
                            }
                        }
                    }

                    for(int j=0; j<M_cur; j=j+2)
                    {
                        if(TM[j] != -1)
                        {
                            TrackLets[TrackCheck_pre[TM[j]]].push_back(make_pair(i, j+1));
                            TrackCheck_cur[j+1] = TrackCheck_pre[TM[j]];
                        }
                        else
                        {
                            vector<pair<int, int>> TraLet(2);
                            TraLet[0] = make_pair(i, j);
                            TraLet[1] = make_pair(i, j+1);

                            TrackLets.push_back(TraLet);
                            ObjectID.push_back(ObjLab[i][j/2]);

                            TrackCheck_cur[j+1] = IDsofer;
                            IDsofer = IDsofer + 1;
                        }
                    }

                }
            }

            TrackCheck_pre = TrackCheck_cur;
        }

        mpMap->nObjID = ObjectID;

        cout << endl;
        cout << "==========================================" << endl;
        cout << "the number of object feature tracklets: " << TrackLets.size()<< endl;
        cout << endl;

        vector<int> TrackLength(N, 0);
        for(int i=0; i<TrackLets.size(); ++i)
        {
            TrackLength[TrackLets[i].size()-2]++;
        }

        for(int i=0; i<N; ++i)
        {
            cout << "The length of" << i+2 << " tracklets is found with the amount of" << TrackLength[i] << " ...." << endl;
        }
        cout << endl;

        return TrackLets;
    }

    void Tracking::RenewFrameInfo(const std::vector<int> &TM_sta)
    {
        cout << endl << "Start Renew Frame informaion ......" << endl;

        //use sampled or detected features
        int max_num_sta = nMaxTrackPointBG;
        int max_num_obj = nMaxTrackPointOBJ;

        vector<cv::KeyPoint> mvKeysTmp;
        vector<cv::KeyPoint> mvCorresTmp;
        vector<cv::Point2f> mvFlowNexTmp;
        vector<int> StaInlierIDTmp;

        // save the inliers from last frame
        for(int i=0; i<TM_sta.size(); ++i)
        {
            if(TM_sta[i]==-1)
            {
                continue;
            }

            int x = currentFrame.mvStatKeys[TM_sta[i]].pt.x;
            int y = currentFrame.mvStatKeys[TM_sta[i]].pt.x;

            if(x >= mImGrayLast.cols || y>=mImGrayLast.rows || x <=0 || y<=0)
                continue;

            if(mSegMap.at<int>(y, x) != 0)
                continue;

            if(mDepthMap.at<float>(y, x) > 40 || mDepthMap.at<float>(y, x) <= 0)
                continue;

            float flow_xe = mFlowMap.at<cv::Vec2f>(y, x)[0];
            float flow_ye = mFlowMap.at<cv::Vec2f>(y, x)[1];

            if(flow_xe != 0 && flow_ye !=0)
            {
                if(currentFrame.mvStatKeys[TM_sta[i]].pt.x + flow_xe < mImGrayLast.cols &&
                    currentFrame.mvStatKeys[TM_sta[i]].pt.y + flow_ye < mImGrayLast.rows &&
                    currentFrame.mvStatKeys[TM_sta[i]].pt.x+flow_xe>0 && currentFrame.mvStatKeys[TM_sta[i]].pt.y + flow_ye>0)
                {
                    mvKeysTmp.push_back(currentFrame.mvStatKeys[TM_sta[i]]);
                    mvCorresTmp.push_back(cv::KeyPoint(currentFrame.mvStatKeys[TM_sta[i]].pt.x+flow_xe, currentFrame.mvStatKeys[TM_sta[i]].pt.y+flow_ye, 0, 0, 0, -1));
                    mvFlowNexTmp.push_back(cv::Point2f(flow_xe, flow_ye));
                    StaInlierIDTmp.push_back(TM_sta[i]);
                }
            }

            if(mvKeysTmp.size() > max_num_sta)
                break;
        }

        //save extra key points to make it a fixed number
        int tot_num = mvKeysTmp.size(), start_id = 0, step= 20;
        vector<cv::KeyPoint> mvKeysTmpCheck = mvKeysTmp;
        vector<cv::KeyPoint> mvKeysSample;
        if(nUseSampleFea==1)
            mvKeysSample = currentFrame.mvStatKeysTmp;
        else
            mvKeysSample = currentFrame.mvKeys;
        while(tot_num < max_num_sta)
        {
            if(start_id == step)
                break;
            for(int i=start_id; i<mvKeysSample.size(); i=i+step)
            {
                //check if this key point is already been used
                float min_dist = 100;
                bool used = false;
                for(int j=0; j>mvKeysTmpCheck.size(); ++j)
                {
                    float cur_dist = sqrt(pow(mvKeysTmpCheck[j].pt.x-mvKeysSample[i].pt.x, 2)+pow(mvKeysTmpCheck[j].pt.y-mvKeysSample[i].pt.y, 2));
                    if(cur_dist<min_dist)
                    {
                        min_dist = cur_dist;
                    }
                    if(min_dist < 1.0)
                    {
                        used = true;
                        break;
                    }
                }
                if(used)
                    continue;

                int x = mvKeysSample[i].pt.x;
                int y = mvKeysSample[i].pt.y;

                if(x >= mImGrayLast.cols || y >= mImGrayLast.rows || x<=0 || y<=0)
                    continue;

                if(mSegMap.at<int>(y, x) != 0)
                    continue;
//                if(mDepthMap.at<float>(y, x)>40 || mDepthMap.at<float>(y, x) <= 0)
//                    continue;

                float flow_xe = mFlowMap.at<cv::Vec2f>(y, x)[0];
                float flow_ye = mFlowMap.at<cv::Vec2f>(y, x)[1];

                if(flow_xe != 0 && flow_ye != 0)
                {
                    if(mvKeysSample[i].pt.x+flow_xe<mImGrayLast.cols && mvKeysSample[i].pt.y+flow_ye < mImGrayLast.rows &&
                        mvKeysSample[i].pt.x+flow_xe >0 && mvKeysSample[i].pt.y + flow_ye > 0)
                    {
                        mvKeysTmp.push_back(mvKeysSample[i]);
                        mvCorresTmp.push_back(cv::KeyPoint(mvKeysSample[i].pt.x + flow_xe, mvKeysSample[i].pt.y+flow_ye, 0, 0, 0, -1));
                        mvFlowNexTmp.push_back(cv::Point2f(flow_xe, flow_ye));
                        StaInlierIDTmp.push_back(-1);
                        tot_num = tot_num + 1;
                    }
                }

                if(tot_num >= max_num_sta)
                    break;
            }

            start_id = start_id + 1;
        }

        currentFrame.N_s_tmp = mvKeysTmp.size();

        //assign the depth value to each key point
        vector<float> mvDepthTmp(currentFrame.N_s_tmp, -1);
        for(int i=0; i<currentFrame.N_s_tmp; i++)
        {
            const cv::KeyPoint &kp = mvKeysTmp[i];

            const float &v = kp.pt.y;
            const float &u = kp.pt.x;

            float d = mDepthMap.at<float>(v, u);
            if(d >0)
                mvDepthTmp[i] = d;
        }

        //create 3d point based on key point, depth and pose
        vector<cv::Point3f> mv3DPointTmp(currentFrame.N_s_tmp);
        for(int i=0; i<currentFrame.N_s_tmp; ++i)
        {
            mv3DPointTmp[i] = Converter::toPoint3f(Optimizer::Get3DinWorld(mvKeysTmp[i], mvDepthTmp[i], mK, Converter::toInvMatrix(currentFrame.camPose)));
        }

        //obtain inlier ID
        currentFrame.nStatInlierID = StaInlierIDTmp;

        //update
        currentFrame.mvStatKeysTmp = mvKeysTmp;
        currentFrame.mvStatDepthTmp = mvDepthTmp;
        currentFrame.mvStat3DPointTmp = mv3DPointTmp;
        currentFrame.mvFlowNext = mvFlowNexTmp;
        currentFrame.mvCorres = mvCorresTmp;

       //update for Dynamic object Features-----------------

       vector<cv::KeyPoint> mvObjKeysTmp;
       //vector<float> mvObjDepthTmp;
       vector<cv::KeyPoint> mvObjCorresTmp;
       vector<cv::Point2f> mvObjFlowNextTmp;
       vector<int> vSemObjLabelTmp;
       vector<int> DynInlierIDTmp;
       vector<int> vObjLabelTmp;

       //again, save the inliers from last frame
       vector<vector<int>> ObjInlierSet = currentFrame.vnObjInlierID;
       vector<int> ObjFeaCount(ObjInlierSet.size());
       for(int i=0; i<ObjInlierSet.size(); ++i)
       {
           //remove failure object
           if(!currentFrame.bObjStat[i])
           {
               ObjFeaCount[i] = -1;
               continue;
           }

           int count = 0;
           for(int j=0; j<ObjInlierSet[i].size(); ++j)
           {
               const int x = currentFrame.mvObjKeys[ObjInlierSet[i][j]].pt.x;
               const int y = currentFrame.mvObjKeys[ObjInlierSet[i][j]].pt.y;

               if(x>=mImGrayLast.cols || y>=mImGrayLast.rows || x<=0 || y<=0)
                   continue;
               if(mSegMap.at<int>(y, x) != 0 && mDepthMap.at<float>(y, x) < 25 && mDepthMap.at<float>(y, x)>0)
               {
                   const float flow_x = mFlowMap.at<cv::Vec2f>(y, x)[0];
                   const float flow_y = mFlowMap.at<cv::Vec2f>(y, x)[1];

                   if(x+flow_x < mImGrayLast.cols && y+flow_y < mImGrayLast.rows && x+flow_x > 0 && y+flow_y > 0)
                   {
                       mvObjKeysTmp.push_back(cv::KeyPoint(x, y, 0, 0, 0, -1));
                       //mvObjDepthTmp.push_back(mDepthMap.at<float>(y, x));
                       vSemObjLabelTmp.push_back(mSegMap.at<int>(y, x));
                       mvObjFlowNextTmp.push_back(cv::Point2f(flow_x, flow_y));
                       mvObjCorresTmp.push_back(cv::KeyPoint(x+flow_x, y+flow_y, 0, 0, 0, -1));
                       DynInlierIDTmp.push_back(ObjInlierSet[i][j]);
                       vObjLabelTmp.push_back(currentFrame.objLabel[ObjInlierSet[i][j]]);
                       count = count + 1;
                   }
               }
           }
           ObjFeaCount[i] = count;
       }

       //save extra key points to make each oject having a fixed number
       vector<vector<int>> ObjSet = currentFrame.vnObjID;
       vector<cv::KeyPoint> mvObjKeysTmpCheck = mvObjKeysTmp;
       for(int i=0; i<ObjSet.size(); ++i)
       {
           //remove failure object
           if(!currentFrame.bObjStat[i])
               continue;

           int SemLabel = currentFrame.nSemPosition[i];
           int tot_num = ObjFeaCount[i];
           int start_id = 0, step = 15;
           while(tot_num < max_num_obj)
           {
               if(start_id==step)
               {
                   break;
               }

               for(int j=start_id; j<mvTmpSemObjLabel.size(); j=j+step)
               {
                   //check the semantic label if it is the same
                   if(mvTmpSemObjLabel[j] != SemLabel)
                       continue;

                   //check if this key point is already been used
                   float min_dst = 100;
                   bool used = false;
                   for(int k=0; k<mvObjKeysTmpCheck.size(); ++k)
                   {
                       float cur_dist = sqrt(pow(mvObjKeysTmpCheck[k].pt.x-mvTmpObjKeys[j].pt.x, 2) + pow(mvObjKeysTmpCheck[k].pt.y-mvTmpObjKeys[j].pt.y, 2));
                       if(cur_dist < min_dst)
                           min_dst = cur_dist;
                       if(min_dst < 1.0)
                       {
                           used = true;
                           break;
                       }
                   }
                   if(used)
                       continue;

                   //save the found one
                   mvObjKeysTmp.push_back(mvTmpObjKeys[j]);
                   //mvObjDepthTmp.push_back(mvTmpObjectDepth[j]);
                   vSemObjLabelTmp.push_back(mvTmpSemObjLabel[j]);
                   mvObjFlowNextTmp.push_back(mvTmpObjFlowNext[j]);
                   mvObjCorresTmp.push_back(mvTmpObjCorres[j]);
                   DynInlierIDTmp.push_back(-1);
                   vObjLabelTmp.push_back(currentFrame.nModLabel[i]);
                   tot_num = tot_num + 1;

                   if(tot_num >= max_num_obj)
                   {
                       break;
                   }
               }
               start_id = start_id + 1;
           }
       }

       //update new appearing objects
       //find the unique labels in semantic label
       auto UniLab = mvTmpSemObjLabel;
       sort(UniLab.begin(), UniLab.end());
       UniLab.erase(unique(UniLab.begin(), UniLab.end()), UniLab.end());
       //find new appearing label
       vector<bool> NewLab(UniLab.size(), false);
       for(int i=0; i<currentFrame.nSemPosition.size(); ++i)
       {
           int CurSemLabel = currentFrame.nSemPosition[i];
           for(int j=0; j<UniLab.size(); ++j)
           {
               if(UniLab[j]==CurSemLabel && currentFrame.bObjStat[i])
               {
                   NewLab[j] = true;
                   break;
               }
           }
       }

       //add the new object key points
       for(int i=0; i<NewLab.size(); ++i)
       {
           if(NewLab[i]==false)
           {
               for(int j=0; j<mvTmpSemObjLabel.size(); j++)
               {
                   if(UniLab[i]==mvTmpSemObjLabel[j])
                   {
                       //save the found one
                       mvObjKeysTmp.push_back(mvTmpObjKeys[j]);
                       //mvObjDepthTmp.push_back(mvTmpObjectDepth[j]);
                       vSemObjLabelTmp.push_back(mvTmpSemObjLabel[j]);
                       mvObjFlowNextTmp.push_back(mvTmpObjFlowNext[j]);
                       mvObjCorresTmp.push_back(mvTmpObjCorres[j]);
                       DynInlierIDTmp.push_back(-1);
                       vObjLabelTmp.push_back(-2);
                   }
               }
           }
       }

       //create 3d point based on key point, depth and pose
       vector<cv::Point3f> mvObj3DPointTmp(mvObjKeysTmp.size());
       for(int i=0; i<mvObjKeysTmp.size(); ++i)
       {
           //mvObj3DPointTmp[i] = Optimizer::Get3DinWorld(mvObjKeysTmp[i], mvObjDepthTmp[i], mK, Converter::toInvMatrix(currentFrame.camPose));
       }

       //update
       currentFrame.mvObjKeys = mvObjKeysTmp;
       //currentFrame.mvObjDepth = mvObjDepthTmp;
       currentFrame.mvObj3DPoint = mvObj3DPointTmp;
       currentFrame.mvObjCorres = mvObjCorresTmp;
       currentFrame.mvObjFlowNext = mvObjFlowNextTmp;
       currentFrame.vSemLabelTmp = vSemObjLabelTmp;
       currentFrame.nDynInlierID = DynInlierIDTmp;
       currentFrame.objLabel = vObjLabelTmp;

       cout << "Renew Frame Info, Done !" << endl;
    }

    void Tracking::UpdateMask()
    {
        cout << "Update Mask ......." << endl;

        //find the unique labels in semantic label
        auto UniLab = mLastFrame.semObjLabel;
        sort(UniLab.begin(), UniLab.end());
        UniLab.erase(unique(UniLab.begin(), UniLab.end()), UniLab.end());

        //collect the predict labels and semantic labels in levels
        vector<vector<int>> ObjID(UniLab.size());
        for(int i=0; i<mLastFrame.semObjLabel.size(); ++i)
        {
            //save object label
            for(int j=0; j<UniLab.size(); ++j)
            {
                if(mLastFrame.semObjLabel[i] == UniLab[j]){
                    ObjID[j].push_back(i);
                    break;
                }
            }
        }

        //check each object label distribution in the coming frame
        for(int i=0; i<ObjID.size(); ++i)
        {
            //collect labels
            vector<int> LabTmp;
            for(int j=0; j<ObjID[i].size(); ++j)
            {
                const int u = mLastFrame.mvObjCorres[ObjID[i][j]].pt.x;
                const int v = mLastFrame.mvObjCorres[ObjID[i][j]].pt.y;
                if(u<mImGray.cols && u>0 && v<mImGray.rows && v>0)
                {
                    LabTmp.push_back(mSegMap.at<int>(v, u));
                }
            }

            if(LabTmp.size()>100)
                continue;

            // find label that appears most in labTmp
            //count duplicates
            map<int, int> dups;
            for(int k : LabTmp)
                ++dups[k];

            // and sort them by descending order
            vector<pair<int, int>> sorted;
            for(auto k : dups)
                sorted.push_back(make_pair(k.first, k.second));
            sort(sorted.begin(), sorted.end(), SortPairInt);

            //remove the missing mask
            if(sorted[0].first==0)
            {
                for(int j=0; j>mImGrayLast.rows; j++)
                {
                    for(int k=0; k<mImGrayLast.cols; k++)
                    {
                        if(mSegMapLast.at<int>(j, k) == UniLab[i])
                        {
                            const int flow_x = mFlowMapLast.at<cv::Vec2f>(j, k)[0];
                            const int flow_y = mFlowMapLast.at<cv::Vec2f>(j, k)[1];

                            if(k+flow_x < mImGrayLast.cols && k+flow_x > 0 && j+flow_y < mImGrayLast.rows && j+flow_y > 0)
                                mSegMap.at<int>(j+flow_y, k+flow_x) = UniLab[i];
                        }
                    }
                }
            }
            //end of recovery
        }

        cout << "Update Mask, Done!" << endl;
    }

    void Tracking::GetMetricError(const vector<cv::Mat> &CamPose, vector<vector<cv::Mat>> &RigMot,
                                  const vector<vector<cv::Mat>> &ObjPosePre, const vector<cv::Mat> &CamPose_gt,
                                  const vector<vector<cv::Mat>> &RigMot_gt, const vector<vector<bool>> &ObjStat)
                                  {
        bool bRMSError = false;
        cout << "=================================================" << endl;

        //absolute trajectory error for camera(RMSE)

        cout << "CAMERA:" << endl;
        float t_sum = 0, r_sum = 0;
        for(int i=0; i<CamPose.size(); ++i)
        {
            cv::Mat T_lc_inv = CamPose[i]*Converter::toInvMatrix(CamPose[i-1]);
            cv::Mat T_lc_gt = CamPose_gt[i-1]*Converter::toInvMatrix(CamPose_gt[i]);
            cv::Mat ate_cam = T_lc_inv*T_lc_gt;

            //translation
            float t_ate_cam = sqrt(pow(ate_cam.at<float>(0, 3), 2) + pow(ate_cam.at<float>(1, 3), 2) + pow(ate_cam.at<float>(2, 3), 2));
            if(bRMSError)
                t_sum = t_sum + t_ate_cam*t_ate_cam;
            else
                t_sum = t_sum + t_ate_cam;

            //rotation
            float trace_ate = 0;
            for(int j=0; j<3; ++j)
            {
                if(ate_cam.at<float>(j, j)>1.0)
                    trace_ate = trace_ate + 1.0 -(ate_cam.at<float>(j, j)- 1.0);
                else
                    trace_ate = trace_ate + ate_cam.at<float>(j, j);
            }
            float r_ate_cam = acos((trace_ate-1.0)/2.0)*180.0 / CV_PI;
            if(bRMSError)
                r_sum = r_sum + r_ate_cam*r_ate_cam;
            else
                r_sum = r_sum + r_ate_cam;

        }
        if(bRMSError)
        {
            t_sum = sqrt(t_sum/(CamPose.size()-1));
            r_sum = sqrt(r_sum/(CamPose.size()-1));
        }
        else
        {
            t_sum = t_sum/(CamPose.size() - 1);
            r_sum = r_sum/(CamPose.size() - 1);
        }

        cout << "average error (Camera):" << "t: " << t_sum << "R: " << r_sum << endl;

        vector<float> each_obj_t(max_id-1, 0);
        vector<float> each_obj_r(max_id-1, 0);
        vector<int> each_obj_count(max_id-1, 0);

        //all motion error for OBJECTS(mean error)
        cout << "OBJECTS:" << endl;
        float r_rpe_sum = 0, t_rpe_sum = 0, obj_count = 0;
        for(int i=0; i < RigMot[i].size(); ++i)
        {
            if(RigMot[i].size()>1)
            {
                for(int j=1; j<RigMot[i].size(); ++j)
                {
                    if(!ObjStat[i][j])
                    {
                        cout << "(" << mpMap->vnRMLabel[i][j] << ")" << "is a failure case" << endl;
                        continue;
                    }

                    cv::Mat RigMotBody = Converter::toInvMatrix(ObjPosePre[i][j])*RigMot[i][j]*ObjPosePre[i][j];
                    cv::Mat rpe_obj = Converter::toInvMatrix(RigMotBody)*RigMot_gt[i][j];

                    //translation error
                    float t_rpe_obj = sqrt(pow(rpe_obj.at<float>(0, 3), 2)+ pow(rpe_obj.at<float>(1, 3), 2) + pow(rpe_obj.at<float>(2, 3), 2));
                    if(bRMSError)
                    {
                        each_obj_t[mpMap->vnRMLabel[i][j] -1 ] = each_obj_t[mpMap->vnRMLabel[i][j] -1] + t_rpe_obj*t_rpe_obj;
                        t_rpe_sum = t_rpe_sum + t_rpe_obj*t_rpe_sum;
                    }
                    else{
                        each_obj_t[mpMap->vnRMLabel[i][j]-1] = each_obj_t[mpMap->vnRMLabel[i][j] -1 ] + t_rpe_obj;
                        t_rpe_sum = t_rpe_sum + t_rpe_obj;
                    }

                    //rotatoin error
                    float trace_rpe = 0;
                    for(int k=0; k<3; ++k)
                    {
                        if(rpe_obj.at<float>(k, k) > 1.0)
                            trace_rpe = trace_rpe + 1.0 - (rpe_obj.at<float>(k, k)-1.0);
                        else
                            trace_rpe = trace_rpe + rpe_obj.at<float>(k, k);
                    }
                    float r_rpe_obj = acos((trace_rpe - 1.0)/2.0) * 180.0 /CV_PI;
                    if(bRMSError){
                        each_obj_r[mpMap->vnRMLabel[i][j] - 1] = each_obj_r[mpMap->vnRMLabel[i][j] -1]+r_rpe_obj*r_rpe_obj;
                        r_rpe_sum = r_rpe_sum + r_rpe_obj*r_rpe_obj;
                    }
                    else{
                        each_obj_r[mpMap->vnRMLabel[i][j]-1] = each_obj_r[mpMap->vnRMLabel[i][j]-1] + r_rpe_obj;
                        r_rpe_sum = r_rpe_sum + r_rpe_obj;
                    }

                    obj_count++;
                    each_obj_count[mpMap->vnRMLabel[i][j]-1] = each_obj_count[mpMap->vnRMLabel[i][j]-1]+1;
                }
            }
        }
        if(bRMSError)
        {
            t_rpe_sum = sqrt(t_rpe_sum/obj_count);
            r_rpe_sum = sqrt(t_rpe_sum/obj_count);
        }
        else
        {
            t_rpe_sum = t_rpe_sum/obj_count;
            r_rpe_sum = r_rpe_sum/obj_count;
        }

        cout << "average error (Over All Objects):" << " t: " << t_rpe_sum << " R: " <<r_rpe_sum << endl;

        //show each object
        for(int i=0; i<each_obj_count.size(); ++i)
        {
            if(bRMSError)
            {
                each_obj_t[i] = sqrt(each_obj_t[i]/each_obj_count[i]);
                each_obj_t[i] = sqrt(each_obj_r[i]/each_obj_count[i]);
            }
            else
            {
                each_obj_t[i] = each_obj_t[i]/each_obj_count[i];
                each_obj_r[i] = each_obj_r[i]/each_obj_count[i];
            }
            if(each_obj_count[i] >= 3)
                cout << endl << "average error of Object " << i+1 << ": " << " t: " << each_obj_r[i] << endl;
        }

        cout << "======================================================" << endl;
    }

    void Tracking::PlotMetricError(const vector<cv::Mat> &CamPose, const vector<vector<cv::Mat>> &RigMot,
                                   const vector<vector<cv::Mat>> &ObjPosePre, const vector<cv::Mat> &CamPse_gt,
                                   const vector<vector<cv::Mat>> &RigMot_gt, const vector<vector<bool>> &ObjStat) {
        //saved evaluated errors
        vector<float> CamPotErr(CamPose.size()-1);
        vector<float> CamTraErr(CamPose.size() -1);
        vector<vector<float>> ObjRotErr(max_id - 1);
        vector<vector<float>> ObjTraErr(max_id - 1);

        bool bRMSError = false, bAccumError = true;
        cout << "=======================================================" << endl;

        //absolute trajectory error for CAMERA (RMSE)
        cout << "CAMERA:" << endl;
        float t_sum = 0, r_sum = 0;
        for(int i=1; i<CamPose.size(); ++i)
        {
            cv::Mat T_lc_inv = CamPose[i]*Converter::toInvMatrix(CamPose[i-1]);
            cv::Mat T_lc_gt = CamPse_gt[i-1]*Converter::toInvMatrix(CamPse_gt[i]);
            cv::Mat ate_cam = T_lc_inv*T_lc_gt;

            //translation
            float t_ate_cam = sqrt(pow(ate_cam.at<float>(0, 3), 2) + pow(ate_cam.at<float>(1, 3), 2)  + pow(ate_cam.at<float>(2, 3), 2));
            if(bRMSError)
                t_sum = t_sum + t_ate_cam * t_ate_cam;
            else
                t_sum = t_sum + t_ate_cam;

            //rotation
            float trace_ate = 0;
            for (int j = 0; j < 3; ++j) {
                if(ate_cam.at<float>(j, j) > 1.0)
                    trace_ate = trace_ate + 1.0 -(ate_cam.at<float>(j, j)-1.0);
                else
                    trace_ate = trace_ate + ate_cam.at<float>(j, j);
            }
            float r_ate_cam = acos((trace_ate - 1.0)/2.0)*180.0/CV_PI;
            if(bRMSError)
                r_sum = r_sum + r_ate_cam*r_ate_cam;
            else
                t_sum = t_sum + t_ate_cam;
            if(bAccumError)
            {
                CamPotErr[i-1] = r_ate_cam/i;
                CamTraErr[i-1] = t_ate_cam/i;
            }
            else
            {
                CamPotErr[i-1] = r_ate_cam;
                CamTraErr[i-1] = t_ate_cam;
            }
        }
        if(bRMSError)
        {
            t_sum = sqrt(t_sum/(CamPose.size()-1));
            r_sum = sqrt(r_sum/(CamPose.size()-1));
        } else{
            t_sum = t_sum/(CamPose.size()-1);
            r_sum = r_sum/(CamPose.size()-1);
        }

        cout << "average error(Camera):" << " t: " << t_sum << " R: " << r_sum << endl;

        vector<float> each_obj_t(max_id-1, 0);
        vector<float> each_obj_r(max_id-1, 0);
        vector<float> each_obj_count(max_id-1, 0);

        //all motion error for OBJECTS(mean error)
        cout << "OBJECTS: " << endl;
        float r_rpe_sum = 0, t_rpe_sum = 0, obj_count = 0;
        for(int i=0; i<RigMot.size(); ++i)
        {
            if(RigMot[i].size()>1)
            {
                for(int j=1; j<RigMot[i].size(); ++j)
                {
                    if(!ObjStat[i][j])
                    {
                        cout << "(" << mpMap->vnRMLabel[i][j] << ")" << " is a failure case." << endl;
                        continue;
                    }

                    cv::Mat RigMotBody = Converter::toInvMatrix(ObjPosePre[i][j])*RigMot[i][j]*ObjPosePre[i][j];
                    cv::Mat rpe_obj = Converter::toInvMatrix(RigMotBody) * RigMot_gt[i][j];

                    //translation error
                    float t_rpe_obj = sqrt(pow(rpe_obj.at<float>(0, 3), 2)+ pow(rpe_obj.at<float>(1, 3), 2)+pow(rpe_obj.at<float>(2, 3), 2));
                    if(bRMSError)
                    {
                        each_obj_t[mpMap->vnRMLabel[i][j]-1] = each_obj_t[mpMap->vnRMLabel[i][j]-1] + t_rpe_obj;
                        t_rpe_sum = t_rpe_sum + t_rpe_obj;
                    }
                    else{
                        each_obj_t[mpMap->vnRMLabel[i][j]-1] = each_obj_t[mpMap->vnRMLabel[i][j]-1] + t_rpe_obj;
                        t_rpe_sum = t_rpe_sum + t_rpe_obj;
                    }

                    //rotation error
                    float trace_rpe = 0;
                    for(int k=0; k<3; ++k)
                    {
                        if(rpe_obj.at<float>(k, k)>1.0)
                            trace_rpe = trace_rpe+1.0 -(rpe_obj.at<float>(k, k)-1.0);
                        else
                            trace_rpe = trace_rpe + rpe_obj.at<float>(k, k);
                    }
                    float r_rpe_obj = acos((trace_rpe-1.0)/2.0)*180.0/CV_PI;
                    if(bRMSError)
                    {
                        each_obj_r[mpMap->vnRMLabel[i][j]-1] = each_obj_r[mpMap->vnRMLabel[i][j]-1] + r_rpe_obj*r_rpe_obj;
                        r_rpe_sum = r_rpe_sum + r_rpe_obj;
                    }
                    else{
                        each_obj_r[mpMap->vnRMLabel[i][j]-1] = each_obj_r[mpMap->vnRMLabel[i][j]-1] + r_rpe_obj;
                        r_rpe_sum = r_rpe_sum + r_rpe_obj;
                    }

                    obj_count ++;
                    each_obj_count[mpMap->vnRMLabel[i][j]-1] = each_obj_count[mpMap->vnRMLabel[i][j]-1] + 1;
                    if(bAccumError)
                    {
                        ObjTraErr[mpMap->vnRMLabel[i][j]-1].push_back(each_obj_t[mpMap->vnRMLabel[i][j]-1]/each_obj_count[mpMap->vnRMLabel[i][j]-1]);
                        ObjRotErr[mpMap->vnRMLabel[i][j]-1].push_back(each_obj_r[mpMap->vnRMLabel[i][j]-1]/each_obj_count[mpMap->vnRMLabel[i][j]-1]);
                    }
                    else
                    {
                        ObjTraErr[mpMap->vnRMLabel[i][j]-1].push_back(t_rpe_obj);
                        ObjRotErr[mpMap->vnRMLabel[i][j]-1].push_back(r_rpe_obj);
                    }
                }
            }
        }
        if(bRMSError)
        {
            t_rpe_sum = sqrt(t_rpe_sum/obj_count);
            r_rpe_sum = sqrt(r_rpe_sum/obj_count);
        }
        else
        {
            t_rpe_sum = t_rpe_sum/obj_count;
            r_rpe_sum = r_rpe_sum/obj_count;
        }

        cout << "average error (Over All Objects):" << " t: " << t_rpe_sum << " R: " << r_rpe_sum << endl;

        //show each object
        for(int i=0; i<each_obj_count.size(); ++i)
        {
            if(bRMSError)
            {
                each_obj_t[i] = sqrt(each_obj_t[i]/each_obj_count[i]);
                each_obj_r[i] = sqrt(each_obj_r[i]/each_obj_count[i]);
            }
            else
            {
                each_obj_t[i] = each_obj_t[i]/each_obj_count[i];
                each_obj_r[i] = each_obj_r[i]/each_obj_count[i];
            }
            if(each_obj_count[i]>=3)
                cout << endl << "average error of Object " << i+1 << ": " << " t: " << each_obj_t[i] << " R: " << each_obj_r[i] << endl;

        }
        cout << "============================================" << endl;

        auto  name1 = "Translation";
        cvplot::setWindowTitle(name1, "Translation Error (Meter)");
        cvplot::moveWindow(name1, 0, 240);
        cvplot::resizeWindow(name1, 800, 240);
        auto &figure1 = cvplot::figure(name1);

        auto name2 = "Rotation";
        cvplot::setWindowTitle(name2, "Rotation Error (Degree)");
        cvplot::resizeWindow(name2, 800, 240);
        auto &figure2 = cvplot::figure(name2);

        figure1.series("Camera")
            .setValue(CamTraErr)
            .type(cvplot::DotLine)
            .color(cvplot::Red);

        figure2.series("Camera")
            .setValue(CamPotErr)
            .type(cvplot::DotLine)
            .color(cvplot::Red);

        for(int i=0; i<max_id-1; ++i)
        {
            switch (i) {
                case 0:
                    figure1.series("Object "+std::to_string(i+1))
                        .setValue(ObjTraErr[i])
                        .type(cvplot::DotLine)
                        .color(cvplot::Purple);
                    figure2.series("Object " + std::to_string(i+1))
                        .setValue(ObjRotErr[i])
                        .type(cvplot::DotLine)
                        .color(cvplot::Purple);
                    break;
                case 1:
                    figure1.series("Object "+std::to_string(i+1))
                        .setValue(ObjTraErr[i])
                        .type(cvplot::DotLine)
                        .color(cvplot::Green);
                    figure2.series("Object "+std::to_string(i+1))
                        .setValue(ObjRotErr[i])
                        .type(cvplot::DotLine)
                        .color(cvplot::Green);
                    break;
                case 2:
                    figure1.series("Object "+std::to_string(i+1))
                        .setValue(ObjTraErr[i])
                        .type(cvplot::DotLine)
                        .color(cvplot::Cyan);
                    figure2.series("Object " + std::to_string(i+1))
                        .setValue(ObjRotErr[i])
                        .type(cvplot::DotLine)
                        .color(cvplot::Cyan);
                    break;
                case 3:
                    figure1.series("Object " + std::to_string(i + 1))
                        .setValue(ObjTraErr[i])
                        .type(cvplot::DotLine)
                        .color(cvplot::Blue);
                    figure2.series("Object " + std::to_string(i+1))
                        .setValue(ObjRotErr[i])
                        .type(cvplot::DotLine)
                        .color(cvplot::Blue);
                    break;
                case 4:
                    figure1.series("Object " + std::to_string((i+1)))
                        .setValue(ObjTraErr[i])
                        .type(cvplot::DotLine)
                        .color(cvplot::Pink);
                    figure2.series("Object" + std::to_string(i+1))
                        .setValue(ObjRotErr[i])
                        .type(cvplot::DotLine)
                        .color(cvplot::Pink);
                    break;
            }
        }

        figure1.show(true);
        figure2.show(true);
    }

    void Tracking::GetVelocityError(const vector<vector<cv::Mat>> &RigMot, const vector<vector<cv::Mat>> &PointDyn,
                                    const vector<vector<int>> &FeaLab, const vector<vector<int>> &RMLab,
                                    const vector<vector<float>> &Velo_gt, const vector<vector<int>> &TmpMatch,
                                    const vector<vector<bool>> &ObjStat) {
        bool bRMSError = true;
        float s_sum = 0, s_gt_sum = 0, obj_count = 0;

        string path = "/Users/alexwei/Evaluation/ijrr2021/";
        string path_sp_e = path + "speed_error.txt";
        string path_sp_est = path + "speed_estimation.txt";
        string path_sp_gt = path + "speed_groundtruth.txt";
        string path_track = path + "tracking_id.txt";
        ofstream save_sp_e, save_sp_est, save_sp_gt, save_tra;
        save_sp_e.open(path_sp_e.c_str(), ios::trunc);
        save_sp_est.open(path_sp_est.c_str(), ios::trunc);
        save_sp_gt.open(path_sp_gt.c_str(), ios::trunc);
        save_tra.open(path_track.c_str(), ios::trunc);

        vector<float> each_obj_est(max_id-1, 0);
        vector<float> each_obj_gt(max_id-1, 0);
        vector<float> each_obj_count(max_id-1, 0);

        cout << "OBJECTS SPEED:" << endl;

        for(int i=0; i<RigMot.size(); ++i)
        {
            save_tra  << i << " " << 0 << " ";

            //check if there are moving onjects, and if all the variables are consistent
            if(RigMot[i].size()>1 && Velo_gt[i].size()>1 &&RMLab[i].size()>1)
            {
                //loop for each object in each frame
                for(int j=1; i<RigMot[i].size(); ++j)
                {
                    // check if this is valid object estimate
                    if(!ObjStat[i][j]) {
                        cout << "(" << mpMap->vnRMLabel[i][j] << ")" << " is a failure case" << endl;
                        continue;
                    }

                    //compute each object centroid
                    cv::Mat ObjCenter = (cv::Mat_<float>(3, 1)  << 0.f, 0.f, 0.f);
                    float ObjFeaCount = 0;
                    if(i==0)
                    {
                        for(int k=0; k<PointDyn[i+1].size(); ++k)
                        {
                            if(FeaLab[i][k]!=RMLab[i][j])
                                continue;
                            if(TmpMatch[i][k] == -1)
                                continue;

                            ObjCenter = ObjCenter + PointDyn[i][TmpMatch[i][k]];
                            ObjFeaCount = ObjFeaCount + 1;
                        }
                        ObjCenter = ObjCenter/ObjFeaCount;
                    }
                    else
                    {
                        for(int k=0; i<PointDyn[i+1].size(); ++k)
                        {
                            if(FeaLab[i][k] != RMLab[i][j])
                                continue;
                            if(TmpMatch[i][k]== -1)
                                continue;
                            ObjCenter = ObjCenter + PointDyn[i][TmpMatch[i][k]];
                            ObjFeaCount = ObjFeaCount + 1;
                        }
                        ObjCenter = ObjCenter/ObjFeaCount;
                    }

                    //compute object velocity
                    cv::Mat sp_est_v = RigMot[i][j].rowRange(0, 3).col(3) - (cv::Mat::eye(3, 3, CV_32F)-RigMot[i][j].rowRange(0, 3))*ObjCenter;
                    float sp_est_norm = sqrt(pow(sp_est_v.at<float>(0), 2) + pow(sp_est_v.at<float>(1), 2) + pow(sp_est_v.at<float>(2), 2)) *36;

                    //compute velocity error
                    float speed_error = sp_est_norm - Velo_gt[i][j];
                    if(bRMSError)
                    {
                        each_obj_est[mpMap->vnRMLabel[i][j]-1] = each_obj_est[mpMap->vnRMLabel[i][j]-1] + sp_est_norm*sp_est_norm;
                        each_obj_gt[mpMap->vnRMLabel[i][j] - 1] = each_obj_gt[mpMap->vnRMLabel[i][j]-1] + Velo_gt[i][j]*Velo_gt[i][j];
                        s_sum = s_sum + speed_error;
                    }
                    else{
                        each_obj_est[mpMap->vnRMLabel[i][j] - 1] = each_obj_est[mpMap->vnRMLabel[i][j]-1] + sp_est_norm;
                        each_obj_gt[mpMap->vnRMLabel[i][j]-1] = each_obj_gt[mpMap->vnRMLabel[i][j] -1] + Velo_gt[i][j];
                        s_sum = s_sum + speed_error*speed_error;
                    }

                    // sum ground truth speed
                    s_gt_sum = s_gt_sum + Velo_gt[i][j];

                    save_sp_e << fixed << setprecision(6) << speed_error << endl;
                    save_sp_est << fixed << setprecision(6) << sp_est_norm << endl;
                    save_sp_gt << fixed << setprecision(6) << Velo_gt[i][j] << endl;
                    save_tra << mpMap->vnRMLabel[i][j] << " ";

                    obj_count = obj_count + 1;
                    each_obj_count[mpMap->vnRMLabel[i][j]-1] = each_obj_count[mpMap->vnRMLabel[i][j]-1] + 1;

                }
                save_tra << endl;

            }
        }

        save_sp_e.close();
        save_sp_est.close();
        save_sp_gt.close();

        if(bRMSError)
            s_sum = sqrt(s_sum/obj_count);
        else
            s_sum = abs(s_sum/obj_count);

        s_gt_sum = s_gt_sum / obj_count;

        cout << "average speed error : " << " s: " << s_sum << "km/h" << "Track Num: " << (int)obj_count << "GT AVG SPEED: " << s_gt_sum << endl;

        for(int i=0; i<each_obj_count.size(); ++i)
        {
            if(bRMSError)
            {
                each_obj_est[i] = sqrt(each_obj_est[i]/each_obj_count[i]);
                each_obj_gt[i] = sqrt(each_obj_gt[i]/each_obj_count[i]);
            }
            else{
                each_obj_est[i] = each_obj_est[i]/each_obj_count[i];
                each_obj_gt[i] = each_obj_gt[i]/each_obj_count[i];
            }
            if(mpMap->nObjTraCount[i] >= 3)
                cout << endl << "average error of Object " << i+1 << " (" << mpMap->nObjTraCount[i] << "/" << mpMap->nObjTraCountGT[i] << "/" << mpMap->nObjTraSemLab[i] << "): " << " (gt) " << each_obj_gt[i] << endl;
        }

        cout << "=====================================================" << endl << endl;
    }

}