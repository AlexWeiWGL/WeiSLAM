
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/flann.hpp>

#include "Frame.h"
#include "Converter.h"
#include <thread>
#include <chrono>

namespace WeiSLAM {
    using namespace std;

    long unsigned int Frame::nNextId = 0;
    bool Frame::initialComputations = true;
    float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
    float Frame::minX, Frame::minY, Frame::maxX, Frame::maxy;
    float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

    Frame::Frame() {}

    Frame::Frame(const Frame &frame)
            : imORBextractorLeft(frame.imORBextractorLeft), imORBextractorRight(frame.imORBextractorRight),
              timeStamp(frame.timeStamp),
              K(frame.K.clone()), distCoef(frame.distCoef.clone()), mbf(frame.mbf), mb(frame.mb),
              thDepth(frame.thDepth), thDepthObj(frame.thDepthObj),
              N(frame.N), mvKeys(frame.mvKeys), mvKeysRight(frame.mvKeysRight), mvKeysUh(frame.mvKeysUh),
              mvuRight(frame.mvuRight), mvDepth(frame.mvDepth),
              descriptor(frame.descriptor.clone()), descriptorsRight(frame.descriptorsRight.clone()),
              mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
              scaleFactor(frame.scaleFactor), scaleLevels(frame.scaleLevels), logScaleFactor(frame.logScaleFactor),
              scaleFactors(frame.scaleFactors),
              invScaleFactors(frame.invScaleFactors), invLevelSigma2(frame.invLevelSigma2),
              levelSigma2(frame.levelSigma2), mTcw_gt(frame.mTcw_gt),
              vObjPose_gt(frame.vObjPose_gt), nSemPosi_gt(frame.nSemPosi_gt), objLabel(frame.objLabel),
              nModLabel(frame.nModLabel), nSemPosition(frame.nSemPosition),
              bObjStat(frame.bObjStat), objMod(frame.objMod), mvCorres(frame.mvCorres), mvObjCorres(frame.mvObjCorres),
              mvFlowNext(frame.mvFlowNext), mvObjFlowNext(frame.mvObjFlowNext) {
        for (int i = 0; i < FRAME_GRID_COLS; i++) {
            for (int j = 0; j < FRAME_GRID_ROWS; j++) {
                mGrid[i][j] = frame.mGrid[i][j];
            }
        }
        if (!frame.camPose.empty())
            SetPose(frame.camPose);
    }

    Frame::Frame(const cv::Mat &imGray, const cv::Mat &imFlow, const cv::Mat &maskSEM,
                 const double &timeStamp, ORBextractor *extractor, cv::Mat &K, cv::Mat &distCoef, const float &bf,
                 const float &thDepth, const float &theDepthObj, const int &UseSampleFea)
            : imORBextractorLeft(extractor), imORBextractorRight(static_cast<ORBextractor *>(NULL)),
              timeStamp(timeStamp), K(K.clone()), distCoef(distCoef.clone()), mbf(bf), thDepthObj(theDepthObj),
              thDepth(thDepth) {
        cout << "Start Constructing Frame ........" << endl;

        mnId = nNextId++;

        scaleLevels = imORBextractorLeft->GetLevels();
        scaleFactor = imORBextractorLeft->GetScaleFactor();
        logScaleFactor = log(scaleFactor);
        scaleFactors = imORBextractorLeft->GetScaleFactors();
        invScaleFactors = imORBextractorLeft->GetInverseScaleFactors();
        levelSigma2 = imORBextractorLeft->GetScaleSigmaSquares();
        invLevelSigma2 = imORBextractorLeft->GetInverseScaleSigmaSquares();

        ExtractORB(0, imGray);

        N = mvKeys.size();

        if (mvKeys.empty())
            return;

        if (UseSampleFea == 0) {
            for (int i = 0; i < mvKeys.size(); ++i) {
                int x = mvKeys[i].pt.x;
                int y = mvKeys[i].pt.y;

                if (maskSEM.at<int>(y, x) != 0)
                    continue;

                float flow_xe = imFlow.at<cv::Vec2f>(y, x)[0];
                float flow_ye = imFlow.at<cv::Vec2f>(y, x)[1];

                if (flow_xe != 0 && flow_ye != 0) {
                    if (mvKeys[i].pt.x + flow_xe < imGray.cols && mvKeys[i].pt.y + flow_ye < imGray.rows &&
                        mvKeys[i].pt.x < imGray.cols && mvKeys[i].pt.y < imGray.rows) {
                        mvStatKeysTmp.push_back(mvKeys[i]);
                        mvCorres.push_back(cv::KeyPoint(mvKeys[i].pt.x + flow_xe, mvKeys[i].pt.y + flow_ye, 0, 0, 0,
                                                        mvKeys[i].octave, -1));
                        mvFlowNext.push_back(cv::Point2f(flow_xe, flow_ye));
                    }
                }
            }
        } else {
            //use sampled features

            clock_t s_1, e_1;
            double fea_det_time;
            s_1 = clock();
            std::vector<cv::KeyPoint> mvKeysSamp = SampleKeyPoints(imGray.rows, imGray.cols);
            e_1 = clock();

            fea_det_time = (double) (e_1 - s_1) / CLOCKS_PER_SEC * 1000;
            std::cout << "feature detection time: " << fea_det_time << std::endl;

            for (int i = 0; i < mvKeysSamp.size(); ++i) {
                int x = mvKeysSamp[i].pt.x;
                int y = mvKeysSamp[i].pt.y;

                if (maskSEM.at<int>(y, x) != 0)
                    continue;

                float flow_xe = imFlow.at<cv::Vec2f>(y, x)[0];
                float flow_ye = imFlow.at<cv::Vec2f>(y, x)[1];

                if (flow_xe != 0 && flow_ye != 0) {
                    if (mvKeysSamp[i].pt.x + flow_xe < imGray.cols && mvKeysSamp[i].pt.y + flow_ye < imGray.rows &&
                        mvKeysSamp[i].pt.x + flow_xe > 0 && mvKeysSamp[i].pt.y + flow_ye > 0) {
                        mvStatKeysTmp.push_back(mvKeysSamp[i]);
                        mvCorres.push_back(
                                cv::KeyPoint(mvKeysSamp[i].pt.x + flow_xe, mvKeysSamp[i].pt.y + flow_ye, 0, 0, 0,
                                             mvKeysSamp[i].octave, -1));
                        mvFlowNext.push_back(cv::Point2f(flow_xe, flow_ye));
                    }
                }
            }
        }

        N_s_tmp = mvCorres.size();

        //semi-dense features on objects
        int step = 4;
        for (int i = 0; i < imGray.rows; i += step) {
            for (int j = 0; j < imGray.cols; j += step) {
                //check ground truth motion mask
                if (maskSEM.at<int>(i, j) != 0) {
                    const float flow_x = imFlow.at<cv::Vec2f>(i, j)[0];
                    const float flow_y = imFlow.at<cv::Vec2f>(i, j)[1];

                    if (j + flow_x < imGray.cols && j + flow_x > 0 && i + flow_y < imGray.rows && i + flow_y > 0) {
                        //save correspondences
                        mvObjFlowNext.push_back(cv::Point2f(flow_x, flow_y));
                        mvObjCorres.push_back(cv::KeyPoint(j + flow_x, i + flow_y, 0, 0, 0, -1));
                        //save pixel location
                        mvObjKeys.push_back(cv::KeyPoint(j, i, 0, 0, 0, -1));
                        semObjLabel.push_back(maskSEM.at<int>(i, j));
                    }
                }
            }
        }

        UndistortKeyPoints();


        if (initialComputations) {
            ComputeImageBounds(imGray);
            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(maxX - minX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(maxy - minY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            initialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();

        cout << "Constructing Frame, Done!" << endl;
    }

    void Frame::AssignFeaturesToGrid() {
        int nReserve = 0.5 * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
        for (unsigned int i = 0; i < FRAME_GRID_COLS; i++) {
            for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j].reserve(nReserve);
        }

        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeysUh[i];

            int nGridPosX, nGridPosY;
            if (PosInGrid(kp, nGridPosX, nGridPosY))
                mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    void Frame::ExtractORB(int flag, const cv::Mat &im) {
        if (flag == 0) {
            (*imORBextractorLeft)(im, cv::Mat(), mvKeys, descriptor);
        } else {
            (*imORBextractorRight)(im, cv::Mat(), mvKeysRight, descriptor);
        }
    }

    void Frame::SetPose(cv::Mat Tcw) {
        camPose = Tcw.clone();
        UpdatePoseMatrices();
    }

    void Frame::UpdatePoseMatrices() {
        mRcw = camPose.rowRange(0, 3).colRange(0, 3);
        mRwc = mRcw.t();
        mtcw = camPose.rowRange(0, 3).col(3);
        mOw = -mRcw.t() * mtcw;
    }

    vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel,
                                            const int maxLevel) const {
        vector<size_t> vIndices;
        vIndices.reserve(N);

        const int nMinCellX = max(0, (int) floor((x - minX - r) * mfGridElementWidthInv));
        if (nMinCellX >= FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = min((int) FRAME_GRID_COLS - 1, (int) ceil((x - minX + r) * mfGridElementWidthInv));
        if (nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = max(0, (int) floor((y - minY - r) * mfGridElementHeightInv));
        if (nMinCellY >= FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = min((int) FRAME_GRID_ROWS - 1, (int) ceil((y - minY + r) * mfGridElementHeightInv));
        if (nMaxCellY < 0)
            return vIndices;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

        for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
            for (int iy = nMaxCellY; iy <= nMaxCellY; iy++) {
                const vector<size_t> vCell = mGrid[ix][iy];
                if (vCell.empty())
                    continue;

                for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                    const cv::KeyPoint &kpUn = mvKeysUh[vCell[j]];
                    if (bCheckLevels) {
                        if (kpUn.octave < minLevel)
                            continue;
                        if (maxLevel >= 0)
                            if (kpUn.octave > maxLevel)
                                continue;
                    }

                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    if (fabs(distx) < r && fabs(disty) < r) {
                        vIndices.push_back(vCell[j]);
                    }
                }
            }
        }
        return vIndices;
    }

    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
        posX = round((kp.pt.x-minX)*mfGridElementWidthInv);
        posY = round((kp.pt.y-minY)*mfGridElementHeightInv);

        //keypoint's coordinates are undistorted, which could cause to go out of the image
        if(posX<0 || posY>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
            return false;

        return true;
    }

    void Frame::UndistortKeyPoints() {
        if (distCoef.at<float>(0) == 0.0) {
            mvKeysUh = mvKeys;
            return;
        }

        //fill matrix with points
        cv::Mat mat(N, 2, CV_32F);
        for (int i = 0; i < N; i++) {
            mat.at<float>(i, 0) = mvKeys[i].pt.x;
            mat.at<float>(i, 1) = mvKeys[i].pt.y;

        }

        //undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, K, distCoef, cv::Mat(), K);
        mat = mat.reshape(1);

        //Fill undistorted keypoint vector
        mvKeysUh.resize(N);
        for (int i = 0; i < N; i++) {
            cv::KeyPoint kp = mvKeys[i];
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            mvKeysUh[i] = kp;
        }
    }

    void Frame::ComputeImageBounds(const cv::Mat &imLeft) {
        if (distCoef.at<float>(0) != 0.0) {
            cv::Mat mat(4, 2, CV_32F);
            mat.at<float>(0, 0) = 0.0;
            mat.at<float>(0, 1) = 0.0;
            mat.at<float>(1, 0) = imLeft.cols;
            mat.at<float>(1, 1) = 0.0;
            mat.at<float>(2, 0) = 0.0;
            mat.at<float>(2, 1) = imLeft.rows;
            mat.at<float>(3, 0) = imLeft.cols;
            mat.at<float>(3, 1) = imLeft.rows;

            //Undistort corners
            mat = mat.reshape(2);
            cv::undistortPoints(mat, mat, K, distCoef, cv::Mat(), K);
            mat = mat.reshape(1);

            minX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
            maxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
            minY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
            maxy = max(mat.at<float>(2, 1), mat.at<float>(3, 1));
        } else {
            minX = 0.0f;
            maxX = imLeft.cols;
            minY = 0.0f;
            maxy = imLeft.rows;
        }
    }

    cv::Mat Frame::UnprojectStereoObject(const int &i, const bool &addnoise)
    {
        float z = mvObj3DPoint[i].z;

        // used for adding noise
        cv::RNG rng((unsigned)time(NULL));

        if(addnoise){
            float noise = rng.gaussian(z*z/(725*0.5)*0.15);
            z = z + noise;  // sigma = z*0.01
            // z = z + 0.0;
            // cout << "noise: " << noise << endl;
        }

        if(z>0)
        {
            const float u = mvObjKeys[i].pt.x;
            const float v = mvObjKeys[i].pt.y;
            const float x = (u-cx)*z*invfx;
            const float y = (v-cy)*z*invfy;
            cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

            // using ground truth
            const cv::Mat Rlw = camPose.rowRange(0,3).colRange(0,3);
            const cv::Mat Rwl = Rlw.t();
            const cv::Mat tlw = camPose.rowRange(0,3).col(3);
            const cv::Mat twl = -Rlw.t()*tlw;

            return Rwl*x3Dc+twl;
        }
        else{
            cout << "found a depth < 0 ......";
            return cv::Mat();
        }
    }

    vector<cv::KeyPoint> Frame::SampleKeyPoints(const int &rows, const int &cols) {
        cv::RNG rng((unsigned) time(NULL));
        int N = 3000;
        int n_div = 20;
        vector<cv::KeyPoint> KeySave;
        vector<vector<cv::KeyPoint>> KeyinGrid(n_div*n_div);

        int x_step = cols/n_div, y_step=rows/n_div;

        int key_num = 0;
        while (key_num<N)
        {
            for(int i=0; i<n_div; ++i)
            {
                for(int j=0; j<n_div; ++j)
                {
                    const float x = rng.uniform(i*x_step, (i+1)*x_step);
                    const float y = rng.uniform(j*y_step, (j+1)*y_step);

                    if(x>cols || y>=rows || x<=0 || y<=0)
                        continue;

                    cv::KeyPoint Key_tmp = cv::KeyPoint(x, y, 0, 0, 0, -1);
                    KeyinGrid[i*n_div+j].push_back(Key_tmp);
                    key_num = key_num + 1;
                    if(key_num>=N)
                        break;
                }
                if(key_num>=N)
                    break;
            }
        }

        for(int i=0; i<KeyinGrid.size(); ++i)
        {
            for(int j=0; j<KeyinGrid[i].size(); ++j)
            {
                KeySave.push_back(KeyinGrid[i][j]);
            }
        }

        return KeySave;
    }

    cv::Mat Frame::Calculate3D(cv::KeyPoint &vKey1, cv::KeyPoint &vKey2, cv::Mat &pose, cv::Mat K) {
        cv::Mat x3D;

        cv::Mat A(4, 4, CV_32F);
        cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
        K.copyTo(P1.rowRange(0, 3).colRange(0, 3));

        cv::Mat P2(3, 4, CV_32F);
        cv::Mat R = pose.rowRange(0, 3).colRange(0, 3);
        cv::Mat t = pose.rowRange(0, 3).col(3);

        R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
        t.copyTo(P2.rowRange(0, 3).col(3));
        P2 = K*P2;

        A.row(0) = vKey1.pt.x*P1.row(2)-P1.row(0);
        A.row(1) = vKey1.pt.y*P1.row(2)-P1.row(1);
        A.row(2) = vKey2.pt.x*P2.row(2)-P2.row(0);
        A.row(3) = vKey2.pt.y*P2.row(2)-P2.row(1);

        cv::Mat u, w, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        x3D = vt.row(3).t();
        x3D = x3D.rowRange(0, 3)/x3D.at<float>(3);
        x3D = x3D.rowRange(0, 3);
        return x3D;
    }

    cv::Mat Frame::ObtainFlowDepthCamera(const int &i, const bool &addnoise) {
        float z = mvStat3DPointTmp[i].z;

        cv::RNG rng((unsigned)time(NULL));

        if(addnoise)
        {
            z = z + rng.gaussian(z*z/(725*0.5)*0.15);
        }

        if(z > 0)
        {
            const float flow_u = mvFlowNext[i].x;
            const float flow_v = mvFlowNext[i].y;

            cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << flow_u, flow_v, z);

            return x3Dc;
        }
        else
        {
            cout << "found a depth value < 0 ...." << endl;
            return cv::Mat();
        }
    }

    cv::Mat Frame::ObtainFlowDepthObject(const int &i, const bool &addnoise) {
        float z = mvObj3DPoint[i].z;

        cv::RNG rng((unsigned) time(NULL));

        if(addnoise){
            z = z + rng.gaussian(z*z/(725*0.5)*0.15);
        }

        if(z <= 0)
        {
            z = 10000;
        }

        if(z>0)
        {
            const float flow_u = mvObjFlowNext[i].x;
            const float flow_v = mvObjFlowNext[i].y;

            cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << flow_u, flow_v, z);

            return x3Dc;
        }
        else{
            cout << "found a depth value < 0 ...." << endl;
            return cv::Mat();
        }
    }

    cv::Mat Frame::UnprojectStereoStat(const int &i, const bool &addnoise) {
        float z = mvStat3DPointTmp[i].z;

        cv::RNG rng((unsigned)time(NULL));

        if(addnoise){
            z = z + rng.gaussian(z*z/(725*0.5)*0.15);
        }

        if(z >= 0)
        {
            const float u = mvStatKeys[i].pt.x;
            const float v = mvStatKeys[i].pt.y;

            const float x = (u-cx)*z*invfx;
            const float y = (v-cy)*z*invfy;
            cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);

            const cv::Mat Rlw = camPose.rowRange(0, 3).colRange(0, 3);
            const cv::Mat Rwl = Rlw.t();
            const cv::Mat tlw = camPose.rowRange(0, 3).col(3);
            const cv::Mat twl = -Rlw.t()*tlw;

            return Rwl*x3Dc+twl;
        }
        else
        {
            cout << "found a depth value < 0 ... statKeys" << endl;
            return cv::Mat();
        }
    }
}
