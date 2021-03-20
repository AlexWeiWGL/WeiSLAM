#include "TwoViewReconstruction.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include <thread>

using namespace std;

namespace WeiSLAM{
    TwoViewReconstruction::TwoViewReconstruction(cv::Mat &k, float sigma, int iterations) {
        mK = k.clone();

        mSigma = sigma;
        mSigma2 = sigma*sigma;
        mMaxIterations = iterations;
    }

    bool TwoViewReconstruction::Reconstruct(const std::vector<cv::KeyPoint> &vKeys1,
                                            const std::vector<cv::KeyPoint> &vKeys2, const std::vector<cv::KeyPoint> &,
                                            const std::vector<int> &vMatches12, cv::Mat &r21, cv::Mat &t21,
                                            std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTraiangulated) {
        mvKeys1.clear();
        mvKeys2.clear();

        mvKeys1 = vKeys1;
        mvKeys2 = vKeys2;

        //fill structures with current keypoints and matches with reference frame
        //reference frame 1 , current frame 2
        mvMatches12.clear();
        mvMatches12.reserve(mvKeys2.size());
        mvbMatched1.resize(mvKeys1.size());
        for(size_t i=0, iend=vMatches12.size(); i< iend; i++)
        {
            if(vMatches12[i] >= 0)
            {
                mvMatches12.push_back(make_pair(i, vMatches12[i]));
                mvbMatched1[i] = true;
            } else
                mvbMatched1[i] = false;
        }

        const int N = mvMatches12.size();

        //indices for minimum set selection
        vector<size_t> vAllIndices;
        vAllIndices.reserve(N);
        vector<size_t> vAvailableIndices;

        for(int i=0; i<N; i++)
        {
            vAllIndices.push_back(i);
        }

        //Generate sets of 8 points for each RANSAC iteration
        mvSets = vector<vector<size_t>>(mMaxIterations, vector<size_t>(8,0));

        DUtils::Random::SeedRandOnce(0);

        for (int it = 0; it < mMaxIterations; it++) {
            vAvailableIndices = vAllIndices;

            //select a minimum set
            for(size_t j=0; j<8; j++)
            {
                int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);
                int idx = vAvailableIndices[randi];

                mvSets[it][j] = idx;

                vAvailableIndices[randi] = vAvailableIndices.back();
                vAvailableIndices.pop_back();
            }
        }

        //launch threads to compute in parallel a fundamental matrix and a homography
        vector<bool> vbMatchesInliersH, vbMatchesInliersF;
        float SH, SF;
        cv::Mat H, F;

        thread threadH(&TwoViewReconstruction::FindHomography, this, ref(vbMatchesInliersH), ref(SH), ref(H));
        thread threadF(&TwoViewReconstruction::FindFundamental, this, ref(vbMatchesInliersF), ref(SF), ref(F));

        //wait until both threads have finished
        threadH.join();
        threadF.join();

        if(SH + SF == 0.f) return false;
        float RH = SH/(SH + SF);

        float minParallax = 1.0;

        //try to reconstruct from homography or fundamental depending on the ratio(0.4-0.45)
        if(RH > 0.5)
        {
            return ReconstructH(vbMatchesInliersH, H, mK, r21, t21, vP3D, vbTraiangulated, minParallax, 50);
        } else
        {
            return ReconstructF(vbMatchesInliersF, F, mK, r21, t21, vP3D, vbTraiangulated, minParallax, 50);
        }
    }

    void TwoViewReconstruction::FindHomography(std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21) {
        //number of putative matches
        const int N = mvMatches12.size();

        //normalize coordinates
        vector<cv::Point2f> vPn1, vPn2;
        cv::Mat T1, T2;
        Normalize(mvKeys1, vPn1, T1);
        Normalize(mvKeys2, vPn2, T2);
        cv::Mat T2inv = T2.inv();

        //best results variables
        score = 0.0;
        vbMatchesInliers = vector<bool>(N, false);

        //iteration variables
        vector<cv::Point2f> vPn1i(8);
        vector<cv::Point2f> vPn2i(8);
        cv::Mat H21i, H12i;
        vector<bool> vbCurrentInliers(N, false);
        float currentScore;

        //perform all RANSAC iterations and save the solution with highest score
        for(int it=0; it<mMaxIterations; it++)
        {
            //select a minimum set
            for(int j=0; j<8; j++)
            {
                int idx = mvSets[it][j];

                vPn1i[j] = vPn1[mvMatches12[idx].first];
                vPn2i[j] = vPn2[mvMatches12[idx].second];
            }

            cv::Mat Hn = ComputeF21(vPn1i, vPn2i);

            H21i = T2inv*Hn*T1;
            H12i = H12i.inv();

            currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

            if(currentScore>score)
            {
                H21 = H21i.clone();
                vbMatchesInliers = vbCurrentInliers;
                score = currentScore;
            }
        }
    }

    void TwoViewReconstruction::FindFundamental(std::vector<bool> &vbInliers, float &score, cv::Mat &f21) {
        const int N = vbInliers.size();

        vector<cv::Point2f> vPn1, vPn2;
        cv::Mat T1, T2;
        Normalize(mvKeys1, vPn1, T1);
        Normalize(mvKeys2, vPn2, T2);
        cv::Mat T2t = T2.t();

        score = 0.0;
        vbInliers = vector<bool>(N, false);

        vector<cv::Point2f> vPn1i(8);
        vector<cv::Point2f> vPn2i(8);
        cv::Mat F21i;
        vector<bool> vbCurrentInliers(N, false);
        float currentScore;

        for(int it=0; it<mMaxIterations; it++)
        {
            for(int j=0; j<8; j++)
            {
                int idx = mvSets[it][j];

                vPn1i[j] = vPn1[mvMatches12[idx].first];
                vPn2i[j] = vPn2[mvMatches12[idx].second];
            }

            cv::Mat Fn = ComputeF21(vPn1i, vPn2i);

            F21i = T2t*Fn*T1;

            currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

            if(currentScore>score){
                f21 = F21i.clone();
                vbInliers = vbCurrentInliers;
                score = currentScore;
            }
        }
    }

    cv::Mat TwoViewReconstruction::ComputeH21(const std::vector<cv::Point2f> &vP1,
                                              const std::vector<cv::Point2f> &vP2) {
        const int N = vP1.size();
        cv::Mat A(2*N, 9, CV_32F);

        for(int i=0; i<N; i++)
        {
            const float u1 = vP1[i].x;
            const float v1 = vP1[i].y;
            const float u2 = vP2[i].x;
            const float v2 = vP2[i].y;

            A.at<float>(2*i, 0) = 0.0;
            A.at<float>(2*i, 1) = 0.0;
            A.at<float>(2*i, 2) = 0.0;
            A.at<float>(2*i, 3) = -u1;
            A.at<float>(2*i, 4) = -v1;
            A.at<float>(2*i, 5) = -1;
            A.at<float>(2*i, 6) = v2*u1;
            A.at<float>(2*i, 7) = v2*v1;
            A.at<float>(2*i, 8) = v2;

            A.at<float>(2*i+1, 0) = u1;
            A.at<float>(2*i+1, 1) = v1;
            A.at<float>(2*i+1, 2) = 1;
            A.at<float>(2*i+1, 3) = 0.0;
            A.at<float>(2*i+1, 4) = 0.0;
            A.at<float>(2*i+1, 5) = 0.0;
            A.at<float>(2*i+1, 6) = -u2*u1;
            A.at<float>(2*i+1, 7) = -u2*v1;
            A.at<float>(2*i+1, 8) = -u2;
        }

        cv::Mat u,w,vt;
        cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        return vt.row(8).reshape(0, 3);
    }

    cv::Mat TwoViewReconstruction::ComputeF21(const std::vector<cv::Point2f> &vP1,
                                              const std::vector<cv::Point2f> &vP2) {
        const int N = vP1.size();
        cv::Mat A(N, 9, CV_32F);

        for(int i=0; i<N; i++)
        {
            const float u1 = vP1[i].x;
            const float v1 = vP1[i].y;
            const float u2 = vP2[i].x;
            const float v2 = vP2[i].y;

            A.at<float>(i, 0) = u2*u1;
            A.at<float>(i, 1) = u2*u1;
            A.at<float>(i, 2) = u2;
            A.at<float>(i, 3) = v2*v1;
            A.at<float>(i, 4) = v2*v1;
            A.at<float>(i, 5) = v2;
            A.at<float>(i, 6) = u1;
            A.at<float>(i, 7) = v1;
            A.at<float>(i, 8) = 1;
        }

        cv::Mat u, w, vt;
        cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        cv::Mat Fpre = vt.row(8).reshape(0, 3);
        cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        w.at<float>(2) = 0;
        return u*cv::Mat::diag(w)*vt;
    }

    float TwoViewReconstruction::CheckHomography(const cv::Mat &H21, const cv::Mat H12,
                                                 std::vector<bool> &vbMatchesInliers, float sigma) {
        const int N = mvMatches12.size();

        const float h11 = H21.at<float>(0, 0);
        const float h12 = H21.at<float>(0, 1);
        const float h13 = H21.at<float>(0, 2);
        const float h21 = H21.at<float>(1, 0);
        const float h22 = H21.at<float>(1, 1);
        const float h23 = H21.at<float>(1, 2);
        const float h31 = H21.at<float>(2, 0);
        const float h32 = H21.at<float>(2, 1);
        const float h33 = H21.at<float>(2, 2);

        const float h11inv = H12.at<float>(0, 0);
        const float h12inv = H12.at<float>(0, 1);
        const float h13inv = H12.at<float>(0, 2);
        const float h21inv = H12.at<float>(1, 0);
        const float h22inv = H12.at<float>(1, 1);
        const float h23inv = H12.at<float>(1, 2);
        const float h31inv = H12.at<float>(2, 0);
        const float h32inv = H12.at<float>(2, 1);
        const float h33inv = H12.at<float>(2, 2);

        vbMatchesInliers.resize(N);

        float score = 0;

        const float th = 5.991;

        const float imvSigmaSqure = 1.0/(sigma*sigma);

        for(int i=0; i<N; i++)
        {
            bool bIn = true;

            const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
            const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

            const float u1 = kp1.pt.x;
            const float v1 = kp1.pt.y;
            const float u2 = kp2.pt.x;
            const float v2 = kp2.pt.y;

            const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
            const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
            const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

            const float squareDist1 = (u1-u2in1)*(u1-u2in1) + (v1-v2in1)*(v1-v2in1);

            const float chiSquare1 = squareDist1*imvSigmaSqure;

            if(chiSquare1>th)
                bIn = false;
            else
                score += th - chiSquare1;

            const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
            const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
            const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

            const float squareDist2 = (u2-u1in2)*(u2-u1in2) + (v2-v1in2)*(v2-v1in2);

            const float chiSquare2 = squareDist2*imvSigmaSqure;

            if(chiSquare2>th)
                bIn = false;
            else
                score += th - chiSquare2;

            if(bIn)
                vbMatchesInliers[i] = true;
            else
                vbMatchesInliers[i] = false;
        }
        return score;
    }


}

