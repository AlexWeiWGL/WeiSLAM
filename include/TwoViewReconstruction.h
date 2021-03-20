//
// Created by alexwei on 2021/3/20.
//

#ifndef WEISLAM_TWOVIEWRECONSTRUCTION_H
#define WEISLAM_TWOVIEWRECONSTRUCTION_H
#include <opencv2/core.hpp>
#include <unordered_set>

namespace WeiSLAM{
    class TwoViewReconstruction{
        typedef std::pair<int, int> Match;

    public:
        //fix the reference frame
        TwoViewReconstruction(cv::Mat& k, float sigma=1.0, int iterations = 200);

        //compute in parallel a fundamental matrix and a homography
        //select a model and tries to recover the motion and the structure from motion
        bool Reconstruct(const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2, const std::vector<cv::KeyPoint>& vKeys,
                         const std::vector<int> &vMatches12, cv::Mat &r21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTraiangulated);

    private:
        void FindHomography(std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
        void FindFundamental(std::vector<bool> &vbInliers, float &score, cv::Mat &f21);

        cv::Mat ComputeH21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);
        cv::Mat ComputeF21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);

        float CheckHomography(const cv::Mat &H21, const cv::Mat H12, std::vector<bool> &vbMatchesInliers, float sigma);
        float CheckFundamental(const cv::Mat &F21, std::vector<bool> &vbMatchesInliers, float sigma);

        bool ReconstructF(std::vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K, cv::Mat &R21, cv::Mat &t21,
                          std::vector<cv::Point3f> &vp3D, std::vector<bool> &vpTriangulated, float minParallax, int minTriangulated);

        bool ReconstructH(std::vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K, cv::Mat &R21, cv::Mat &t21,
                          std::vector<cv::Point3f> &vp3D, std::vector<bool> &vpTriangulated, float minParallax, int minTriangulated);

        void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

        void Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

        int CheckRT(const cv::Mat &R, const cv::Mat &t, const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2,
                    const std::vector<Match> &vMatches12, std::vector<bool> &vbInliers,
                    const cv::Mat &K, std::vector<cv::Point3f> &vP3D, float th12, std::vector<bool> &vbGood, float &parapllax);

        void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);

        //keypoints from reference frame (Frame 1)
        std::vector<cv::KeyPoint> mvKeys1;

        //KeyPoints from current frame (Frame 2)
        std::vector<cv::KeyPoint> mvKeys2;

        //current matches from reference to current
        std::vector<Match> mvMatches12;
        std::vector<bool> mvbMatched1;

        //calibration
        cv::Mat mK;

        //standard Deviation and Variance
        float mSigma, mSigma2;

        //Ransac max iterations
        int mMaxIterations;

        //Ransac sets
        std::vector<std::vector<size_t>> mvSets;
    };
}
#endif //WEISLAM_TWOVIEWRECONSTRUCTION_H
