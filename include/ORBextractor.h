
#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>

namespace WeiSLAM{
    class ExtractorNode{
        public:
            ExtractorNode():bNoMore(false){}

            void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);
            
            std::vector<cv::KeyPoint> vKeys;
            cv::Point2i UL, UR, BL, BR;
            std::list<ExtractorNode>::iterator lit;
            bool bNoMore;
    };

    class ORBextractor{
        public:
            enum{HARRIS_SCORE=0, FAST_SCORE=1};

            ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                         int iniThFAST, int minThFAST);
            
            ~ORBextractor(){}

            void operator()(cv::InputArray _image, cv::InputArray _mask,
                           std::vector<cv::KeyPoint>& _keypoints,
                           cv::OutputArray _descriptors);
            
            int inline GetLevels(){
                return nlevels;
            }

            float inline GetScaleFactor(){
                return scaleFactor;
            }

            std::vector<float> inline GetScaleFactors(){
                return mvScaleFactor;
            }

            std::vector<float> inline GetScaleSigmaSquares(){
                return mvLevelSigma2;
            }

            std::vector<float> inline GetInverseScaleSigmaSquares(){
                return mvInvLevelSigma2;
            }

            std::vector<cv::Mat> mvImagePyramid;

        protected:

            void ComputePyramid(cv::Mat image);
            void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>> & allKeypoints);
            std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
                                                    const int &maxX, const int &MinY, const int &maxY, const int &nFeatures, const int &level);
            
            void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint>> & allKeypoints);
            std::vector<cv::Point> pattern;

            int nfeatures;
            double scaleFactor;
            int nlevels;
            int iniThFAST;
            int minThFAST;

            std::vector<int> mnFeaturesPerLevel;
            
            std::vector<int> umax;

            std::vector<float> mvScaleFactor;
            std::vector<float> mvInvScaleFactor;
            std::vector<float> mvLevelSigma2;
            std::vector<float> mvInvLevelSigma2;

    };
}

#endif