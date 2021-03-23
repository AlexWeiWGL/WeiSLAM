#include "cvplot/highgui.h"
#include <opencv2/highgui/highgui.hpp>

namespace cvplot{
    using namespace std;

    int createTrackbar(const string &trackbarname, const string winname, int *value, int count,
                       TrackbarCallback onchange, void *userdata){
        return cv::createTrackbar(trackbarname, winname, value, count, onchange, userdata);
    }

    void destroyAllWindows(){cv::destroyAllWindows();}

    void destroyWindow(const string &view){
        Window::current().view(view).hide();
    }

    int getMouseWheelDelta(int flags){
#if CV_MAJOR_VERSION > 2
        return cv::getMouseWheelDelta(flags);
#else
        return -1;
#endif
    }

    int getTrackbarPos(const string &trackname,
                       const string &winname){
        return cv::getTrackbarPos(trackname, winname);
    }

    double getWindowProperty(const string &winname, int prop_id){
        return cv::getWindowProperty(winname, prop_id);
    }

    void show(const string &view, void *img){
        Window::current().view(view).drawImage(img);
        Window::current().view(view).finish();
        Window::current().view(view).flush();
    }

    void moveWindow(const string &view, int x, int y){
        Window::current().view(view).offset({x, y});
    }

    void namedWindow(const string &view, int flags){
        Window::current().view(view);
    }

    void resizeWindow(const string &view, int width, int height){
        Window::current().view(view).size({width, height});
    }

    void resizeWindow(const string &view, const Size &size){
        Window::current().view(view).size({size.width, size.height});
    }

    Rect selectROI(const string &windowName, void *img, bool showCrosshair,
                   bool fromCenter){
#if CV_MAJOR_VERSION > 2
        auto rect = cv::selectROI(windowName, (cv::InputArray)img, showCrosshair, fromCenter);
        return Rect(rect.x, rect.y, rect.width, rect.height);
#else
        return Rect(-1, -1, -1, -1);
#endif
    }

    Rect selectROI(void *img, bool showCrosshair, bool fromCenter){
#if CV_MAJOR_VERSION > 2
        auto rect = cv::selectROI((cv::InputArray)img, showCrosshair, fromCenter);
        return Rect(rect.x, rect.y, rect.width, rect.height);
#else
        return Rect(-1, -1, -1, -1);
#endif
    }

    void selectROIs(const string &windowName, void *img, vector<Rect> &boundingBoxes, bool showCrosshair,
                    bool fromCenter){
#if CV_MAJOR_VERSION > 2
        vector<cv::Rect> boxes;
        for (auto b : boundingBoxes) {
            boxes.push_back(cv::Rect(b.x, b.y, b.width, b.height));
        }

        cv::selectROIs(windowName, (cv::InputArray)img, boxes, showCrosshair,
                      fromCenter);
#endif
    }

    void setMouseCallback(const string &view, MouseCallback onMouse, void *userdata){
        Window::current().view(view).mouse(onMouse, userdata);
    }

    void setTrackbarMax(const string &trackbarname, const string &winname,
                        int maxval){
#if CV_MAJOR_VERSION > 2
        cv::setTrackbarMax(trackbarname, winname, maxval);
#endif
    }

    void setTrackbarMin(const string &trackbarname, const string &winname,
                        int minval){
#if CV_MAJOR_VERSION > 2
        cv::setTrackbarMin(trackbarname, winname, minval);
#endif
    }

    void setTrackbarPos(const string &trackbarname, const string winname,
                        int pos){
        cv::setTrackbarPos(trackbarname, winname, pos);
    }

    void setWindowProperty(const string &winname, int prop_id,
                           double prop_value){
        cv::setWindowProperty(winname, prop_id, prop_value);
    }

    void setWindowTitle(const string &view, const string &title){
        Window::current().view(view).title(title);
    }

    int startWindowThread(){
        return cv::startWindowThread();
    }

    int waitKey(int delay){return Util::key(delay);}

    int waitKeyEx(int delay){return Util::key(delay);}
}
