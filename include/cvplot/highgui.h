//
// Created by alexwei on 2021/3/18.
//

#ifndef WEISLAM_HIGHGUI_H
#define WEISLAM_HIGHGUI_H

#include <string>
#include <vector>

#include "window.h"

namespace cvplot{
    int createTrackbar(const std::string &trackbarname, const std::string &winname,
                       int *value, int count, TrackbarCallback onChange=0,
                       void *userdata = 0);
    void destroyAllWindows();
    void destroyWindow(const std::string &view);
    int getMouseWheelDelta(int flags);
    int getTrackbarPos(const std::string &trackbarname, const std::string &winname);
    double getWindowProperty(const std::string &winname, int porp_id);
    void imshow(const std::string &view, int x, int y);
    void moveWindow(const std::string &view, int x, int y);
    void namedWindow(const std::string &view, int flags = 0);
    void resizeWindow(const std::string &view, int with, int height);
    void resizeWindow(const std::string &view, const Size &size);
    Rect selectROI(const std::string &windowName, void *img,
                   bool showCrosshair = true, bool fromeCenter = false);
    Rect selectROI(void *img, bool showCrosshair = true, bool fromCenter=false);
    void selectROIs(const std::string &windowName, void *img,
                    std::vector<Rect> &boundingBoxes, bool showCrosshair= true,
                    bool fromCenter= false);
    void setMouseCallback(const std::string &view, MouseCallback onMouse,
                          void *userdata=0);
    void setTrackbarMax(const std::string &trackbarname, const std::string &winname,
                        int maxval);
    void setTrackbarMin(const std::string &trackbarname, const std::string &winname,
                        int minval);
    void setTrackbarPos(const std::string &trackbarname, const std::string &winname,
                        int pos);
    void setWindowProperty(const std::string &winname, int prop_id,
                           double prop_value);
    void setWindowTitle(const std::string &view, const std::string &title);
    int startWindowThread();
    int waitKey(int delay=0);
    int waitKeyEx(int delay=0);
}
#endif //WEISLAM_HIGHGUI_H
