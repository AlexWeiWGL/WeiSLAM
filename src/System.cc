#include "../include/System.h"
#include "../include/Converter.h"
#include <thread>
#include <iomanip>

#include <unistd.h>

namespace WeiSLAM{
    System::System(const string &strSettingsFile, const eSensor sensor):mSensor(sensor){
        cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if(!fsSettings.isOpened()){
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }

        map = new Map();

        tracker = new Tracking(this, map, strSettingsFile, mSensor);

    }

    cv::Mat System::TrackRGBD(const cv::Mat &im, cv::Mat &depthmap, const cv::Mat &flowmap, const cv::Mat &masksem,
                              const cv::Mat &mTcw_gt, const vector<vector<float>> &vObjPose_gt,
                              const double &timestamp, cv::Mat &imTraj, const int &nImage)
    {
        if(mSensor != RGBD){
            cerr << "ERROR: you called TrackRGB but input sensor was not set to RGB." << endl;
            exit(-1);
        }

        cv::Mat Tcw = tracker->GrabImageRGB(im, depthmap, flowmap, masksem, mTcw_gt, vObjPose_gt, timestamp, imTraj, nImage);

        return Tcw;
    }

    cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp, string filename) {
        if(mSensor!=MONOCULAR)
        {
            cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular !!!"<< endl;
            exit(-1);
        }

    }

    void System::SaveResultsIJRR2021(const string &filename)
    {
        cout << endl << "Saving Result into TXT File ..." << endl;

        ofstream save_objmot, save_objmot_gt;
        string path_objmot = filename + "obj_mot_mono_new.txt";
        string path_objmot_gt = filename + "obj_mot_gt.txt";
        save_objmot.open(path_objmot.c_str(), ios::trunc);
        save_objmot_gt.open(path_objmot_gt.c_str(), ios::trunc);

        int start_frame = 0;

        for(int i=0; i<map->vmRigidMotion.size(); ++i)
        {
            if(map->vmRigidMotion[i].size() > 1)
            {
                for(int j=1; j<map->vmRigidMotion[i].size(); ++j)
                {
                    save_objmot << start_frame + i + 1 << " " << map->vnRMLabel[i][j] << " " << fixed << setprecision(9) <<
                                    map->vmRigidMotion[i][j].at<float>(0,0) << " " << map->vmRigidMotion[i][j].at<float>(0,1) << " " << map->vmRigidMotion[i][j].at<float>(0,2) << " " << map->vmRigidMotion[i][j].at<float>(0, 3) << " " <<
                                    map->vmRigidMotion[i][j].at<float>(1,0) << " " << map->vmRigidMotion[i][j].at<float>(1,1) << " " << map->vmRigidMotion[i][j].at<float>(1,2) << " " << map->vmRigidMotion[i][j].at<float>(1, 3) << " " <<
                                    map->vmRigidMotion[i][j].at<float>(1,0) << " " << map->vmRigidMotion[i][j].at<float>(2,1) << " " << map->vmRigidMotion[i][j].at<float>(2,2) << " " << map->vmRigidMotion[i][j].at<float>(2, 3) << " " <<
                                    0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << endl;
                    save_objmot_gt << start_frame + i + 1 << " " << map->vnRMLabel[i][j] << " " << fixed << setprecision(9) <<
                                    map->vmRigidMotion_GT[i][j].at<float>(0, 0) << " " << map->vmRigidMotion_GT[i][j].at<float>(0, 1) << " " << map->vmRigidMotion_GT[i][j].at<float>(0, 2) << " " << map->vmRigidMotion_GT[i][j].at<float>(0, 3) << " " <<
                                    map->vmRigidMotion_GT[i][j].at<float>(1, 0) << " " << map->vmRigidMotion_GT[i][j].at<float>(1, 1) << " " << map->vmRigidMotion_GT[i][j].at<float>(1, 2) << " " << map->vmRigidMotion_GT[i][j].at<float>(1, 3) << " " <<
                                    map->vmRigidMotion_GT[i][j].at<float>(2, 0) << " " << map->vmRigidMotion_GT[i][j].at<float>(2, 1) << " " << map->vmRigidMotion_GT[i][j].at<float>(2, 2) << " " << map->vmRigidMotion_GT[i][j].at<float>(2, 3) << " " <<
                                    0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << endl;
                }
            }
        }

        save_objmot.close();
        save_objmot_gt.close();

        vector<cv::Mat> CamPose_ini = map->vmCameraPose;

        ofstream save_traj_ini;
        string path_ini = filename + "initial_mono_new.txt";

        save_traj_ini.open(path_ini.c_str(), ios::trunc);

        for(int i=0; i<CamPose_ini.size(); ++i){
            save_traj_ini << start_frame + i << " " << fixed << setprecision(9) << 
                            CamPose_ini[i].at<float>(0, 0) << " " << CamPose_ini[i].at<float>(0, 1) << CamPose_ini[i].at<float>(0, 2) << " " << CamPose_ini[i].at<float>(0, 3) << " " <<
                            CamPose_ini[i].at<float>(1, 0) << " " << CamPose_ini[i].at<float>(1, 1) << CamPose_ini[i].at<float>(1, 2) << " " << CamPose_ini[i].at<float>(1, 3) << " " <<
                            CamPose_ini[i].at<float>(2, 0) << " " << CamPose_ini[i].at<float>(2, 1) << CamPose_ini[i].at<float>(2, 2) << " " << CamPose_ini[i].at<float>(2, 3) << " " <<
                            0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << endl;

        }
        
        save_traj_ini.close();

        vector<cv::Mat> CamPose_ref = map->vmCameraPose_RF;

        ofstream save_traj_ref;
        string path_ref = filename + "refined_mono_new.txt";
        save_traj_ref.open(path_ref.c_str(), ios::trunc);

        for(int i=0; i<CamPose_ref.size(); ++ i)
        {
            save_traj_ref << start_frame + i << " " << fixed << setprecision(9) <<
                            CamPose_ref[i].at<float>(0, 0) << " " << CamPose_ref[i].at<float>(0, 1) << " " <<CamPose_ref[i].at<float>(0, 2) << " " << CamPose_ref[i].at<float>(0, 3) << " " <<
                            CamPose_ref[i].at<float>(1, 0) << " " << CamPose_ref[i].at<float>(1, 1) << " " <<CamPose_ref[i].at<float>(1, 2) << " " << CamPose_ref[i].at<float>(1, 3) << " " <<  
                            CamPose_ref[i].at<float>(2, 0) << " " << CamPose_ref[i].at<float>(2, 1) << " " <<CamPose_ref[i].at<float>(2, 2) << " " << CamPose_ref[i].at<float>(2, 3) << " " <<
                            0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << endl;
        }

        save_traj_ref.close();


        vector<cv::Mat> CamPose_gt = map->vmCameraPose_GT;

        ofstream save_traj_gt;
        string path_gt = filename + "cam_pose_gt.txt";
        save_traj_gt.open(path_gt.c_str(), ios::trunc);

        for(int i=0; i<CamPose_gt.size(); ++i)
        {
            save_traj_gt << start_frame + i << " " << fixed << setprecision(9) << 
                            CamPose_gt[i].at<float>(0, 0) << " " << CamPose_gt[i].at<float>(0, 1) << " " << CamPose_gt[i].at<float>(0,2) << " " << CamPose_gt[i].at<float>(0, 3) << " " <<
                            CamPose_gt[i].at<float>(1, 0) << " " << CamPose_gt[i].at<float>(1, 1) << " " << CamPose_gt[i].at<float>(1,2) << " " << CamPose_gt[i].at<float>(1, 3) << " " <<
                            CamPose_gt[i].at<float>(2, 0) << " " << CamPose_gt[i].at<float>(2, 1) << " " << CamPose_gt[i].at<float>(2,2) << " " << CamPose_gt[i].at<float>(2, 3) << " " <<
                            0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << endl;
        }

        save_traj_gt.close();



        vector<vector<float>> All_timing = map->vfAll_time;
        vector<float> LBA_timing = map->fLBA_time;

        int obj_time_count = 0;

        vector<float> avg_timing(All_timing[0].size(), 0);
        for(int i=0; i<All_timing.size(); ++i)
        {
            for(int j=0; j<All_timing[i].size(); ++j)
            {
                if(j==3 && All_timing[i][j] != 0)
                {
                    avg_timing[j] = avg_timing[j] + All_timing[i][j];
                    obj_time_count = obj_time_count + 1;
                }
                else
                    avg_timing[j] = avg_timing[j] + All_timing[i][j];
            }
        }

        cout << "Time of all components: " << endl;
        for(int j=0; j < avg_timing.size(); ++j)
        {
            if (j == 3)
                cout << "(" << j << "): " << avg_timing[j]/obj_time_count << " ";
            else
                cout << "(" << j << "): " << avg_timing[j]/All_timing.size() << " ";
        }
        cout << endl;

        float avg_lba_timing = 0;
        for(int i=0; i < LBA_timing.size(); ++i)
            avg_lba_timing = avg_lba_timing + LBA_timing[i];
        cout << "Time of local bundle adjustment: " << avg_lba_timing/LBA_timing.size() << endl;
    
    }
}