#define _CRT_SECURE_NO_WARNINGS

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
#include <windows.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <thread>
#include <sstream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <string>
#include <utility>

// Custom includes
#include "dfd_dnn_analysis.h"
#include "get_platform.h"
#include "file_parser.h"
#include "get_current_time.h"
#include "num2string.h"
//#include "center_cropper.h"
//#include "gorgon_common.h"
#include "array_image_operations.h"

// Net Version
// Things must go in this order since the array size is determined
// by the network header file
#include "dfd_net_v14.h"
#include "load_dfd_data.h"
#include "eval_dfd_net_performance.h"


// dlib includes
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>

// this is for enabling GUI support, i.e. opening windows
#ifndef DLIB_NO_GUI_SUPPORT
    #include <dlib/gui_widgets.h>
#endif

using namespace std;
using namespace dlib;

// -------------------------------GLOBALS--------------------------------------

extern const uint32_t img_depth;
extern const uint32_t secondary;
std::string platform;

std::string logfileName = "dfd_net_rw_analysis_results_";

// ----------------------------------------------------------------------------

void get_platform_control(void)
{
    get_platform(platform);
}

//-----------------------------------------------------------------------------

void print_usage(void)
{
    std::cout << "Enter the following as arguments into the program:" << std::endl;
    std::cout << "<config file>" << std::endl;
    std::cout << endl;
}

//-----------------------------------------------------------------------------

int main(int argc, char** argv)
{
    int idx;

    std::string sdate, stime;

    std::ofstream DataLogStream;
    //std::ofstream trainLogStream;
    //std::string train_inputfile;
    std::string test_inputfile;
    std::string net_name;
    std::string parseFilename;
    std::string results_name;
    std::string data_directory;
    std::string data_home;

    std::vector<std::vector<std::string>> test_file;
    std::vector<std::pair<std::string, std::string>> image_files;
    
    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

    std::vector<std::array<dlib::matrix<uint16_t>, img_depth>> te, te_crop;
    std::vector<dlib::matrix<uint16_t>> gt_train, gt_test, gt_crop;

    std::pair<uint64_t, uint64_t > crop_size(32, 32);
    std::pair<uint32_t, uint32_t> scale(1, 1);

    // these are the parameters to load in an image to make sure that it is the correct size
    // for the network.  The first number makes sure that the image is a modulus of the number
    // and the second number is an offest from the modulus.  This is used based on the network
    // structure (downsampling and upsampling tensor sizes).
    std::pair<uint32_t, uint32_t> mod_params(16, 0);  
    
    //double nmae_error = 0.0;
    //double nrmse_error = 0.0;
    //double ssim_val = 0.0;
    //double silog_error = 0.0;

    //////////////////////////////////////////////////////////////////////////////////

    if (argc == 1)
    {
        print_usage();
        std::cin.ignore();
        return 0;
    }

    get_platform_control();
    uint8_t HPC = 0;
    
    if(platform.compare(0,3,"HPC") == 0)
    {
        std::cout << "HPC Platform Detected." << std::endl;
        HPC = 1;
    }
    
    
    // setup save variable locations
    const std::string os_file_sep = "/";
    std::string program_root;
    std::string output_save_location;
    
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
    program_root = get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\") + os_file_sep;

#else    
    if(HPC == 1)
    {
        //HPC version
        program_root = get_path(get_path(get_path(std::string(argv[0]), os_file_sep), os_file_sep), os_file_sep) + os_file_sep;
    }
    else
    {
        // Ubuntu
        program_root = "/home/owner/DfD/dfd_rw_analysis/";
    }

#endif

    std::cout << "Reading Inputs... " << std::endl;
    std::cout << "Platform:               " << platform << std::endl;
    std::cout << "program_root:           " << program_root << std::endl;

   try {
        
        ///////////////////////////////////////////////////////////////////////////////
        // Step 1: Read in the training images
        ///////////////////////////////////////////////////////////////////////////////
        data_home = path_check(get_env_variable("DATA_HOME"));

        parseFilename = argv[1];
        
        // parse through the supplied input file
        parse_dfd_analysis_file(parseFilename, test_inputfile, net_name, results_name, output_save_location, crop_size, scale);
        
        if (test_inputfile == "" | net_name == "")
        {
            std::cout << "------------------------------------------------------------------" << std::endl;
            std::cout << "Error parsing input file.  No test input data file or network file provided." << std::endl;
            std::cout << "test_inputfile: " << test_inputfile << std::endl;
            std::cout << "net_name: " << net_name << std::endl;
            std::cout << "results_name: " << results_name << std::endl;
            std::cout << "Press Enter to continue..." << std::endl;

            std::cin.ignore();
            return 0;
        }

        std::cout << "output_save_location:   " << output_save_location << std::endl;

        // load the test data
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        parse_csv_file(test_inputfile, test_file);
        data_directory = data_home + test_file[0][0];
#else
        if (HPC == 1)
        {
            parse_csv_file(test_inputfile, test_file);
            data_directory = data_home + test_file[0][2];
        }
        else
        {
            parse_csv_file(test_inputfile, test_file);
            data_directory = data_home + test_file[0][1];
        }
#endif

        test_file.erase(test_file.begin());
        
        std::cout << "data_directory:         " << data_directory << std::endl;

        get_current_time(sdate, stime);
        logfileName = logfileName + results_name + "_" + sdate + "_" + stime + ".txt";

        std::cout << "Log File:               " << (output_save_location + logfileName) << std::endl;
        DataLogStream.open((output_save_location + logfileName), ios::out | ios::app);

        std::cout << "Data Input File:        " << test_inputfile << std::endl << std::endl;

        // Add the date and time to the start of the log file
        DataLogStream << "------------------------------------------------------------------" << std::endl;
        DataLogStream << "Version: 2.1    Date: " << sdate << "    Time: " << stime << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;

        DataLogStream << "Platform:             " << platform << std::endl;
        DataLogStream << "program_root:         " << program_root << std::endl;
        DataLogStream << "output_save_location: " << output_save_location << std::endl;
        DataLogStream << "data_directory:       " << data_directory << std::endl;
        DataLogStream << "Data Input File:      " << test_inputfile << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;

        std::cout << "Test image sets to parse: " << test_file.size() << std::endl;

        DataLogStream << "Test image sets to parse: " << test_file.size() << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;

        std::cout << "Loading test images..." << std::endl;
        
        start_time = chrono::system_clock::now();
        load_dfd_data(test_file, data_directory, mod_params, te, gt_test, image_files);
        
        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
        std::cout << "Loaded " << te.size() << " test image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl << std::endl;

        std::cout << "Input Array Depth: " << img_depth << std::endl;
        std::cout << "Secondary data loading value: " << secondary << std::endl << std::endl;
        DataLogStream << "Input Array Depth: " << img_depth << std::endl;
        DataLogStream << "Secondary data loading value: " << secondary << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl << std::endl;
        
        // save the eval crop size
        std::cout << "Eval Crop Size: " << crop_size.first << "x" << crop_size.second << std::endl << std::endl;
        DataLogStream << "Eval Crop Size: " << crop_size.first << "x" << crop_size.second << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;
        
        ///////////////////////////////////////////////////////////////////////////////
        // Step 2: Load the network
        ///////////////////////////////////////////////////////////////////////////////
        set_dnn_prefer_smallest_algorithms();

        dfd_net_type dfd_net;

        std::cout << "Loading " << net_name << std::endl;
        deserialize(net_name) >> dfd_net;

        std::cout << dfd_net << std::endl;

        DataLogStream << "Net Name: " << net_name << std::endl;
        DataLogStream << dfd_net << std::endl;
        //DataLogStream << "------------------------------------------------------------------" << std::endl;
        
        ///////////////////////////////////////////////////////////////////////////////
        // Step 3: Analyze the results of the network
        ///////////////////////////////////////////////////////////////////////////////
  
        std::cout << "Ready to analyze the network performance..." << std::endl;
        
#ifndef DLIB_NO_GUI_SUPPORT
        dlib::image_window win0;
        dlib::image_window win1;
        dlib::image_window win2;
#endif

        double nmae_accum = 0.0;
        double nrmse_accum = 0.0;
        double ssim_accum = 0.0;
        double var_gt_accum = 0.0;
        double var_dm_accum = 0.0;
        double silog_accum = 0.0;

        uint64_t count = 0;

        dlib::matrix<uint16_t> map;
        dlib::matrix<double,1,6> results = dlib::zeros_matrix<double>(1,6);
        
        // run through the network once.  This primes the GPU and stabilizes the timing
        // don't need the results.
        eval_net_performance(dfd_net, te[0], gt_test[0], map, crop_size, scale);
      
        for (idx = 0; idx < te.size(); ++idx)
        {
            // time and analyze the results
            start_time = chrono::system_clock::now(); 
            results = eval_net_performance(dfd_net, te[idx], gt_test[idx], map, crop_size, scale);
            stop_time = chrono::system_clock::now();

            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

#ifndef DLIB_NO_GUI_SUPPORT
            dlib::matrix<dlib::rgb_pixel> rgb_img;
            merge_channels(te[idx], rgb_img, 0);

            win0.clear_overlay();
            win0.set_image(rgb_img);
            win0.set_title("Input Image");
            
            win1.clear_overlay();
            win1.set_image(mat_to_rgbjetmat(dlib::matrix_cast<float>(gt_test[idx]), 0.0, 255.0));
            win1.set_title("Groundtruth Depthmap");

            win2.clear_overlay();
            win2.set_image(mat_to_rgbjetmat(dlib::matrix_cast<float>(map), 0.0, 255.0));
            win2.set_title("DFD DNN Depthmap");
#endif

            std::string image_filename = output_save_location + "depthmap_image_" + results_name + num2str(idx, "_%05d") + ".png";
            
            std::cout << "------------------------------------------------------------------" << std::endl;
            std::cout << "Depthmap generation completed in: " << elapsed_time.count() << " seconds." << std::endl;
            std::cout << "Image Size (h x w): " << map.nr() << " x " << map.nc() << std::endl;
            std::cout << "Focus File:     " << image_files[idx].first << std::endl;
            std::cout << "Defocus File:   " << image_files[idx].second << std::endl;
            std::cout << "Depth Map File: " << image_filename << std::endl;
            std::cout << "NMAE   " << std::setw(5) << std::setfill('0') << idx << ": " << std::fixed << std::setprecision(5) << results(0, 0) << std::endl;
            std::cout << "NRMSE  " << std::setw(5) << std::setfill('0') << idx << ": " << std::fixed << std::setprecision(5) << results(0, 1) << std::endl;
            std::cout << "SSIM   " << std::setw(5) << std::setfill('0') << idx << ": " << std::fixed << std::setprecision(5) << results(0, 2) << std::endl;
            std::cout << "SILOG  " << std::setw(5) << std::setfill('0') << idx << ": " << std::fixed << std::setprecision(5) << results(0, 3) << std::endl;           
            std::cout << "Var_GT " << std::setw(5) << std::setfill('0') << idx << ": " << std::fixed << std::setprecision(5) << results(0, 4) << std::endl;
            std::cout << "Var_DM " << std::setw(5) << std::setfill('0') << idx << ": " << std::fixed << std::setprecision(5) << results(0, 5) << std::endl;

            DataLogStream << "------------------------------------------------------------------" << std::endl;
            DataLogStream << "Depthmap generation completed in: " << elapsed_time.count() << " seconds." << std::endl;
            DataLogStream << "Image Size (h x w): " << map.nr() << " x " << map.nc() << std::endl;
            DataLogStream << "Focus File:     " << image_files[idx].first << std::endl;
            DataLogStream << "Defocus File:   " << image_files[idx].second << std::endl;
            DataLogStream << "Depth Map File: " << image_filename << std::endl;
            DataLogStream << "NMAE   " << std::setw(5) << std::setfill('0') << idx << ": " << std::fixed << std::setprecision(5) << results(0, 0) << std::endl;
            DataLogStream << "NRMSE  " << std::setw(5) << std::setfill('0') << idx << ": " << std::fixed << std::setprecision(5) << results(0, 1) << std::endl;
            DataLogStream << "SSIM   " << std::setw(5) << std::setfill('0') << idx << ": " << std::fixed << std::setprecision(5) << results(0, 2) << std::endl;
            DataLogStream << "SILOG  " << std::setw(5) << std::setfill('0') << idx << ": " << std::fixed << std::setprecision(5) << results(0, 3) << std::endl;
            DataLogStream << "Var_GT " << std::setw(5) << std::setfill('0') << idx << ": " << std::fixed << std::setprecision(5) << results(0, 4) << std::endl;
            DataLogStream << "Var_DM " << std::setw(5) << std::setfill('0') << idx << ": " << std::fixed << std::setprecision(5) << results(0, 5) << std::endl;

            // add code to save image
            dlib::save_png(dlib::matrix_cast<uint8_t>(map), image_filename);

            nmae_accum += results(0, 0);
            nrmse_accum += results(0, 1);
            ssim_accum += results(0, 2);
            silog_accum += results(0, 3);
            var_gt_accum += results(0, 4);
            var_dm_accum += results(0, 5);
            ++count;

            //std::cout << "Press Enter to continue..." << std::endl;
            //std::cin.ignore();
            //std::string key;
            //char key;
            //std::getline(cin,key);

            //if(key.compare("q")==0)
            //    break;


        }
        
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Average Image Analysis Results:" << std::endl;
        std::cout << "Average NMAE:   " << nmae_accum / (double)te.size() << std::endl;
        std::cout << "Average NRMSE:  " << nrmse_accum / (double)te.size() << std::endl;
        std::cout << "Average SSIM:   " << ssim_accum / (double)te.size() << std::endl;
        std::cout << "Average SILOG:  " << silog_accum / (double)te.size() << std::endl;
        std::cout << "Average Var_GT: " << var_gt_accum / (double)te.size() << std::endl;
        std::cout << "Average Var_DM: " << var_dm_accum / (double)te.size() << std::endl;

        //std::cout << "Average VIPF Val:  " << vipf_accum / (double)count << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Average NMAE, NRMSE, SSIM, SILOG,  Var_GT, Var_DM: " << nmae_accum / (double)te.size() << ", " << nrmse_accum / (double)te.size() << ", " << ssim_accum / (double)te.size()
                  << ", " << silog_accum / (double)te.size() << ", " << var_gt_accum / (double)te.size() << ", " << var_dm_accum / (double)te.size() << std::endl;
        std::cout << std::endl;

        DataLogStream << "------------------------------------------------------------------" << std::endl;
        DataLogStream << "Average Image Analysis Results:" << std::endl;
        DataLogStream << "Average NMAE:   " << nmae_accum / (double)te.size() << std::endl;
        DataLogStream << "Average NRMSE:  " << nrmse_accum / (double)te.size() << std::endl;
        DataLogStream << "Average SSIM:   " << ssim_accum / (double)te.size() << std::endl;       
        DataLogStream << "Average SILOG:  " << silog_accum / (double)te.size() << std::endl;
        DataLogStream << "Average Var_GT: " << var_gt_accum / (double)te.size() << std::endl;
        DataLogStream << "Average Var_DM: " << var_dm_accum / (double)te.size() << std::endl;

        // just save everything for easy copying
        DataLogStream << "------------------------------------------------------------------" << std::endl;
        DataLogStream << "Average NMAE, NRMSE, SSIM, Var_GT, Var_DM: " << nmae_accum / (double)te.size() << ", " << nrmse_accum / (double)te.size() << ", " << ssim_accum / (double)te.size()
                      << ", " << silog_accum / (double)te.size() << ", " << var_gt_accum / (double)te.size() << ", " << var_dm_accum / (double)te.size() << std::endl;
        DataLogStream << std::endl;


    #if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        Beep(500, 1000);
    #endif

        std::cout << "End of Program." << std::endl;
        DataLogStream.close();

        std::cin.ignore();

#ifndef DLIB_NO_GUI_SUPPORT
        win0.close_window();
        win1.close_window();
        win2.close_window();
#endif

    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;

        DataLogStream << e.what() << std::endl;
        DataLogStream << "------------------------------------------------------------------" << std::endl;

        DataLogStream.close();
       
        std::cout << "Press Enter to close..." << std::endl;
        std::cin.ignore();

    }
    return 0;

}    // end of main
       
        
        

