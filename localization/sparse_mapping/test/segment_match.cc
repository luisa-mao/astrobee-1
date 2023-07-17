#include <Python.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sparse_mapping/sparse_map.h>
#include <boost/filesystem.hpp>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/dynamic_message.h>

#include <google/protobuf/descriptor_database.h>


void get_all_matches_for_sim_img(std::string const& query_img_path, std::string const& sim_img_path,
                    Eigen::Matrix2Xd const& keypoint_list,
                    std::map<int, int> const& fid_to_pid,
                    std::vector<Eigen::Vector3d> const& pid_to_xyz,
                    PyObject* module, 
                    camera::CameraParameters const& camera_params,
                    std::vector<Eigen::Vector2d> observations,
                    std::vector<Eigen::Vector3d> landmarks
                    )
    // from list of matched keypoints, find fid -> pid -> xyz
    // don't forget to undistort the keypoints
{

    PyObject* get_matched_keypoints_for_sim_img_func = PyObject_GetAttrString(module, "get_matched_keypoints_for_sim_img");

    // Convert string arguments to PyObjects
    PyObject* arg1Obj = PyUnicode_FromString(query_img_path.c_str());
    PyObject* arg2Obj = PyUnicode_FromString(sim_img_path.c_str());

    // Create a tuple of arguments
    PyObject* argsTuple = PyTuple_New(2);
    PyTuple_SetItem(argsTuple, 0, arg1Obj);
    PyTuple_SetItem(argsTuple, 1, arg2Obj);

    PyObject* get_matched_keypoints_for_sim_img_result = PyObject_CallObject(get_matched_keypoints_for_sim_img_func, argsTuple);
    Py_DECREF(argsTuple);
    
    // Extract the individual elements from the tuple
    PyObject* keypoints1_obj = PyTuple_GetItem(get_matched_keypoints_for_sim_img_result, 0);
    PyObject* keypoints2_obj = PyTuple_GetItem(get_matched_keypoints_for_sim_img_result, 1);

    // Extract the keypoints from the lists
    // std::vector<cv::KeyPoint> keypoints1;
    std::vector<Eigen::Vector2d> keypoints2;

    for (Py_ssize_t i = 0; i < PyList_Size(keypoints1_obj); ++i) {
        PyObject* keypoint_obj = PyList_GetItem(keypoints1_obj, i);

        float x = PyFloat_AsDouble(PyTuple_GetItem(keypoint_obj, 0));
        float y = PyFloat_AsDouble(PyTuple_GetItem(keypoint_obj, 1));

        // Create a cv::KeyPoint object and add it to keypoints1
        observations.push_back(Eigen::Vector2d(x, y));
    }

    for (Py_ssize_t i = 0; i < PyList_Size(keypoints2_obj); ++i) {
        PyObject* keypoint_obj = PyList_GetItem(keypoints2_obj, i);

        float x = PyFloat_AsDouble(PyTuple_GetItem(keypoint_obj, 0));
        float y = PyFloat_AsDouble(PyTuple_GetItem(keypoint_obj, 1));

        // Create a cv::KeyPoint object and add it to keypoints2
        keypoints2.push_back(Eigen::Vector2d(x, y));
    }


    // Cleanup
    Py_XDECREF(get_matched_keypoints_for_sim_img_func);
    Py_XDECREF(get_matched_keypoints_for_sim_img_result);

    // get the landmarks from keypoints2
    for (int i = 0; i < keypoints2.size(); ++i) {
        Eigen::Vector2d keypoint = keypoints2[i];
        keypoint.x() = keypoint.x() - 1280/2;
        keypoint.y() = keypoint.y() - 960/2;
        camera_params.Convert<camera::DISTORTED_C, camera::UNDISTORTED_C>(keypoint, &keypoint);
        for (int j = 0; j < keypoint_list.cols(); ++j) {
            Eigen::Vector2d keypoint_list_keypoint = keypoint_list.col(j);
            if (keypoint_list_keypoint == keypoint) {
                if (fid_to_pid.count(j) == 0)
                    continue;
                const int landmark_id = fid_to_pid.at(j);
                landmarks.push_back(pid_to_xyz[landmark_id]);
                break;
            }
        }
    }

    std::cout << "observations: " << observations.size() << std::endl;
    std::cout << "landmarks: " << landmarks.size() << std::endl;
}

void create_map(){
    // Create a Sparse Map
    sparse_mapping::SparseMap map("/home/lmao/Documents/20210304_aach.map", true);
    std::cout<<"Loaded map with "<<map.GetNumFrames()<<std::endl;
}

void get_module() {
    // Import the Python module containing your functions
    // PyObject* module_name = PyUnicode_FromString("test_matches");
    PyObject* module_name = PyUnicode_FromString("make_matches");
    std::cout << "module_name: " << "make_matches" << std::endl;
    PyObject* module = PyImport_Import(module_name);
    std::cout << "module: " << module << std::endl;
    Py_DECREF(module_name);
    std::cout << "Imported Python module" << std::endl;

    // Check if the module was imported successfully
    if (module == NULL) {
        PyErr_Print();
        Py_Finalize();
        // return 1;
    }
    std::cout << "Successfully imported Python module" << std::endl;
    // clean up
    Py_XDECREF(module);
}

int main() {


    // Initialize the Python interpreter
    Py_Initialize();
    std::cout << "Initialized Python interpreter" << std::endl;

    // Set the Python sys path to include the current directory
    PyRun_SimpleString("import sys\n"
                       "sys.path.append('/home/lmao/Documents')");
    std::cout << "Set Python sys path" << std::endl;

    // Import the Python module containing your functions
    PyObject* module_name = PyUnicode_FromString("test");
    std::cout << "module_name: " << module_name << std::endl;
    PyObject* module = PyImport_Import(module_name);
    std::cout << "module: " << module << std::endl;
    Py_DECREF(module_name);
    std::cout << "Imported Python module" << std::endl;

    // Check if the module was imported successfully
    if (module == NULL) {
        PyErr_Print();
        Py_Finalize();
        return 1;
    }
    std::cout << "Successfully imported Python module" << std::endl;

    // Create a Sparse Map
    sparse_mapping::SparseMap map("/home/lmao/Documents/20210304_aach.map", true);
    std::cout<<"Loaded map with "<<map.GetNumFrames()<<std::endl;

    std::vector<int> indices;
    cv::Mat test_descriptors;
    Eigen::Matrix2Xd test_keypoints;
    std::string img_path = "/home/lmao/Documents/yaw2_images/20.jpg";
    std::cout << "img_path: " << img_path << std::endl;
    map.DetectFeaturesFromFile(img_path,
                                      false,
                                      &test_descriptors,
                                      &test_keypoints);
    std::cout << "test_descriptors: " << test_descriptors.size() << std::endl;

    // query the vocab database
    sparse_mapping::QueryDB(map.GetDetectorName(),
                            &(map.vocab_db_),
                            // Notice that we request more similar
                            // images than what we need. We'll prune
                            // them below.
                            20,
                            test_descriptors,
                            &indices);

    
    int cid = indices[0];
    Eigen::Matrix2Xd keypoint_list = map.cid_to_keypoint_map_[cid];
    std::map<int, int> fid_to_pid = map.cid_fid_to_pid_[cid];

    std::string map_image_dir = "/srv/novus_1/amoravar/data/images/latest_map_imgs/2020-09-24/";
    std::string path = map.cid_to_filename_[cid];
    boost::filesystem::path filePath(path);
    std::string filename = filePath.filename().string();
    std::string sim_img_path = map_image_dir + filename;
    std::cout << "sim_img_path: " << sim_img_path << std::endl;

    std::vector<Eigen::Vector2d> observations;
    std::vector<Eigen::Vector3d> landmarks;

    // get_all_matches_for_sim_img(img_path, sim_img_path,
    //                 keypoint_list,
    //                 fid_to_pid,
    //                 map.pid_to_xyz_,
    //                 module, 
    //                 map.GetCameraParameters(),
    //                 observations,
    //                 landmarks);

    // clean up
    Py_XDECREF(module);

    // Finalize the Python interpreter
    Py_Finalize();


    /////////////////////////////////
    // map.cid_to_filename_;
    // 






    return 0;
}
