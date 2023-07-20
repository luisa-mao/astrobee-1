#include <Python.h>
#include <iostream>
#include <vector>
#include <set>
#include <opencv2/opencv.hpp>
#include <sparse_mapping/sparse_map.h>
#include <boost/filesystem.hpp>
#include <ff_common/thread.h>
#include <ff_common/utils.h>
#include <sparse_mapping/reprojection.h>
#include "json.hpp"

using json = nlohmann::json;

class PythonInterpreter {
public:
    PyObject* module;  // Public member variable for the module
    bool initialized = false;
    static PythonInterpreter& getInstance() {
        static PythonInterpreter instance;
        return instance;
    }

    void initialize() {
        if (initialized) {
            return;
        }
        initialized = true;
        Py_Initialize();
        std::cout << "Initialized Python interpreter" << std::endl;

        // Set the Python sys path to include the current directory
        PyRun_SimpleString("import sys\n"
                           "sys.path.append('/home/lmao/Documents')");
        std::cout << "Set Python sys path" << std::endl;

        // Import the Python module containing your functions
        PyObject* module_name = PyUnicode_FromString("make_matches");
        std::cout << "module_name: " << module_name << std::endl;
        module = PyImport_Import(module_name);
        std::cout << "module: " << module << std::endl;
        Py_DECREF(module_name);
        std::cout << "Imported Python module" << std::endl;
    }

private:
    PythonInterpreter() {
        // Private constructor to prevent direct instantiation
    }

    ~PythonInterpreter() {
        // Destructor to clean up resources
        Py_Finalize();
    }

    // Delete the copy constructor and assignment operator to prevent copies
    PythonInterpreter(const PythonInterpreter&) = delete;
    PythonInterpreter& operator=(const PythonInterpreter&) = delete;
};

/////////////////////////////////////////////////////////////

void saveDataToJson(const std::vector<Eigen::Vector2d>& observations, const std::vector<Eigen::Vector3d>& landmarks, const std::string& filename) {
    json data;

    // Convert observations to JSON
    json obs_json;
    for (const auto& obs : observations) {
        json obs_entry;
        obs_entry["x"] = obs[0];
        obs_entry["y"] = obs[1];
        obs_json.push_back(obs_entry);
    }
    data["observations"] = obs_json;

    // Convert landmarks to JSON
    json landmarks_json;
    for (const auto& landmark : landmarks) {
        json landmark_entry;
        landmark_entry["x"] = landmark[0];
        landmark_entry["y"] = landmark[1];
        landmark_entry["z"] = landmark[2];
        landmarks_json.push_back(landmark_entry);
    }
    data["landmarks"] = landmarks_json;

    // Save data to JSON file
    std::ofstream file(filename);
    if (file.is_open()) {
        file << data.dump(4);  // Indentation of 4 spaces
        file.close();
        std::cout << "Data saved to " << filename << std::endl;
    } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
}

void test_cpp_and_python_get_same_keypoints(sparse_mapping::SparseMap & map, PyObject* module) {
    cv::Mat test_descriptors;
    Eigen::Matrix2Xd test_keypoints;
    std::string query_img_path = "/home/lmao/Documents/yaw2_images/129.jpg";
    std::string sim_img_path = "/srv/novus_1/amoravar/data/images/latest_map_imgs/2020-09-24/1599238881.9869831.jpg";
    // std::cout << "img_path: " << img_path << std::endl;
    map.DetectFeaturesFromFile(sim_img_path,
                                      false,
                                      &test_descriptors,
                                      &test_keypoints);

    PyObject* get_matched_keypoints_for_sim_img_func = PyObject_GetAttrString(module, "get_matched_keypoints_for_sim_img");
    // Convert string arguments to PyObjects
    PyObject* arg1Obj = PyUnicode_FromString(query_img_path.c_str());
    PyObject* arg2Obj = PyUnicode_FromString(sim_img_path.c_str());

    // Create a tuple of arguments
    PyObject* argsTuple = PyTuple_New(2);
    PyTuple_SetItem(argsTuple, 0, arg1Obj);
    PyTuple_SetItem(argsTuple, 1, arg2Obj);

    // std::cout << "created tuple of arguments "<< std::endl;

    // Check if arg1Obj and arg2Obj were built correctly
    const char* arg1 = PyUnicode_AsUTF8(arg1Obj);
    const char* arg2 = PyUnicode_AsUTF8(arg2Obj);

    PyObject* get_matched_keypoints_for_sim_img_result = PyObject_CallObject(get_matched_keypoints_for_sim_img_func, argsTuple);
    Py_DECREF(argsTuple);

    // std::cout << "called python function "<< std::endl;
    
    // Extract the individual elements from the tuple
    PyObject* keypoints1_obj = PyTuple_GetItem(get_matched_keypoints_for_sim_img_result, 0);
    PyObject* keypoints2_obj = PyTuple_GetItem(get_matched_keypoints_for_sim_img_result, 1);

    // std::cout << "got keypoints "<< std::endl;

    // Extract the keypoints from the lists
    std::vector<Eigen::Vector2d> keypoints1;
    std::vector<Eigen::Vector2d> keypoints2;

    for (Py_ssize_t i = 0; i < PyList_Size(keypoints2_obj); ++i) {
        PyObject* keypoint_obj = PyList_GetItem(keypoints2_obj, i);

        float x = PyFloat_AsDouble(PyTuple_GetItem(keypoint_obj, 0));
        float y = PyFloat_AsDouble(PyTuple_GetItem(keypoint_obj, 1));

        // Create a cv::KeyPoint object and add it to keypoints2
        Eigen::Vector2d keypoint(x - 1280/2, y - 960/2);
        map.GetCameraParameters().Convert<camera::DISTORTED_C, camera::UNDISTORTED_C>(keypoint, &keypoint);
        keypoints2.push_back(keypoint);
    }
    // print the lengths
    std::cout << "test keypoints: " << test_keypoints.size() << std::endl;
    std::cout << "keypoints2 length: " << keypoints2.size() << std::endl;
    // check if all points in keypoints2 are in test_keypoints
    for (auto& keypoint : keypoints2) {
        bool found = false;
        for (int i = 0; i < test_keypoints.cols(); ++i) {
            if (keypoint == test_keypoints.col(i)) {
                found = true;
                break;
            }
        }
        if (!found) {
            std::cout << "keypoint not found in test_keypoints" << std::endl;
            // print the keypoint
            std::cout << keypoint << std::endl;
        }
    }



    // Cleanup
    Py_XDECREF(get_matched_keypoints_for_sim_img_func);
    Py_XDECREF(get_matched_keypoints_for_sim_img_result);
}



void get_all_matches_for_sim_img(std::string const& query_img_path, std::string const& sim_img_path,
                    Eigen::Matrix2Xd const& keypoint_list,
                    std::map<int, int> const& fid_to_pid,
                    std::vector<Eigen::Vector3d> const& pid_to_xyz,
                    PyObject* module, 
                    camera::CameraParameters const& camera_params,
                    std::vector<Eigen::Vector2d> &observations,
                    std::vector<Eigen::Vector3d> &landmarks
                    )
    // from list of matched keypoints, find fid -> pid -> xyz
    // don't forget to undistort the keypoints
{

    PyObject* get_matched_keypoints_for_sim_img_func = PyObject_GetAttrString(module, "get_matched_keypoints_for_sim_img");
    // Check if get_matched_keypoints_for_sim_img_func is a valid function object
    int isCallable = PyCallable_Check(get_matched_keypoints_for_sim_img_func);

    if (isCallable) {
        // std::cout << "get_matched_keypoints_for_sim_img_func is a valid function object." << std::endl;
    } else {
        std::cout << "get_matched_keypoints_for_sim_img_func is not a valid function object." << std::endl;
    }

    // std::cout << "getting python module "<< std::endl;

    // Convert string arguments to PyObjects
    PyObject* arg1Obj = PyUnicode_FromString(query_img_path.c_str());
    PyObject* arg2Obj = PyUnicode_FromString(sim_img_path.c_str());

    // Create a tuple of arguments
    PyObject* argsTuple = PyTuple_New(2);
    PyTuple_SetItem(argsTuple, 0, arg1Obj);
    PyTuple_SetItem(argsTuple, 1, arg2Obj);

    // std::cout << "created tuple of arguments "<< std::endl;

    // Check if arg1Obj and arg2Obj were built correctly
    const char* arg1 = PyUnicode_AsUTF8(arg1Obj);
    const char* arg2 = PyUnicode_AsUTF8(arg2Obj);

    PyObject* get_matched_keypoints_for_sim_img_result = PyObject_CallObject(get_matched_keypoints_for_sim_img_func, argsTuple);
    Py_DECREF(argsTuple);

    // std::cout << "called python function "<< std::endl;
    
    // Extract the individual elements from the tuple
    PyObject* keypoints1_obj = PyTuple_GetItem(get_matched_keypoints_for_sim_img_result, 0);
    PyObject* keypoints2_obj = PyTuple_GetItem(get_matched_keypoints_for_sim_img_result, 1);

    // std::cout << "got keypoints "<< std::endl;

    // Extract the keypoints from the lists
    std::vector<Eigen::Vector2d> keypoints1;
    std::vector<Eigen::Vector2d> keypoints2;

    for (Py_ssize_t i = 0; i < PyList_Size(keypoints1_obj); ++i) {
        PyObject* keypoint_obj = PyList_GetItem(keypoints1_obj, i);

        float x = PyFloat_AsDouble(PyTuple_GetItem(keypoint_obj, 0));
        float y = PyFloat_AsDouble(PyTuple_GetItem(keypoint_obj, 1));

        // Create a cv::KeyPoint object and add it to keypoints1
        keypoints1.push_back(Eigen::Vector2d(x, y));
    }

    // std::cout << "got keypoints 1 "<< keypoints1.size()<< std::endl;

    for (Py_ssize_t i = 0; i < PyList_Size(keypoints2_obj); ++i) {
        PyObject* keypoint_obj = PyList_GetItem(keypoints2_obj, i);

        float x = PyFloat_AsDouble(PyTuple_GetItem(keypoint_obj, 0));
        float y = PyFloat_AsDouble(PyTuple_GetItem(keypoint_obj, 1));

        // Create a cv::KeyPoint object and add it to keypoints2
        keypoints2.push_back(Eigen::Vector2d(x, y));
    }

    // std::cout << "got keypoints 2 "<< keypoints2.size()<< std::endl;


    // Cleanup
    Py_XDECREF(get_matched_keypoints_for_sim_img_func);
    Py_XDECREF(get_matched_keypoints_for_sim_img_result);

    for (int i = 0; i < keypoints2.size(); ++i) {
        Eigen::Vector2d keypoint = keypoints2[i];
        keypoint.x() = keypoint.x() - 1280/2;
        keypoint.y() = keypoint.y() - 960/2;
        camera_params.Convert<camera::DISTORTED_C, camera::UNDISTORTED_C>(keypoint, &keypoint);
        bool found_keypoint = false;
        for (int j = 0; j < keypoint_list.cols(); ++j) {
            Eigen::Vector2d keypoint_list_keypoint = keypoint_list.col(j);
            if (keypoint_list_keypoint == keypoint) {
                if (fid_to_pid.count(j) == 0) {
                    continue;
                }
                // undistort the observation
                Eigen::Vector2d obs = keypoints1[i];
                obs.x() = obs.x() - 1280/2;
                obs.y() = obs.y() - 960/2;
                camera_params.Convert<camera::DISTORTED_C, camera::UNDISTORTED_C>(obs, &obs);
                // skip if obs is already in observations
                // if (std::find(observations.begin(), observations.end(), obs) != observations.end()) {
                //     continue;
                // }

                observations.push_back(obs);
                const int landmark_id = fid_to_pid.at(j);
                landmarks.push_back(pid_to_xyz[landmark_id]);
                found_keypoint = true;
                break;
            }
        }
        // if (!found_keypoint) {
        //     std::cout << "Could not find keypoint " << keypoint << std::endl;
        // }
        
    }
}


void plotPointsOnImage(const std::string& image_path, const std::vector<Eigen::Vector2d>& observations, const std::vector<Eigen::Vector2d>& inliers) {
    // Load the image
    cv::Mat image = cv::imread(image_path);

    // Calculate the desired scale factor for image resizing
    double scale_factor = 0.5;  // Adjust this value as needed for the desired scaling

    // Resize the image
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(), scale_factor, scale_factor);

    // Draw circles for each point on the resized image
    for (const auto& point : observations) {
        int x = static_cast<int>(point[0] * scale_factor);
        int y = static_cast<int>(point[1] * scale_factor);
        cv::circle(resized_image, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
    }

    // Draw circles for each point on the resized image
    for (const auto& point : inliers) {
        int x = static_cast<int>(point[0] * scale_factor);
        int y = static_cast<int>(point[1] * scale_factor);
        cv::circle(resized_image, cv::Point(x, y), 3, cv::Scalar(255, 0, 0), -1);
    }

    // Display the resized image with points
    cv::imshow("Resized Image with Points", resized_image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
bool sparse_mapping::test_func(){
    return true;
}

bool sparse_mapping::semantic_localize(sparse_mapping::SparseMap &map, cv::Mat &image_descriptors, Eigen::Matrix2Xd &image_keypoints,
                               camera::CameraModel &camera, std::vector<Eigen::Vector3d> &inlier_landmarks, std::vector<Eigen::Vector2d>&inlier_observations){
    static int count = 0;
    // Initialize the Python interpreter
    PythonInterpreter::getInstance().initialize();
    PyObject *module = PythonInterpreter::getInstance().module;

    std::vector<int> indices;
    // query the vocab database
    sparse_mapping::QueryDB(map.GetDetectorName(),
                            &(map.vocab_db_),
                            // Notice that we request more similar
                            // images than what we need. We'll prune
                            // them below.
                            20,
                            image_descriptors,
                            &indices);

    std::vector<int> similarity_rank(indices.size(), 0);
    std::vector<std::vector<cv::DMatch> > all_matches(indices.size());
    int total = 0;
    // TODO(oalexan1): Use multiple threads here?
    for (size_t i = 0; i < indices.size(); i++) {
        int cid = indices[i];
        interest_point::FindMatches(image_descriptors,
                                    map.cid_to_descriptor_map_[cid],
                                    &all_matches[i]);

        for (size_t j = 0; j < all_matches[i].size(); j++) {
        if (map.cid_fid_to_pid_[cid].count(all_matches[i][j].trainIdx) == 0)
            continue;
        similarity_rank[i]++;
        }
    }
    std::vector<int> highly_ranked = ff_common::rv_order(similarity_rank);

    std::string map_image_dir = "/srv/novus_1/amoravar/data/images/latest_map_imgs/2020-09-24/";
    std::string img_path = "/home/lmao/Documents/yaw2_images/"+std::to_string(count)+".jpg";
    std::vector<Eigen::Vector3d> landmarks;
    std::vector<Eigen::Vector2d> observations;


    for (int i = 0; i < 20 ; i++){ // change this to 20
        int cid = indices[highly_ranked[i]];
        Eigen::Matrix2Xd keypoint_list = map.cid_to_keypoint_map_[cid];
        std::map<int, int> fid_to_pid = map.cid_fid_to_pid_[cid];
        std::string path = map.cid_to_filename_[cid];
        boost::filesystem::path filePath(path);
        std::string filename = filePath.filename().string();
        std::string sim_img_path = map_image_dir + filename;
        // std::cout << cid << " sim_img_path: " << sim_img_path << std::endl;

        get_all_matches_for_sim_img(img_path, sim_img_path,
                        keypoint_list,
                        fid_to_pid,
                        map.pid_to_xyz_,
                        module, 
                        map.GetCameraParameters(),
                        observations,
                        landmarks);
    }

    int ransac_inlier_tolerance = 5;
    int num_ransac_iterations = 1000;
    int ret = sparse_mapping::RansacEstimateCamera(landmarks, observations,
                                 num_ransac_iterations,
                                 ransac_inlier_tolerance, &camera,
                                 &inlier_landmarks, &inlier_observations,
                                 true);
    count++;
    return (ret == 0);
}

int main() {


    // Initialize the Python interpreter
    PythonInterpreter::getInstance().initialize();
    PyObject *module = PythonInterpreter::getInstance().module;
    std::cout << "module: " << module << std::endl;
    // Create a Sparse Map
    sparse_mapping::SparseMap map("/home/lmao/Documents/20210304_aach.surf.vocab.hist_eq.map", true);
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

    std::vector<int> similarity_rank(indices.size(), 0);
    std::vector<std::vector<cv::DMatch> > all_matches(indices.size());
    int total = 0;
    // TODO(oalexan1): Use multiple threads here?
    for (size_t i = 0; i < indices.size(); i++) {
        int cid = indices[i];
        interest_point::FindMatches(test_descriptors,
                                    map.cid_to_descriptor_map_[cid],
                                    &all_matches[i]);

        for (size_t j = 0; j < all_matches[i].size(); j++) {
        if (map.cid_fid_to_pid_[cid].count(all_matches[i][j].trainIdx) == 0)
            continue;
        similarity_rank[i]++;
        }
    }
    std::vector<int> highly_ranked = ff_common::rv_order(similarity_rank);


    std::string map_image_dir = "/srv/novus_1/amoravar/data/images/latest_map_imgs/2020-09-24/";

    std::vector<Eigen::Vector2d> observations;
    std::vector<Eigen::Vector3d> landmarks;

    for (int i = 0; i < 20 ; i++){ // change this to 20
        int cid = indices[highly_ranked[i]];
        Eigen::Matrix2Xd keypoint_list = map.cid_to_keypoint_map_[cid];
        std::map<int, int> fid_to_pid = map.cid_fid_to_pid_[cid];
        std::string path = map.cid_to_filename_[cid];
        boost::filesystem::path filePath(path);
        std::string filename = filePath.filename().string();
        std::string sim_img_path = map_image_dir + filename;
        std::cout << cid << " sim_img_path: " << sim_img_path << std::endl;

        get_all_matches_for_sim_img(img_path, sim_img_path,
                        keypoint_list,
                        fid_to_pid,
                        map.pid_to_xyz_,
                        module, 
                        map.GetCameraParameters(),
                        observations,
                        landmarks);
    }



    int ransac_inlier_tolerance = 3;
    int num_ransac_iterations = 1000;
    std::vector<Eigen::Vector3d> inlier_landmarks;
    std::vector<Eigen::Vector2d> inlier_observations;
    camera::CameraModel camera(Eigen::Vector3d(),
                             Eigen::Matrix3d::Identity(),
                             map.GetCameraParameters());

    std::cout << "observations: " << observations.size() << std::endl;
    std::cout << "landmarks: " << landmarks.size() << std::endl;

    int ret = sparse_mapping::RansacEstimateCamera(landmarks, observations,
                                 num_ransac_iterations,
                                 ransac_inlier_tolerance, &camera,
                                 &inlier_landmarks, &inlier_observations,
                                 true);

    int n = observations.size();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (observations[i] == observations[j]) {
                std::cout << "duplicate observations: " << observations[i][0] << " " << observations[i][1] << std::endl;
                std::cout << "corresponding landmark: " << landmarks[i][0] << " " << landmarks[i][1] << " " << landmarks[i][2] << std::endl;
                std::cout << "_________________________________" << std::endl;
            }
        }
    }

    std::cout << "---------------------" << std::endl;


    // check if there are any duplicates in inlier_observations and print the duplicates
    n = inlier_observations.size();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (inlier_observations[i] == inlier_observations[j]) {
                std::cout << "duplicate inlier_observations: " << inlier_observations[i][0] << " " << inlier_observations[i][1] << std::endl;
                std::cout << "corresponding landmark: " << inlier_landmarks[i][0] << " " << inlier_landmarks[i][1] << " " << inlier_landmarks[i][2] << std::endl;
                std::cout << "_________________________________" << std::endl;
            }
        }
    }

    std::cout << "ret: " << ret << std::endl;
    std::cout << "num inliers " << inlier_landmarks.size() << std::endl;
    // clean up
    Py_XDECREF(module);

    // Finalize the Python interpreter
    Py_Finalize();


    /////////////////////////////////
    // map.cid_to_filename_;
    // 
    // print inlier observations and inlier landmarks

    // for (int i = 0; i < inlier_landmarks.size(); i++){
    //     std::cout << "inlier_landmarks: " << inlier_landmarks[i] << std::endl;
    //     std::cout << "inlier_observations: " << inlier_observations[i] << std::endl;
    // }

    // saveDataToJson(observations, landmarks, "obs_landmarks.json");
    // saveDataToJson(inlier_observations, inlier_landmarks, "inlier_obs_landmarks.json");

    // NOTE
    // maybe get rid of duplicate matches before going to ransac bc it takes 4 matches to get a matrix

    // distort observations
    for (int i = 0; i < observations.size(); i++){
        map.GetCameraParameters().Convert<camera::UNDISTORTED_C, camera::DISTORTED_C>(observations[i], &(observations[i]));
        observations[i].x() += 1280/2;
        observations[i].y() += 960/2;
    }

    for (int i = 0; i < inlier_observations.size(); i++){
        map.GetCameraParameters().Convert<camera::UNDISTORTED_C, camera::DISTORTED_C>(inlier_observations[i], &inlier_observations[i]);
        inlier_observations[i].x() += 1280/2;
        inlier_observations[i].y() += 960/2;
    }
    // print camera
    std::cout << "camera: x " << camera.GetPosition()[0] << std::endl;
    std::cout << "camera: y " << camera.GetPosition()[1] << std::endl;
    std::cout << "camera: z " << camera.GetPosition()[2] << std::endl;

    plotPointsOnImage(img_path, observations, inlier_observations);



    return 0;
}
