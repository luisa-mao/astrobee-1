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
#include <gflags/gflags.h>
#include "json.hpp"

using json = nlohmann::json;

// make a string flag
DEFINE_string(semantic_loc_bag_img_dir, "/home/lmao/Documents/yaw2_images", "Path to bag images");
class PythonInterpreter {
public:
    PyObject* get_matched_keypoints_for_sim_img_func;
    PyObject* get_keypoints_from_masked_image_func;
    PyObject* module;
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
        get_matched_keypoints_for_sim_img_func = PyObject_GetAttrString(module, "get_matched_keypoints_for_sim_img");
        get_keypoints_from_masked_image_func = PyObject_GetAttrString(module, "get_keypoints_from_masked_image");

    }

private:
    PythonInterpreter() {
        // Private constructor to prevent direct instantiation
    }

    ~PythonInterpreter() {
        // Destructor to clean up resources
        Py_DECREF(get_matched_keypoints_for_sim_img_func);
        Py_DECREF(get_keypoints_from_masked_image_func);
        Py_DECREF(module);
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


void get_all_matches_for_sim_img(std::string const& query_img_path, std::string const& sim_img_path,
                    Eigen::Matrix2Xd const& keypoint_list,
                    std::map<int, int> const& fid_to_pid,
                    std::vector<Eigen::Vector3d> const& pid_to_xyz,
                    PyObject* get_matched_keypoints_for_sim_img_func, 
                    camera::CameraParameters const& camera_params,
                    std::vector<Eigen::Vector2d> &observations,
                    std::vector<Eigen::Vector3d> &landmarks
                    )
    // from list of matched keypoints, find fid -> pid -> xyz
    // don't forget to undistort the keypoints
{

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
    // Py_XDECREF(get_matched_keypoints_for_sim_img_func);
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

                const int landmark_id = fid_to_pid.at(j);
                // skip if the match is already added
                std::vector<Eigen::Vector2d>::iterator it = std::find(observations.begin(), observations.end(), obs);

                if (it != observations.end()) {
                    // Element found, compute the index using std::distance
                    size_t index = std::distance(observations.begin(), it);

                    // if obs is in observations and landmark is in landmarks at corresonding index, skip
                    if (landmarks[index] == pid_to_xyz[landmark_id]) {
                        continue;
                    }
                }

                observations.push_back(obs);
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

void get_keypoints_from_image(std::string img_path, PyObject *module, std::vector<Eigen::Vector2d> & keypoints){
    // call the python function
    PyObject* arg1Obj = PyUnicode_FromString(img_path.c_str());
    PyObject* argsTuple = PyTuple_New(1);
    PyTuple_SetItem(argsTuple, 0, arg1Obj);
    PyObject* get_keypoints_from_masked_image_result = PyObject_CallObject(module, argsTuple);
    Py_DECREF(argsTuple);

    // Extract the results and put them in a vector
    for (Py_ssize_t i = 0; i < PyList_Size(get_keypoints_from_masked_image_result); ++i) {
        PyObject* keypoint_obj = PyList_GetItem(get_keypoints_from_masked_image_result, i);

        float x = PyFloat_AsDouble(PyTuple_GetItem(keypoint_obj, 0));
        float y = PyFloat_AsDouble(PyTuple_GetItem(keypoint_obj, 1));

        // Create a Eigen::Vector2d object and add it to keypoints
        keypoints.push_back(Eigen::Vector2d(x, y));
    }
    // print keypoints list
    // std::cout << "keypoints size after python function " << keypoints.size() << std::endl;
    // for (int j = 0; j < 5; j++){
    //     std::cout << keypoints[j][0] <<"  "<< keypoints[j][1] << std::endl;
    // }

    
}

void fix_vocab(sparse_mapping::SparseMap &og_map, sparse_mapping::SparseMap &map) {
    map.vocab_db_ = og_map.vocab_db_;
    map.Save("/home/lmao/Documents/baburen.map");

}

void compare_map_cids(sparse_mapping::SparseMap &og_map, sparse_mapping::SparseMap &map) {
    std::string map_image_dir = "/srv/novus_1/amoravar/data/images/latest_map_imgs/2020-09-24/";
    // get a list of filenames for all the images in the map
    std::vector<std::string> og_map_filenames;
    for (auto const& x : og_map.cid_to_filename_) {
        // if (count > 5){ // this just to test with a few images
        //     break;
        // }
        // split the path and get the filename
        boost::filesystem::path filePath(x);
        std::string filename = filePath.filename().string();
        std::string full_image_path = map_image_dir + filename;
        // check if the file exists
        if (boost::filesystem::exists(full_image_path)) {
            og_map_filenames.push_back(full_image_path);
        }
    }
    int count = 0;
    std::vector<std::string> map_filenames;
    for (auto const& x : map.cid_to_filename_) {
        // if (count > 20){ // this just to test with a few images
        //     break;
        // }
        // split the path and get the filename
        boost::filesystem::path filePath(x);
        // print the path
        // std::cout << "filePath: " << filePath << std::endl;
        std::string filename = filePath.filename().string();
        std::string full_image_path = map_image_dir + filename;
        // check if the file exists
        if (boost::filesystem::exists(full_image_path)) {
            map_filenames.push_back(full_image_path);
        }
        count++;
    }

    // print lengths
    std::cout << "og_map_filenames size: " << og_map_filenames.size() << std::endl;
    std::cout << "map_filenames size: " << map_filenames.size() << std::endl;

    // if filename in map is not in og_map, print the filename and index
    for (int i = 0; i < map_filenames.size(); i++){
        std::string filename = map_filenames[i];
        if (std::find(og_map_filenames.begin(), og_map_filenames.end(), filename) == og_map_filenames.end()){
            std::cout << "filename: " << filename << " index: " << i << std::endl;
        }
    }
}


void make_map_from_masked_images(sparse_mapping::SparseMap &map){ // pass by reference
    // initialize the python interpreter
    PythonInterpreter::getInstance().initialize();
    PyObject *module = PythonInterpreter::getInstance().get_keypoints_from_masked_image_func;
    
    std::string map_image_dir = "/srv/novus_1/amoravar/data/images/latest_map_imgs/2020-09-24/";
    // get a list of filenames for all the images in the map
    std::vector<std::string> new_cid_to_filename;
    int count = 0;
    for (auto const& x : map.cid_to_filename_) {
        // if (count > 5){ // this just to test with a few images
        //     break;
        // }
        // split the path and get the filename
        boost::filesystem::path filePath(x);
        std::string filename = filePath.filename().string();
        std::string full_image_path = map_image_dir + filename;
        // check if the file exists
        if (boost::filesystem::exists(full_image_path)) {
            new_cid_to_filename.push_back(full_image_path);
            count++;
        }
    }

    std::vector<Eigen::Matrix2Xd > cid_to_keypoint_map_new;
    std::vector<cv::Mat> cid_to_descriptor_map_new;


    // std::vector<Eigen::Vector3d> pid_to_xyz_new;
    std::vector<Eigen::Affine3d > cid_to_cam_t_global_new(map.cid_to_cam_t_global_.begin(),
                                                            map.cid_to_cam_t_global_.begin()+new_cid_to_filename.size()); // probably don't need this bc cids are in the same order
    std::vector<std::map<int, int> > cid_fid_to_pid_new;

    // call the python function
    std::vector<Eigen::Vector2d> keypoints;
    std::vector<int> fids; // maps fid of new to fid of old
    for (int i = 0; i < new_cid_to_filename.size(); i++){
        // print the cid
        std::cout << "cid: " << i << std::endl;
        get_keypoints_from_image(new_cid_to_filename[i], module, keypoints);

        // find indices keypoint in keypoint_map
        Eigen::Matrix2Xd keypoint_list = map.cid_to_keypoint_map_[i];
        for (int j = 0; j < keypoints.size(); j++){
            Eigen::Vector2d keypoint = keypoints[j];
            keypoint.x() = keypoint.x() - 1280/2;
            keypoint.y() = keypoint.y() - 960/2;
            map.camera_params_.Convert<camera::DISTORTED_C, camera::UNDISTORTED_C>(keypoint, &keypoint);
            bool found_keypoint = false;
            for (int k = 0; k < keypoint_list.cols(); k++){
                Eigen::Vector2d keypoint_list_keypoint = keypoint_list.col(k);
                if (keypoint_list_keypoint == keypoint){
                    fids.push_back(k);
                    found_keypoint = true;
                    break;
                }
            }
            if (!found_keypoint){
                // std::cout << "Could not find keypoint " << keypoint << std::endl;
            }
        }
        // print fids list length

        // make keypoints into an Eigen::Matrix2d and add to cid_to_keypoint_map_new
        Eigen::Matrix2Xd keypoint_matrix(2, fids.size());
        for (int j = 0; j < fids.size(); j++){
            keypoint_matrix.col(j) = keypoint_list.col(fids[j]);
        }
        cid_to_keypoint_map_new.push_back(keypoint_matrix);

        // make new cv::Mat and add to cid_to_descriptor_map_new
        cv::Mat descriptors_old = map.cid_to_descriptor_map_[i];
        // get fid indices from descriptors_old and make into new cv::Mat
        cv::Mat descriptors_new(fids.size(), descriptors_old.cols, descriptors_old.type());
        for (int j = 0; j < fids.size(); j++){
            descriptors_old.row(fids[j]).copyTo(descriptors_new.row(j));
        }
        // print the size of the new descriptors
        cid_to_descriptor_map_new.push_back(descriptors_new);

        // change the fids
        std::map<int, int> fid_to_pid_new;
        std::map<int, int> fid_to_pid = map.cid_fid_to_pid_[i];
        for (int j = 0; j < fids.size(); j++){
            fid_to_pid_new[j] = fid_to_pid[fids[j]];
        }
        cid_fid_to_pid_new.push_back(fid_to_pid_new);
        // clear the keypoints vector
        keypoints.clear();
        fids.clear();
    }

    // make pid_cid_to_fid from cid_fid_to_pid
    std::vector<std::map<int, int> > pid_to_cid_fid_new;
    pid_to_cid_fid_new.resize(map.pid_to_xyz_.size(), std::map<int, int>());
    for (int cid = 0; cid < cid_fid_to_pid_new.size(); cid++){
        std::map<int, int> fid_to_pid = cid_fid_to_pid_new[cid];
        for (auto const& x : fid_to_pid) {
            pid_to_cid_fid_new[x.second][cid] = x.first;
        }
    }


    // set all map attributes to the new attrubutes
    map.cid_to_filename_ = new_cid_to_filename;
    map.cid_to_keypoint_map_ = cid_to_keypoint_map_new;
    map.cid_to_descriptor_map_ = cid_to_descriptor_map_new;
    map.cid_fid_to_pid_ = cid_fid_to_pid_new;
    map.pid_to_cid_fid_ = pid_to_cid_fid_new;
    map.cid_to_cam_t_global_ = cid_to_cam_t_global_new;
    std::cout<< "set all map attributes to the new attributes " << std::endl;

    // make new vocabulary
    BuildDBforDBoW2(&map, "SURF",
                     5, 10,
                     0);
    std::cout<< "made new vocabulary " << std::endl;
    // save the map
    map.Save("/home/lmao/Documents/test_masked_map.map");

}

bool sparse_mapping::semantic_localize(sparse_mapping::SparseMap &map, cv::Mat &image_descriptors, Eigen::Matrix2Xd &image_keypoints,
                               camera::CameraModel &camera, std::vector<Eigen::Vector3d> &inlier_landmarks, std::vector<Eigen::Vector2d>&inlier_observations){
    static int count = 0;
    // Initialize the Python interpreter
    PythonInterpreter::getInstance().initialize();
    PyObject *module = PythonInterpreter::getInstance().get_matched_keypoints_for_sim_img_func;

    std::vector<int> indices;
    // query the vocab database
    sparse_mapping::QueryDB(map.GetDetectorName(),
                            &(map.vocab_db_),
                            // Notice that we request more similar
                            // images than what we need. We'll pruneFZF
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
    std::string img_path = FLAGS_semantic_loc_bag_img_dir +std::to_string(count)+".jpg";
    std::cout << "img_path: " << img_path << std::endl;
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
        // std::cout << i<< " " << cid << " sim_img_path: " << sim_img_path << std::endl;

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

/*
    // Initialize the Python interpreter
    PythonInterpreter::getInstance().initialize();
    PyObject *module = PythonInterpreter::getInstance().get_matched_keypoints_for_sim_img_func;
    std::cout << "module: " << module << std::endl;

    // Create a Sparse Map
*/
    sparse_mapping::SparseMap og_map("/home/lmao/Documents/20210304_aach.surf.vocab.hist_eq.map", false);
    sparse_mapping::SparseMap map("/srv/novus_1/rsoussan/maps/luisa_surf/20220111_Soundsee_Cabanel_Base_cmS4_cmS3_reg.surf.map", true);
    std::cout<<"Loaded map with "<<map.GetNumFrames()<<std::endl;
    std::cout<<"Loaded og map with "<<og_map.GetNumFrames()<<std::endl;
    // make_map_from_masked_images(map);
    // fix_vocab(og_map, map);
    compare_map_cids(og_map, map);

/*
    std::vector<int> indices;
    cv::Mat test_descriptors;
    Eigen::Matrix2Xd test_keypoints;
    std::string img_path = "/home/lmao/Documents/yaw2_images/129.jpg";
    // std::string img_path = "/home/lmao/Documents/yaw2_images/20.jpg";
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
    // Py_XDECREF(module);

    // Finalize the Python interpreter
    // Py_Finalize();


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


*/
    return 0;
}
