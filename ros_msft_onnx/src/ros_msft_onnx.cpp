#include <ros/ros.h>
#include <boost/bind/bind.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/LinearMath/Quaternion.h>
#include <opencv2/opencv.hpp>

#include "ros_msft_onnx/ros_msft_onnx.h"
#include "ros_msft_onnx/yolo_box.h"
#include "ros_msft_onnx/pose_parser.h"

#include <string>
#include <codecvt>
#include <fstream>
#include <sstream>

using namespace std;

using convert_type = std::codecvt_utf8<wchar_t>;
using wstring_to_utf8 = std::wstring_convert<convert_type, wchar_t>;

const uint32_t kDefaultTensorWidth = 416;
const uint32_t kDefaultTensorHeight = 416;

using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;

static std::wstring to_wstring(std::string str)
{
    return strconverter.from_bytes(str);
}

OnnxProcessor::OnnxProcessor()
    : _process(ImageProcessing::Scale), _confidence(0.70f), _debug(false), _normalize(false)
{
}

bool OnnxProcessor::init(ros::NodeHandle &nh, ros::NodeHandle &nhPrivate)
{

    if (nhPrivate.getParam("image_processing", _imageProcessingType))
    {
        if (_imageProcessingType == "crop")
        {
            _process = Crop;
        }
        else if (_imageProcessingType == "scale")
        {
            _process = Scale;
        }
        else if (_imageProcessingType == "resize")
        {
            _process = Resize;
        }
        else
        {
            ROS_WARN("ONNX: unknown image processing type: %s", _imageProcessingType.c_str());
            // default;
        }
    }

    int temp = 0;
    if (nhPrivate.getParam("tensor_width", temp) && temp > 0)
    {
        _tensorWidth = (uint)temp;
    }
    else
    {
        _tensorWidth = kDefaultTensorWidth;
    }

    temp = 0;
    if (nhPrivate.getParam("tensor_height", temp) && temp > 0)
    {
        _tensorHeight = (uint)temp;
    }
    else
    {
        _tensorHeight = kDefaultTensorHeight;
    }

    nhPrivate.param<bool>("onnx_fake", _fake, false);

    nhPrivate.getParam("link_name", _linkName);

    if (!nhPrivate.getParam("onnx_model_path", _onnxModel) ||
        _onnxModel.empty())
    {
        ROS_ERROR("ONNX: onnx_model_path parameter has not been set.");
        return false;
    }

    if (nhPrivate.getParam("calibration", _calibration))
    {
        try
        {
            cv::FileStorage fs(_calibration, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
            fs["camera_matrix"] >> _camera_matrix;
            fs["distortion_coefficients"] >> _dist_coeffs;
        }
        catch (std::exception &e)
        {
            ROS_ERROR("Failed to read the calibration file, continuing without calibration.\n%s", e.what());
            // no calibration for you.
            _calibration = "";
        }
    }

    float conf;
    if (nhPrivate.getParam("confidence", conf))
    {
        _confidence = conf;
    }

    bool d;
    if (nhPrivate.getParam("debug", d))
    {
        _debug = d;
    }

    std::string imageTopic;
    if (!nhPrivate.getParam("image_topic", imageTopic) ||
        imageTopic.empty())
    {
        imageTopic = "/camera/image_raw";
    }

    _detect_pub = nh.advertise<visualization_msgs::MarkerArray>("tracked_objects", 1);

    image_transport::ImageTransport it(nh);
    _cameraSub = it.subscribe(imageTopic.c_str(), 1, &OnnxProcessor::ProcessImage, this);
    _image_pub = it.advertise("tracked_objects/image", 1);

    try
    {
        //*************************************************************************
        // initialize  enviroment...one enviroment per process
        // enviroment maintains thread pools and other state info
        _env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");

        // initialize session options if needed
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.DisableMemPattern();    // Required for DirectML
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);    // Required for DirectML
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        std::string modelString = _onnxModel;

#ifdef _WIN32
        auto modelFullPath = to_wstring(modelString).c_str();
#else
        auto modelFullPath = _onnxModel.c_str();
#endif

        _session = std::make_shared<Ort::Session>(*_env, modelFullPath, session_options);
        _allocator = std::make_shared<Ort::AllocatorWithDefaultOptions>();
    }
    catch (std::exception &e)
    {
        ROS_ERROR("ONNX: Failed to Start ML Session: %s", e.what());
        return false;
    }
    return true;
}

void OnnxProcessor::ProcessImage(const sensor_msgs::ImageConstPtr &image)
{
    if (_session == nullptr)
    {
        // Failed to initialize model or do not have one yet.
        return;
    }

    // Convert back to an OpenCV Image
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("ONNX: cv_bridge exception: %s", e.what());
        return;
    }

    cv::Size mlSize(_tensorWidth, _tensorHeight);
    cv::Mat rgb_image;
    cv::Mat image_resized;
    cv::Size s = cv_ptr->image.size();
    float aspectRatio = (float)s.width / (float)s.height;
    if (s.width <= 0 || s.height <= 0)
    {
        ROS_ERROR("ONNX: irrational image size received; one dimention zero or less");
        return;
    }

    if (_process == Crop &&
        (uint)s.width > _tensorWidth &&
        (uint)s.height > _tensorHeight)
    {
        // crop
        cv::Rect ROI((s.width - _tensorWidth) / 2, (s.height - _tensorHeight) / 2, _tensorWidth, _tensorHeight);
        image_resized = cv_ptr->image(ROI);
    }
    else if (_process == Resize)
    {
        cv::resize(cv_ptr->image, image_resized, mlSize, 0, 0, cv::INTER_CUBIC);
    }
    else
    {
        // We want to extract a correct apsect ratio from the center of the image
        // but scale the whole frame so that there are no borders.

        // First downsample
        cv::Size downsampleSize;

        if (aspectRatio > 1.0f)
        {
            downsampleSize.height = mlSize.height;
            downsampleSize.width = mlSize.height * aspectRatio;
        }
        else
        {
            downsampleSize.width = mlSize.width;
            downsampleSize.height = mlSize.width * aspectRatio;
        }

        cv::resize(cv_ptr->image, image_resized, downsampleSize, 0, 0, cv::INTER_CUBIC);

        // now extract the center
        cv::Rect ROI((downsampleSize.width - _tensorWidth) / 2, (downsampleSize.height - _tensorHeight) / 2, _tensorWidth, _tensorHeight);
        image_resized = image_resized(ROI);
    }

    // Convert to RGB
    cv::cvtColor(image_resized, rgb_image, cv::COLOR_BGR2RGB);

    // Set the image to 32-bit floating point values for tensorization.
    cv::Mat image_32_bit;
    rgb_image.convertTo(image_32_bit, CV_32F);

    if (_normalize)
    {
        cv::normalize(image_32_bit, image_32_bit, 0.0f, 1.0f, cv::NORM_MINMAX);
    }

    // Extract color channels from interleaved data
    cv::Mat channels[3];
    cv::split(image_32_bit, channels);

    size_t input_tensor_size = _tensorHeight * _tensorWidth * 3;

    std::vector<float> input_tensor_values(input_tensor_size);

    memcpy(&input_tensor_values[0], (float *)channels[0].data, _tensorWidth * _tensorHeight * sizeof(float));
    memcpy(&input_tensor_values[_tensorWidth * _tensorHeight], (float *)channels[1].data, _tensorWidth * _tensorHeight * sizeof(float));
    memcpy(&input_tensor_values[2 * _tensorWidth * _tensorHeight], (float *)channels[2].data, _tensorWidth * _tensorHeight * sizeof(float));

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::TypeInfo type_info = _session->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_node_dims = {1, 3, _tensorWidth, _tensorHeight};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());

    // score model & input tensor, get back output tensor
    std::vector<float> output;
    try
    {
        auto output_tensors = _session->Run(Ort::RunOptions{nullptr}, _input_node_names.data(), &input_tensor, 1, _output_node_names.data(), 1);

        auto &output_tensor = output_tensors.front();
        auto output_type_info = output_tensor.GetTensorTypeAndShapeInfo();
        size_t output_total_len = output_type_info.GetElementCount();

        float *floatarr = output_tensor.GetTensorMutableData<float>();

        output.resize(output_total_len);
        memcpy(&output[0], floatarr, output_total_len * sizeof(float));
    }
    catch (std::exception &e)
    {
        ROS_ERROR("ONNX: Session Failed!: %s", e.what());
        return;
    }

    ProcessOutput(output, image_resized);
}

bool OnnxTracker::init(ros::NodeHandle &nh, ros::NodeHandle &nhPrivate)
{
    _nh = nh;
    _nhPrivate = nhPrivate;

    f = boost::bind(&OnnxTracker::callback, this, boost::placeholders::_1, boost::placeholders::_2);
    server.setCallback(f);

    return _status;
}

void OnnxTracker::callback(ros_msft_onnx::reconfigConfig &config, uint32_t level)
{
    ROS_INFO("Reconfigure Request: %d %d %f %s %s %s %s %s %s %s %s\n",
             config.tensor_height, config.tensor_width, config.confidence, config.onnx_model_path.c_str(), config.onnx_label_path.c_str(),
             config.anchors_path.c_str(), config.tracker_type.c_str(), config.image_processing.c_str(), config.input_node_name.c_str(), 
             config.output_node_name.c_str(), config.debug ? "True" : "False");
    stopProcessor();
    startProcessor(config);
}

void OnnxTracker::startProcessor(ros_msft_onnx::reconfigConfig &config)
{

    // Parameters.
    _nhPrivate.setParam("tensor_height", config.tensor_height);
    _nhPrivate.setParam("tensor_width", config.tensor_width);
    _nhPrivate.setParam("confidence", config.confidence);
    _nhPrivate.setParam("onnx_model_path", config.onnx_model_path);
    _nhPrivate.setParam("onnx_label_path", config.onnx_label_path);
    _nhPrivate.setParam("anchors_path", config.anchors_path);
    _nhPrivate.setParam("tracker_type", config.tracker_type);
    _nhPrivate.setParam("image_processing", config.image_processing);
    _nhPrivate.setParam("input_node_name", config.input_node_name);
    _nhPrivate.setParam("output_node_name", config.output_node_name);
    _nhPrivate.setParam("debug", config.debug);
    std::string trackerType;
    if (_nhPrivate.getParam("tracker_type", trackerType))
    {
        if (trackerType == "yolo")
        {
            _processor = std::make_shared<yolo::YoloProcessor>();
        }
        else if (trackerType == "pose")
        {
            _processor = std::make_shared<pose::PoseProcessor>();
        }
    }

    if (_processor == nullptr)
    {
        ROS_INFO("ONNX: Processor not specified, selecting yolo as the default");
        _processor = std::make_shared<yolo::YoloProcessor>();
    }

    _status = _processor->init(_nh, _nhPrivate);
}

void ::OnnxTracker::stopProcessor()
{
    _nhPrivate.deleteParam("tensor_height");
    _nhPrivate.deleteParam("tensor_width");
    _nhPrivate.deleteParam("confidence");
    _nhPrivate.deleteParam("onnx_model_path");
    _nhPrivate.deleteParam("onnx_label_path");
    _nhPrivate.deleteParam("anchors_path");
    _nhPrivate.deleteParam("tracker_type");
    _nhPrivate.deleteParam("image_processing");
    _nhPrivate.deleteParam("input_node_name");
    _nhPrivate.deleteParam("output_node_name");
    _nhPrivate.deleteParam("debug");
}

bool OnnxTracker::shutdown()
{
    _nh.shutdown();
    _nhPrivate.shutdown();
    return true;
}
