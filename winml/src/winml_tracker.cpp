#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/LinearMath/Quaternion.h>
#include <opencv2/opencv.hpp>


// Include ROS files before Windows, as there are overlapping symbols
#include <vcruntime.h>
#include <windows.h>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Graphics.h>
#include <winrt/Windows.Graphics.Imaging.h>

#include "winml_tracker/winml_tracker.h"
#include "winml_tracker/yolo_box.h"
#include "winml_tracker/pose_parser.h"

#include <string>
#include <codecvt>
#include <fstream>
#include <sstream>

using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Windows::AI::MachineLearning;
using namespace Windows::Media;
using namespace Windows::Graphics::Imaging;
using namespace Windows::Storage;
using namespace std;

using convert_type = std::codecvt_utf8<wchar_t>;
using wstring_to_utf8 = std::wstring_convert<convert_type, wchar_t>;

const uint32_t kDefaultTensorWidth = 416;
const uint32_t kDefaultTensorHeight = 416;

std::string strFromHstring(hstring hstr)
{
    auto h = hstr.c_str();
    auto converted = wstring_to_utf8().to_bytes(h);
    return converted;
}

WinMLProcessor::WinMLProcessor()
: _process(ImageProcessing::Scale)
, _confidence(0.70f)
, _debug(false)
, _normalize(false)
{

}

bool WinMLProcessor::init(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate)
{
    std::string imageProcessingType;
    if (nhPrivate.getParam("image_processing", imageProcessingType))
    {
        if (imageProcessingType == "crop")
        {
            _process = Crop;
        }
        else if (imageProcessingType == "scale")
        {
            _process = Scale;
        }
        else if (imageProcessingType == "resize")
        {
            _process = Resize;
        }
        else
        {
            ROS_WARN("WINML: unknown image processing type: %s", imageProcessingType);
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

    nhPrivate.param<bool>("winml_fake", _fake, false);
    
    nhPrivate.getParam("link_name", _linkName);

    if (!nhPrivate.getParam("onnx_model_path", _onnxModel) ||
        _onnxModel.empty())
    {
        ROS_ERROR("WINML: onnx_model_path parameter has not been set.");
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
        imageTopic = "/cv_camera/image_raw";
    }

    _detect_pub = nh.advertise<visualization_msgs::MarkerArray>("tracked_objects", 1);

    image_transport::ImageTransport it(nh);
    _cameraSub = it.subscribe(imageTopic.c_str(), 1, &WinMLProcessor::ProcessImage, this);
    _image_pub = it.advertise("tracked_objects/image", 1);
    try
    {
        // Load the ML model
        hstring modelPath = hstring(wstring_to_utf8().from_bytes(_onnxModel));
        _model = LearningModel::LoadFromFilePath(modelPath);

        // Create a WinML session
        _session = LearningModelSession(_model, LearningModelDevice(LearningModelDeviceKind::Cpu));
    }
    catch (hresult_error const& e)
    {
        ROS_ERROR("WINML: Failed to Start ML Session!: %s", strFromHstring(e.message()).c_str());
        return false;
    }
    catch (std::exception& e)
    {
        ROS_ERROR("WINML: Failed to Start ML Session: %s", e.what());
        return false;
    }

    return true;
}

void WinMLProcessor::ProcessImage(const sensor_msgs::ImageConstPtr& image) 
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
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("WINML: cv_bridge exception: %s", e.what());
        return;
    }

    cv::Size mlSize(_tensorWidth, _tensorHeight);
    cv::Mat rgb_image;
    cv::Mat image_resized;
    cv::Size s = cv_ptr->image.size();
    float aspectRatio = (float)s.width / (float)s.height;
    if (s.width <= 0 || s.height <= 0)
    {
        ROS_ERROR("WINML: irrational image size received; one dimention zero or less");
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

    // Setup the model binding
    LearningModelBinding binding(_session);
    vector<int64_t> grid_shape({ 1, _channelCount, _colCount, _rowCount});
    try
    {
        binding.Bind(_outName, TensorFloat::Create(grid_shape));
    }
    catch (hresult_error const& e)
    {
        ROS_ERROR("WINML: Failed to bind!: %s", strFromHstring(e.message()).c_str());
        return;
    }

    // Create a Tensor from the CV Mat and bind it to the session
    std::vector<float> image_data(1 * 3 * _tensorWidth * _tensorHeight);
    memcpy(&image_data[0], (float *)channels[0].data, _tensorWidth * _tensorHeight * sizeof(float));
    memcpy(&image_data[_tensorWidth * _tensorHeight], (float *)channels[1].data, _tensorWidth * _tensorHeight * sizeof(float));
    memcpy(&image_data[2 * _tensorWidth * _tensorHeight], (float *)channels[2].data, _tensorWidth * _tensorHeight * sizeof(float));
    TensorFloat image_tensor = TensorFloat::CreateFromArray({ 1, 3, _tensorWidth, _tensorHeight }, image_data);

    try
    {
        binding.Bind(_inName, image_tensor);
    }
    catch (hresult_error const& e)
    {
        ROS_ERROR("WINML: Failed to bind!: %s", strFromHstring(e.message()).c_str());
        return;
    }
    catch (std::exception& e)
    {
        ROS_ERROR("WINML: Failed to bind!: %s", e.what());
        return;
    }

    if (_fake)
    {
        std::vector<float> grids;
        ProcessOutput(grids, image_resized);
    }
    else
    {
        // Call WinML
        try
        {
            auto results = _session.Evaluate(binding, L"RunId");
            if (!results.Succeeded())
            {
                ROS_ERROR("WINML: Evaluation of object tracker failed!");
                return;
            }

            // Convert the results to a vector and parse the bounding boxes
            auto grid_result = results.Outputs().Lookup(_outName).as<TensorFloat>().GetAsVectorView();
            std::vector<float> grids(grid_result.Size());
            winrt::array_view<float> grid_view(grids);
            grid_result.GetMany(0, grid_view);

            ProcessOutput(grids, image_resized);
        }
        catch (hresult_error const& e)
        {
            ROS_ERROR("WINML: Evaluation of object tracker failed!: %s", strFromHstring(e.message()));
            return;
        }
        catch (std::exception& e)
        {
            ROS_ERROR("WINML: Evaluation of object tracker failed!: %s", e.what());
            return;
        }
    }

    return;
}

bool WinMLTracker::init(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate)
{
    _nh = nh;
    _nhPrivate = nhPrivate;

    // Parameters.
    std::string trackerType;
    if (nhPrivate.getParam("tracker_type", trackerType))
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
        ROS_INFO("WINML: Processor not specified, selecting yolo as the default");
        _processor = std::make_shared<yolo::YoloProcessor>();
    }

    return _processor->init(_nh, nhPrivate);
}

bool WinMLTracker::shutdown()
{
    _nh.shutdown();
    _nhPrivate.shutdown();

    return true;
}
