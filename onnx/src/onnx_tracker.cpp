#include <onnx_object_detection/onnx_tracker.h>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>

const uint32_t kDefaultTensorWidth = 416;
const uint32_t kDefaultTensorHeight = 416;

static std::wstring to_wstring(std::string str)
{
    return strconverter.from_bytes(str);
}

/////////////////////// ONNX PROCESSOR /////////////////////////

OnnxProcessor::OnnxProcessor(): 
_confidence (0.70f)
,_debug(false)
,_normalize(false)
{

}

virtual void init(rclcpp::Node::SharedPtr& node)
{
    _node = node;
    _node->get_parameter("confidence", _confidence);
    _node->get_parameter("debug", _debug);
    _node->get_parameter("link_name", _linkName);
    _fake = false;

    int temp = 0;
    if (_node->get_parameter("tensor_width", temp) && temp > 0)
    {
        _tensorWidth = (uint)temp;
    }
    else 
    {
        _tensorWidth = kDefaultTensorWidth;
    }

    temp = 0;
    if ( _node->get_parameter("tensor_height", temp) && temp > 0)
    {
        _tensorHeight = (uint)temp;
    }
    else 
    {
        _tensorHeight = kDefaultTensorHeight;
    }

    if (!_node->get_parameter("onnx_model_path", _onnxModel) ||
        _onnxModel.empty())
    {
        RCLCPP_ERROR("Onnx: onnx_model_path parameter has not been set.");
    }

    if (_node->get_parameter("calibration", _calibration))
    {
        try
        {
            cv::FileStorage fs(_calibration, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
            fs["camera_matrix"] >> _camera_matrix;
            fs["distortion_coefficients"] >> _dist_coeffs;
        }
        catch (std::exception &e)
        {
            RCLCPP_ERROR(_node->get_logger(),"Failed to read the calibration file, continuing without calibration.\n%s", e.what());
            // no calibration for you.
            _calibration = "";
        }
    }

    std::string image_topic_ = "image_raw";
    std::string visual_marker_topic_ = "visual_markers";
    std::string image_pub_topic_ = "image_debug_raw";
    std::string detect_pose_topic_ = "detected_object";

    publisher_ = _node->create_publisher<visualization_msgs::msg::Marker>(visual_marker_topic_, 10);
    image_pub_ = _node->create_publisher<sensor_msgs::msg::Image>(image_pub_topic_, 10);
    subscription_ = _node->create_subscription<sensor_msgs::msg::Image>(
            image_topic_, 10, std::bind(&OnnxProcessor::ProcessImage, _node, _1));
    detect_pose_pub_ = _node->create_publisher<onnx_msgs::DetectedObjectPose>(detect_pose_topic_, 1); 
    
    // Generate onnx session
    //*************************************************************************
    // initialize  enviroment...one enviroment per process
    // enviroment maintains thread pools and other state info
    _env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");

    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
    auto modelFullPath = ::to_wstring(_onnxModel).c_str();
#else
    auto modelFullPath = _onnxModel.c_str();
#endif

    _session = std::make_shared<Ort::Session>(*_env, modelFullPath, session_options);
    _allocator = std::make_shared<Ort::AllocatorWithDefaultOptions>();
    DumpParameters();

}

void OnnxProcessor::ProcessImage(const sensor_msgs::msg::Image::SharedPtr msg) 
{
    // Convert back to an OpenCV Image
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    cv::Size s = cv_ptr->image.size();
    cv::Rect ROI((s.width - 416) / 2, (s.height - 416) / 2, 416, 416);
    cv::Mat image = cv_ptr->image(ROI);

    // Convert to RGB
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

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

    // TODO: [insert new onnx code] In the WinML version this is where the binding was created 
    // It depended on yolo/pose processor vars like _channelCount, _colCount, _rowCount
    // TODO: Add in "if(_fake)" flow

    size_t input_tensor_size = _tensorHeight * _tensorWidth * 3;

    std::vector<float> input_tensor_values(input_tensor_size);

    memcpy(&input_tensor_values[0], (float *)channels[0].data, _tensorWidth * _tensorHeight * sizeof(float));
    memcpy(&input_tensor_values[_tensorWidth * _tensorHeight], (float *)channels[1].data, _tensorWidth * _tensorHeight * sizeof(float));
    memcpy(&input_tensor_values[2 * _tensorWidth * _tensorHeight], (float *)channels[2].data, _tensorWidth * _tensorHeight * sizeof(float));

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::TypeInfo type_info = _session->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_node_dims = {1, 3, 416, 416};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());

    // score model & input tensor, get back output tensor
    std::vector<const char*> input_node_names = {"image"};
    std::vector<const char*> output_node_names = {"grid"};
    auto output_tensors = _session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);

    auto &output_tensor = output_tensors.front();
    auto output_type_info = output_tensor.GetTensorTypeAndShapeInfo();
    size_t output_total_len = output_type_info.GetElementCount();

    float* floatarr = output_tensor.GetTensorMutableData<float>();
    std::vector<float> output;
    output.resize(output_total_len);
    memcpy(&output[0], floatarr, output_total_len * sizeof(float));

    // TODO: [insert new onnx code] In the WinML version this is where the results were retrieved and passed on to the process output function
    // [replace with new onnx code below]
    auto results = _session.Evaluate(binding, L"RunId"); 
    if (!results.Succeeded())
    {
        RCLCPP_ERROR(_node->get_logger(), "ONNX: Evaluation of object tracker failed!");
        return;
    }

    // Convert the results to a vector and parse the bounding boxes
    auto grid_result = results.Outputs().Lookup(_outName).as<TensorFloat>().GetAsVectorView();
    std::vector<float> grids(grid_result.Size());
    winrt::array_view<float> grid_view(grids);
    grid_result.GetMany(0, grid_view);

    ProcessOutput(grids, image_resized);
    // [replace with new onnx code above]
}

void OnnxProcessor::DumpParameters()
{
    //*************************************************************************
    // print model input layer (node names, types, shape etc.)

    // print number of model input nodes
    size_t num_input_nodes = _session->GetInputCount();
    std::vector<const char*> input_node_names;
    input_node_names.resize(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                            // Otherwise need vector<vector<>>

    printf("Number of inputs = %zu\n", num_input_nodes);

    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
        // print input node names
        char* input_name = _session->GetInputName(i, *_allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = _session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    }

    //*************************************************************************
    // print model output layer (node names, types, shape etc.)
    size_t num_output_nodes = _session->GetOutputCount();
    std::vector<const char*> output_node_names;
    output_node_names.resize(num_output_nodes);
    std::vector<int64_t> output_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                            // Otherwise need vector<vector<>>

    printf("Number of outputs = %zu\n", num_output_nodes);

    // iterate over all output nodes
    for (int i = 0; i < num_output_nodes; i++) {
        // print output node names
        char* output_name = _session->GetOutputName(i, *_allocator);
        printf("Output %d : name=%s\n", i, output_name);
        output_node_names[i] = output_name;

        // print input node types
        Ort::TypeInfo type_info = _session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Output %d : type=%d\n", i, type);

        // print input shapes/dims
        output_node_dims = tensor_info.GetShape();
        printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
        for (int j = 0; j < output_node_dims.size(); j++)
            printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    }
}


/////////////////////////////////////////////// ONNX TRACKER //////////////////////////////////////////////////
void OnnxTracker::init(rclcpp::Node::SharedPtr& node)
{
    // Parameters.
    std::string trackerType;
    if (node->get_parameter("tracker_type", trackerType))
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

    // Tracker time was no specified 
    if (_processor == nullptr)
    {
        RCLCPP_INFO(node->get_logger(), "Onnx Tracker: Processor not specified, selecting yolo as the default");
        _processor = std::make_shared<yolo::YoloProcessor>();
    }

    _processor->init(node);
}