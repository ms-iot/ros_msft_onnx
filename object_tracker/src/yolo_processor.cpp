#include <rclcpp/rclcpp.hpp>
#include <object_tracker/yolo_processor.h>
#include <sensor_msgs/image_encodings.hpp>
#include <ament_index_cpp/get_resource.hpp>

#include <string>
#include <codecvt>
#include <locale>

using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;

static std::wstring to_wstring(std::string str)
{
    return strconverter.from_bytes(str);
}

const int ROW_COUNT = 13;
const int COL_COUNT = 13;
const int CHANNEL_COUNT = 125;
const int BOXES_PER_CELL = 5;
const int BOX_INFO_FEATURE_COUNT = 5;
const int CLASS_COUNT = 20;
const float CELL_WIDTH = 32;
const float CELL_HEIGHT = 32;

static const std::string labels[CLASS_COUNT] =
{
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

YoloProcessor::YoloProcessor()
: _process(ImageProcessing::Scale),
_tensorHeight(416),
_tensorWidth(416),
_confidence(0.7f)
{

}

static std::wstring GetTinyYOLOv2ModelPath()
{
    std::string content;
    std::string prefix_path;
    ament_index_cpp::get_resource("packages", "object_tracker", content, &prefix_path);
    return ::to_wstring(prefix_path + "/share/object_tracker/models/tinyyolov2-8.onnx");
}

void YoloProcessor::init()
{
    //*************************************************************************
    // initialize  enviroment...one enviroment per process
    // enviroment maintains thread pools and other state info
    _env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");

    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    _session = std::make_shared<Ort::Session>(*_env, GetTinyYOLOv2ModelPath().c_str(), session_options);

    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    _allocator = std::make_shared<Ort::AllocatorWithDefaultOptions>();

    // print number of model input nodes
    size_t num_input_nodes = _session->GetInputCount();
    _input_node_names.resize(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                            // Otherwise need vector<vector<>>

    printf("Number of inputs = %zu\n", num_input_nodes);

    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
        // print input node names
        char* input_name = _session->GetInputName(i, *_allocator);
        printf("Input %d : name=%s\n", i, input_name);
        _input_node_names[i] = input_name;

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
    _output_node_names.resize(num_output_nodes);
    std::vector<int64_t> output_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                            // Otherwise need vector<vector<>>

    printf("Number of outputs = %zu\n", num_output_nodes);

    // iterate over all output nodes
    for (int i = 0; i < num_output_nodes; i++) {
        // print output node names
        char* output_name = _session->GetOutputName(i, *_allocator);
        printf("Output %d : name=%s\n", i, output_name);
        _output_node_names[i] = output_name;

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

void YoloProcessor::ProcessImage(const sensor_msgs::msg::Image::SharedPtr image) 
{
    // Convert back to an OpenCV Image
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);

    cv::Size s = cv_ptr->image.size();
    cv::Rect ROI((s.width - _tensorWidth) / 2, (s.height - _tensorHeight) / 2, _tensorWidth, _tensorHeight);
    cv::Mat image_resized = cv_ptr->image(ROI);

    // Convert to RGB
    cv::Mat rgb_image;
    cv::cvtColor(image_resized, rgb_image, cv::COLOR_BGR2RGB);

    // Set the image to 32-bit floating point values for tensorization.
    cv::Mat image_32_bit;
    rgb_image.convertTo(image_32_bit, CV_32F);

    // Extract color channels from interleaved data
    cv::Mat channels[3];
    cv::split(image_32_bit, channels);

    size_t input_tensor_size = _tensorHeight * _tensorWidth * 3;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!

    std::vector<float> input_tensor_values(input_tensor_size);

    memcpy(&input_tensor_values[0], (float *)channels[0].data, _tensorWidth * _tensorHeight * sizeof(float));
    memcpy(&input_tensor_values[_tensorWidth * _tensorHeight], (float *)channels[1].data, _tensorWidth * _tensorHeight * sizeof(float));
    memcpy(&input_tensor_values[2 * _tensorWidth * _tensorHeight], (float *)channels[2].data, _tensorWidth * _tensorHeight * sizeof(float));

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::TypeInfo type_info = _session->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_node_dims = {1, 3, 416, 416};
    //input_node_dims = tensor_info.GetShape();
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

    ProcessOutput(output, image_resized);
}

void YoloProcessor::ProcessOutput(std::vector<float> output, cv::Mat& image)
{
    auto boxes = GetRecognizedObjects(output, _confidence);

    // If we found a person, send a message
    for (std::vector<YoloBox>::iterator it = boxes.begin(); it != boxes.end(); ++it)
    {
        printf("%s\n", it->label);
    }
}

std::vector<YoloBox> YoloProcessor::GetRecognizedObjects(std::vector<float> modelOutputs, float threshold)
{
    static float anchors[] =
    {
        1.08f, 1.19f, 3.42f, 4.41f, 6.63f, 11.38f, 9.42f, 5.11f, 16.62f, 10.52f
    };
    static int featuresPerBox = BOX_INFO_FEATURE_COUNT + CLASS_COUNT;
    static int stride = featuresPerBox * BOXES_PER_CELL;

    std::vector<YoloBox> boxes;

    for (int cy = 0; cy < ROW_COUNT; cy++)
    {
        for (int cx = 0; cx < COL_COUNT; cx++)
        {
            for (int b = 0; b < BOXES_PER_CELL; b++)
            {
                int channel = (b * (CLASS_COUNT + BOX_INFO_FEATURE_COUNT));
                float tx = modelOutputs[GetOffset(cx, cy, channel)];
                float ty = modelOutputs[GetOffset(cx, cy, channel + 1)];
                float tw = modelOutputs[GetOffset(cx, cy, channel + 2)];
                float th = modelOutputs[GetOffset(cx, cy, channel + 3)];
                float tc = modelOutputs[GetOffset(cx, cy, channel + 4)];

                float x = ((float)cx + Sigmoid(tx)) * CELL_WIDTH;
                float y = ((float)cy + Sigmoid(ty)) * CELL_HEIGHT;
                float width = (float)exp(tw) * CELL_WIDTH * anchors[b * 2];
                float height = (float)exp(th) * CELL_HEIGHT * anchors[b * 2 + 1];

                float confidence = Sigmoid(tc);
                if (confidence < threshold)
                    continue;

                std::vector<float> classes(CLASS_COUNT);
                float classOffset = channel + BOX_INFO_FEATURE_COUNT;

                for (int i = 0; i < CLASS_COUNT; i++)
                    classes[i] = modelOutputs[GetOffset(cx, cy, i + classOffset)];

                Softmax(classes);

                // Get the index of the top score and its value
                auto iter = std::max_element(classes.begin(), classes.end());
                float topScore = (*iter) * confidence;
                int topClass = std::distance(classes.begin(), iter);

                if (topScore < threshold)
                    continue;

                YoloBox top_box = {
                    labels[topClass],
                    (x - width / 2),
                    (y - height / 2),
                    width,
                    height,
                    topScore
                };
                boxes.push_back(top_box);
            }
        }
    }

    return boxes;
}

int YoloProcessor::GetOffset(int x, int y, int channel)
{
    // YOLO outputs a tensor that has a shape of 125x13x13, which 
    // WinML flattens into a 1D array.  To access a specific channel 
    // for a given (x,y) cell position, we need to calculate an offset
    // into the array
    static int channelStride = ROW_COUNT * COL_COUNT;
    return (channel * channelStride) + (y * COL_COUNT) + x;
}

float YoloProcessor::Sigmoid(float value)
{
    float k = (float)std::exp(value);
    return k / (1.0f + k);
}

void YoloProcessor::Softmax(std::vector<float> &values)
{
    float max_val{ *std::max_element(values.begin(), values.end()) };
    std::transform(values.begin(), values.end(), values.begin(),
        [&](float x) { return std::exp(x - max_val); });

    float exptot = std::accumulate(values.begin(), values.end(), 0.0);
    std::transform(values.begin(), values.end(), values.begin(),
        [&](float x) { return (float)(x / exptot); });
}
