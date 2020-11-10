#pragma once

class WinMLProcessor
{
public:
    WinMLProcessor();

    void ProcessImage(const sensor_msgs::ImageConstPtr& image);

    typedef enum 
    {
        Scale,
        Crop,
        Resize
    } ImageProcessing;

    void setImageProcessing(ImageProcessing process)
    {
        _process = process;
    }

    virtual bool init(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate);

private:
    ImageProcessing _process;

protected:
    virtual void ProcessOutput(std::vector<float> output, cv::Mat& image) = 0;

    bool _fake;
    winrt::hstring _inName;
    winrt::hstring _outName;
    std::string _linkName;
    std::string _onnxModel;
    std::string _calibration;

    cv::Mat _camera_matrix;
    cv::Mat _dist_coeffs;

    
    float _confidence;

    bool _debug;
    bool _normalize;

    uint _tensorWidth;
    uint _tensorHeight;

    int _channelCount;
    int _rowCount;
    int _colCount;
    winrt::Windows::AI::MachineLearning::LearningModel _model = nullptr;
    winrt::Windows::AI::MachineLearning::LearningModelSession _session = nullptr;

    ros::Publisher _detect_pub;
    image_transport::Publisher _image_pub;
    image_transport::Publisher _debug_image_pub;
    image_transport::Subscriber _cameraSub;


};

class WinMLTracker
{
    ros::NodeHandle _nh;
    ros::NodeHandle _nhPrivate;

    std::shared_ptr<WinMLProcessor> _processor;

public:
    WinMLTracker() { };

    bool init(ros::NodeHandle& nh, ros::NodeHandle& nhPrivate);
    bool shutdown();
};

