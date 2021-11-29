#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
// #include <opencv2/cudaarithm.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
// #include <cuda.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <robot_msg_definitions/bounding_box_2d_depth.h>
#include <robot_msg_definitions/bounding_boxes_2d_depth.h>
#include <ros/package.h>
#include <ros/ros.h>

#include <object_detection/logging.h>
#include <object_detection/utils.h>

#define CUDA_CHECK(callstr)                                                    \
  {                                                                            \
    cudaError_t error_code = callstr;                                          \
    if (error_code != cudaSuccess) {                                           \
      std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":"    \
                << __LINE__;                                                   \
      assert(0);                                                               \
    }                                                                          \
  }

constexpr float NMS_THRESH = 0.4;
constexpr float CONF_THRESH = 0.5;
constexpr size_t MAX_OUTPUT_BBOX_COUNT = 1000;

std::string engine_filename;
int image_size = -1;
enum ObjectClass { CLASS_FORKLIFT = 0, CLASS_PEDESTRIAN = 1, NUM_CLASS = 2 };
int input_size = -1;
int output_size = -1;
// total IO ports, 1 input, 3 intermediate outputs, 1 final output
constexpr int BUFFER_SIZE = 5;

sample::Logger gLogger;

nvinfer1::ICudaEngine *engine;
nvinfer1::IExecutionContext *context;
nvinfer1::IRuntime *runtime;
cudaStream_t stream;
std::vector<void *> buffers(BUFFER_SIZE); // buffers for input and output data
std::shared_ptr<float[]> input_data;
std::shared_ptr<float[]> output_data;

image_transport::Subscriber image_sub;
ros::Subscriber robot_camera_info_sub;
ros::Publisher robot_camera_info_pub;
image_transport::Publisher image_pub;
sensor_msgs::ImagePtr image_ptr_output_msg;
ros::Publisher bboxes_publisher;
robot_msg_definitions::bounding_boxes_2d_depth bboxes_output_msg;

// initialize invalid values
boost::array<double, 12> P_orig = {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                   -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
boost::array<double, 12> P_pub;
sensor_msgs::CameraInfo camera_info_outward;

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims &dims) {
  size_t size = 1;
  for (size_t i = 0; i < dims.nbDims; ++i) {
    size *= dims.d[i];

    std::cout << dims.d[i] << ", ";
  }
  std::cout << std::endl;

  return size;
}

void prepareEngine() {
  const std::string engine_path =
      ros::package::getPath("object_detection") + "/engines/" + engine_filename;
  ROS_INFO("engine_path = %s", engine_path.c_str());
  std::ifstream engine_file(engine_path, std::ios::binary);

  if (!engine_file.good()) {
    ROS_ERROR("no such engine file: %s", engine_path.c_str());
    return;
  }

  char *trt_model_stream = nullptr;
  size_t trt_stream_size = 0;
  engine_file.seekg(0, engine_file.end);
  trt_stream_size = engine_file.tellg();
  engine_file.seekg(0, engine_file.beg);
  trt_model_stream = new char[trt_stream_size];
  assert(trt_model_stream);
  engine_file.read(trt_model_stream, trt_stream_size);
  engine_file.close();

  runtime = nvinfer1::createInferRuntime(gLogger);
  assert(runtime != nullptr);
  engine = runtime->deserializeCudaEngine(trt_model_stream, trt_stream_size);
  assert(engine != nullptr);
  context = engine->createExecutionContext();
  assert(context != nullptr);
  if (engine->getNbBindings() != BUFFER_SIZE) {
    ROS_ERROR("engine->getNbBindings() == %d, but should be %d",
              engine->getNbBindings(), BUFFER_SIZE);
  }

  // get sizes of input and output and allocate memory required for input data
  // and for output data
  std::vector<nvinfer1::Dims> input_dims;
  std::vector<nvinfer1::Dims> output_dims;
  for (size_t i = 0; i < engine->getNbBindings(); ++i) {
    const size_t binding_size =
        getSizeByDim(engine->getBindingDimensions(i)) * sizeof(float);
    if (binding_size == 0) {
      ROS_ERROR("binding_size == 0");

      delete[] trt_model_stream;
      return;
    }

    cudaMalloc(&buffers[i], binding_size);
    if (engine->bindingIsInput(i)) {
      input_dims.emplace_back(engine->getBindingDimensions(i));
      ROS_INFO("Input layer, size = %lu", binding_size);
    } else {
      output_dims.emplace_back(engine->getBindingDimensions(i));
      ROS_INFO("Output layer, size = %lu", binding_size);
    }
  }

  CUDA_CHECK(cudaStreamCreate(&stream));

  delete[] trt_model_stream;

  ROS_INFO("Engine preparation finished");
}

void nonMaximumSuppression(std::vector<std::vector<BoundingBox>> &bboxes) {
  for (int c = 0; c < ObjectClass::NUM_CLASS; ++c) {
    bboxes[c].reserve(output_size);
  }

  for (int i = 0; i < output_size; i += 7) {
    const float conf = output_data.get()[i + 4];
    if (conf > CONF_THRESH) {
      // ROS_INFO(
      //     "object detected: class = %d, conf = %.4f, tx = %.4f, ty = %.4f, tw
      //     = %.4f, th = %.4f", (int)
      //     (output_data.get()[i+6]>output_data.get()[i+5]),
      //     output_data.get()[i+4],
      //     output_data.get()[i+0],
      //     output_data.get()[i+1],
      //     output_data.get()[i+2],
      //     output_data.get()[i+3]
      // );

      assert(conf <= 1.0f);

      const int class_id =
          (int)(output_data.get()[i + 6] > output_data.get()[i + 5]);
      bboxes[class_id].push_back(BoundingBox(output_data.get() + i));
    }
  }

  for (int c = 0; c < ObjectClass::NUM_CLASS; ++c) {
    // ROS_INFO("class %d has %lu available bboxes", c, bboxes[c].size());

    std::sort(bboxes[c].begin(), bboxes[c].end(),
              BoundingBox::sortComparisonFunction);
    const size_t bboxes_size = bboxes[c].size();
    size_t valid_count = 0;

    for (size_t i = 0; i < bboxes_size && valid_count < MAX_OUTPUT_BBOX_COUNT;
         ++i) {
      if (!bboxes[c][i].valid_) {
        continue;
      }

      for (size_t j = i + 1; j < bboxes_size; ++j) {
        bboxes[c][i].compareWith(bboxes[c][j], NMS_THRESH);
      }

      ++valid_count;
    }

    // ROS_INFO("class %d has %lu valid bboxes", c, valid_count);
  }
}

cv::Mat PreprocessImage(const cv::Mat &image_orig) {
  cv::Mat image_new;

  // Step 1: CropImageToSquare

  const int rows_orig = image_orig.rows;
  const int cols_orig = image_orig.cols;
  const int cropped_size = std::min(rows_orig, cols_orig);

  // if P matrix initialized
  if (P_orig[0] > 0) {
    P_pub = P_orig;
  }

  if (rows_orig != cols_orig) {
    if (cropped_size < 100) {
      ROS_WARN("the new size after cropping is too small: %d", cropped_size);
    }

    const int pixel_start = std::abs(rows_orig - cols_orig) / 2;
    const int pixel_end = pixel_start + cropped_size;

    if (rows_orig > cols_orig) {
      image_new = image_orig(cv::Range(pixel_start, pixel_end),
                             cv::Range(0, cols_orig));
      // y offset should be decreased
      // ref: https://ksimek.github.io/2013/08/13/intrinsic/
      P_pub[6] -= static_cast<double>(pixel_start);
    } else {
      // rows < cols; y < x
      // eg. rows = 1080, cols = 1440
      image_new = image_orig(cv::Range(0, rows_orig),
                             cv::Range(pixel_start, pixel_end));
      // x offset should be decreased
      // ref: https://ksimek.github.io/2013/08/13/intrinsic/
      P_pub[2] -= static_cast<double>(pixel_start);
    }
  } else {
    image_new = image_orig.clone();
  }

  if (image_new.rows != image_new.cols) {
    ROS_ERROR("After cropping, rows (%d) != cols (%d)", image_new.rows,
              image_new.cols);
  }

  // Step 2: Resize Image
  const double dst_size_over_src_size =
      static_cast<double>(image_size) / static_cast<double>(cropped_size);
  cv::resize(image_new, image_new, cv::Size(image_size, image_size), 0.0, 0.0,
             cv::INTER_LINEAR);

  // if P matrix initialized
  if (P_orig[0] > 0) {
    for (int i = 0; i < 8; ++i) {
      P_pub[i] *= dst_size_over_src_size;
    }
  }

  return image_new;
}

void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
  // float* gpu_input = (float *) buffers[0];

  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat image = PreprocessImage(cv_ptr->image);

  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  image.convertTo(image, CV_32FC3, 1.f / 255.f);

  if (image.empty()) {
    ROS_ERROR("image is empty");
    return;
  }
  if (image.rows != image_size || image.cols != image_size) {
    ROS_ERROR("image size is (%d, %d), not equal to (%d, %d)", image.rows,
              image.cols, image_size, image_size);
    return;
  }

  // cv::cuda::GpuMat gpu_image;
  // gpu_image.upload(image);

  // std::vector<cv::cuda::GpuMat> chw;
  // const cv::Size input_size = cv::Size(image_size, image_size);
  // for (size_t i = 0; i < 3; ++i) // 3 is the channel number: R G B
  // {
  //     chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i *
  //     image_size * image_size));
  // }
  // // cv::cuda::split(image, chw);

  int i = 0;
  for (int row = 0; row < image_size; ++row) {
    for (int col = 0; col < image_size; ++col) {
      input_data.get()[i] = image.at<cv::Vec3f>(row, col)[0];
      input_data.get()[i + image_size * image_size] =
          image.at<cv::Vec3f>(row, col)[1];
      input_data.get()[i + 2 * image_size * image_size] =
          image.at<cv::Vec3f>(row, col)[2];
      ++i;
    }
  }

  auto start = std::chrono::system_clock::now();

  CUDA_CHECK(cudaMemcpyAsync(buffers[0], input_data.get(),
                             input_size * sizeof(float), cudaMemcpyHostToDevice,
                             stream));
  context->enqueue(1, buffers.data(), stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output_data.get(), buffers[4],
                             output_size * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  auto end = std::chrono::system_clock::now();
  // ROS_INFO("YOLO takes %d ms",
  //          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
  //              .count());

  std::vector<std::vector<BoundingBox>> bboxes(ObjectClass::NUM_CLASS);
  nonMaximumSuppression(bboxes);

  // cv::Mat image_labelled;
  // cv::cvtColor(image, image_labelled, cv::COLOR_BGR2RGB);
  const std::vector<cv::Scalar> colors{
      cv::Scalar(0, 255, 0), // green for class 0
      cv::Scalar(0, 0, 255)  // red for class 1
  };

  // turn bboxes into the form of ros msg, and publish them
  bboxes_output_msg.bboxes.clear();
  for (int c = 0; c < ObjectClass::NUM_CLASS; ++c) {
    for (const auto &bbox : bboxes[c]) {
      if (!bbox.valid_) {
        continue;
      }

      const cv::Rect rect(bbox.x_min_, bbox.y_min_, bbox.w_, bbox.h_);
      cv::rectangle(image, rect, colors[c]);

      robot_msg_definitions::bounding_box_2d_depth bbox_tmp;
      // note that the meaning of the message type and the entries in the
      // message is not consistent with the actual implication
      bbox_tmp.class_id = c;
      bbox_tmp.min_x = bbox.x_min_;
      bbox_tmp.min_y = bbox.y_min_;
      bbox_tmp.dx = bbox.w_;
      bbox_tmp.dy = bbox.h_;
      bbox_tmp.x_3d = -1.0;
      bbox_tmp.z_3d = -1.0;
      bboxes_output_msg.bboxes.push_back(bbox_tmp);
    }
  }
  bboxes_output_msg.header.stamp = ros::Time::now();
  bboxes_publisher.publish(bboxes_output_msg);

  image.convertTo(image, CV_8UC3, 255.0f);
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
  std_msgs::Header image_ptr_output_msg_header;
  image_ptr_output_msg_header.stamp = ros::Time::now();
  image_ptr_output_msg =
      cv_bridge::CvImage(image_ptr_output_msg_header, "bgr8", image)
          .toImageMsg();
  image_pub.publish(image_ptr_output_msg);

  // if P matrix initialized
  if (P_orig[0] > 0) {
    camera_info_outward.header.stamp = ros::Time::now();
    camera_info_outward.height = image_size;
    camera_info_outward.width = image_size;
    camera_info_outward.P = P_pub;

    robot_camera_info_pub.publish(camera_info_outward);
  }
}

void cameraInfoCallback(const sensor_msgs::CameraInfo msg) {
  // for (auto n:msg.D)
  //   ROS_INFO("camera info%f", n);
  P_orig = msg.P;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "object_detection_node");

  ros::NodeHandle nh_private("~");
  nh_private.getParam("engine_filename", engine_filename);
  nh_private.getParam("image_size", image_size);
  input_size = 3 * image_size * image_size;
  output_size =
      3 * image_size / 8 * image_size / 8 * (5 + ObjectClass::NUM_CLASS) +
      3 * image_size / 16 * image_size / 16 * (5 + ObjectClass::NUM_CLASS) +
      3 * image_size / 32 * image_size / 32 * (5 + ObjectClass::NUM_CLASS);
  input_data = std::shared_ptr<float[]>(new float[input_size]);
  output_data = std::shared_ptr<float[]>(new float[output_size]);

  prepareEngine();

  ros::NodeHandle nh;

  image_transport::ImageTransport image_transporter(nh);
  image_sub = image_transporter.subscribe("/robot_camera/image_rect_color", 10,
                                          imageCallback);
  robot_camera_info_sub =
      nh.subscribe("/robot_camera/camera_info", 1000, cameraInfoCallback);
  robot_camera_info_pub = nh.advertise<sensor_msgs::CameraInfo>(
      "/object_detection/camera_info", 1000);
  image_pub = image_transporter.advertise("/object_detection/image", 10);
  bboxes_publisher =
      nh.advertise<robot_msg_definitions::bounding_boxes_2d_depth>(
          "/object_detection/bboxes", 10);

  ros::Rate loop_rate(50);
  ROS_INFO("In object_detection_node\n");
  while (ros::ok()) {
    ros::spinOnce();
    loop_rate.sleep();
  }

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(buffers[0]));
  CUDA_CHECK(cudaFree(buffers[1]));
  CUDA_CHECK(cudaFree(buffers[2]));
  CUDA_CHECK(cudaFree(buffers[3]));
  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  return 0;
}
