# Cola Object Detection

This repository serves as the supplementary example for the tutorial on the [MRSD wiki page](https://roboticsknowledgebase.com/wiki/sensing/yolov5-tensort/). Under `src`, `object_detection` is the main package for YOLOv5. Its node subscribes to the rectified camera image and publishes bounding boxes. `robot_msg_definitions` defines the ROS msg files, which can be shared across different machines. `spinnaker_camera_driver` is the ROS API package of our camera, which could be replaced if a different camera is used.

# Dependencies

- ROS
- OpenCV
- CUDA
- TensorRT