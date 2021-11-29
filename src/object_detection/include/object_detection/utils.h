#pragma once

class BoundingBox {
    public:
        BoundingBox (float* data): 
            class_id_((int) (data[6]>data[5])),
            confidence_(data[4]),
            x_(data[0]),
            y_(data[1]),
            w_(data[2]),
            h_(data[3]),
            x_min_(data[0] - data[2]/2),
            x_max_(data[0] + data[2]/2),
            y_min_(data[1] - data[3]/2),
            y_max_(data[1] + data[3]/2),
            area_(data[2] * data[3])
            {}

        static bool sortComparisonFunction(const BoundingBox& bbox_0, const BoundingBox& bbox_1) {
            return bbox_0.confidence_ > bbox_1.confidence_;
        }

        float calculateIOU (const BoundingBox& bbox) {
            const float x_min_new = std::max(x_min_, bbox.x_min_);
            const float x_max_new = std::min(x_max_, bbox.x_max_);
            const float w_new = x_max_new - x_min_new;
            if (w_new <= 0.0f) {
                return 0.0f;
            }

            const float y_min_new = std::max(y_min_, bbox.y_min_);
            const float y_max_new = std::min(y_max_, bbox.y_max_);
            const float h_new = y_max_new - y_min_new;
            if (h_new <= 0.0f) {
                return 0.0f;
            }

            return w_new * h_new / (area_ + bbox.area_ - w_new * h_new);
        } 

        void compareWith(BoundingBox& bbox, const float thred_IOU) {
            if (bbox.valid_ == false || class_id_ != bbox.class_id_) {
                return;
            }

            if (calculateIOU(bbox) >= thred_IOU) {
                // ROS_INFO(
                //     "bbox0: tx = %.4f, ty = %.4f, tw = %.4f, th = %.4f", 
                //     x_, y_, w_, h_
                // );
                // ROS_INFO(
                //     "bbox1: tx = %.4f, ty = %.4f, tw = %.4f, th = %.4f", 
                //     bbox.x_, bbox.y_, bbox.w_, bbox.h_
                // );
                // ROS_INFO("IOU = %.4f\n", calculateIOU(bbox));
                bbox.valid_ = false;
            }
        }

        int class_id_;
        float confidence_;
        float x_; // center x
        float y_; // center y
        float w_; // width
        float h_; // height
        float x_min_;
        float x_max_;
        float y_min_;
        float y_max_;
        float area_;
        
        bool valid_ = true;
};