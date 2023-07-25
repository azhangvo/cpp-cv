#include <iostream>
#include <Eigen/Dense>
#include <zmq.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>

using Eigen::MatrixXd;
using namespace std;

void Gaussian(const cv::Mat &src, cv::Mat &dst, int kernel_size = 5, double sigma = 1.0) {
    assert(kernel_size % 2 == 1 && kernel_size > 0);
    int k = (kernel_size - 1) / 2;

    double kernel[kernel_size][kernel_size];
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j <= i; j++) {
            double val =
                    1.0 / (2 * M_PI * sigma * sigma) *
                    exp(-((i - k) * (i - k) + (j - k) * (j - k)) / 2.0 / sigma / sigma);
            kernel[i][j] = val;
            kernel[j][i] = val;
        }
    }

    cv::Mat kernel_mat(kernel_size, kernel_size, CV_64F, kernel);
    cv::filter2D(src, dst, CV_32F, kernel_mat, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
}

void Sobel(const cv::Mat &src, cv::Mat &magnitude, cv::Mat &phase) {
    float x_kernel_data[3][3] = {{-1, 0, 1},
                                 {-2, 0, 2},
                                 {-1, 0, 1}};
    float y_kernel_data[3][3] = {{1,  2,  1},
                                 {0,  0,  0},
                                 {-1, -2, -1}};
    cv::Mat x_res, y_res, x_kernel(3, 3, CV_32F, x_kernel_data), y_kernel(3, 3, CV_32F, y_kernel_data);

    cv::filter2D(src, x_res, CV_32F, x_kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(src, y_res, CV_32F, y_kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

//    cout << "dx" << endl;
//    print(x_res);
//    cout << endl << "dy" << endl;
//    print(y_res);
//    cout << endl;

//    cv::magnitude(x_res, y_res, magnitude);
    magnitude = abs(x_res) + abs(y_res);
    cv::phase(x_res, y_res, phase, true);
    cv::subtract(phase, 360, phase, (phase > 180));
}

void NMS(cv::Mat mag, cv::Mat grad) {
    for (int i = 1; i < mag.size().height - 1; i++) {
        for (int j = 1; j < mag.size().width - 1; j++) {
            float val = mag.at<float>(i, j), dir = grad.at<float>(i, j);
//            cout << j << " " << i << " " << val << " " << dir << endl;
            if ((dir <= -67.5 || dir >= 67.5) && (val < mag.at<float>(i + 1, j) || val < mag.at<float>(i - 1, j))) {
                mag.at<float>(i, j) = 0;
//                cout << "Zeroed out 0" << endl;
            } else if ((dir > -67.5 && dir <= -22.5) &&
                       (val < mag.at<float>(i + 1, j + 1) || val < mag.at<float>(i - 1, j - 1))) {
                mag.at<float>(i, j) = 0;
//                cout << "Zeroed out 1" << endl;
            } else if (dir > -22.5 && dir <= 22.5 && (val < mag.at<float>(i, j + 1) || val < mag.at<float>(i, j - 1))) {
                mag.at<float>(i, j) = 0;
//                cout << "Zeroed out 2" << endl;
            } else if ((dir > 22.5 && dir < 67.5) &&
                       (val < mag.at<float>(i + 1, j - 1) || val < mag.at<float>(i - 1, j + 1))) {
                mag.at<float>(i, j) = 0;
//                cout << "Zeroed out 3" << endl;
            }
        }
    }
}

queue<pair<int, int>> DoubleThresholding(cv::Mat mat, double low = 0.15, double high = 0.6) {
    queue<pair<int, int>> q;
    for (int i = 0; i < mat.size().height; i++) {
        for (int j = 0; j < mat.size().width; j++) {
            float val = mat.at<float>(i, j);
            if (val > high) {
                mat.at<float>(i, j) = 255;
                q.push(make_pair(i, j));
            } else if (val > low)
                mat.at<float>(i, j) = 1;
            else
                mat.at<float>(i, j) = 0;
        }
    }

    return q;
}

void Hysteresis(cv::Mat mat, queue<pair<int, int>> q) {
    int directions[8][2] = {{-1, 1},
                            {-1, 0},
                            {-1, -1},
                            {0,  1},
                            {0,  -1},
                            {1,  1},
                            {1,  0},
                            {1,  -1}};
    while (!q.empty()) {
        pair<int, int> point = q.front();
        q.pop();
        for (auto dir: directions) {
            int nx = point.second + dir[1], ny = point.first + dir[0];
            if (0 <= ny && ny < mat.size().height && 0 <= nx && nx < mat.size().width &&
                mat.at<float>(ny, nx) == 1) {
                mat.at<float>(ny, nx) = 255;
                q.push(make_pair(ny, nx));
            }
        }
    }
}

int main() {
//    float test[5][5] = {{0.5, 0.8, 0.9, 1,   1},
//                        {0.2, 0.5, 0.8, 0.9, 1},
//                        {0.1, 0.2, 0.5, 0.8, 0.9},
//                        {0,   0.1, 0.2, 0.5, 0.8},
//                        {0,   0,   0.1, 0.2, 0.5}};
////    float test[5][5] = {{1,   1,   1,   1,   1},
////                        {0.9, 0.9, 0.9, 0.9, 0.9},
////                        {0.5, 0.5, 0.5, 0.5, 0.5},
////                        {0.1, 0.1, 0.1, 0.1, 0.1},
////                        {0,   0,   0,   0,   0}};
//    cv::Mat test_mat(5, 5, CV_32F, test), mag_test, grad_test, cv_sobel_out;
//
//    cv::flip(test_mat, test_mat, 0);
//
//    cout << "Mat" << endl;
//    print(test_mat);
//    cout << endl;
//
//    Sobel(test_mat, mag_test, grad_test);
//
//    cout << "Mag" << endl;
//    print(mag_test);
//    cout << endl << "Grad" << endl;
//    print(grad_test);
//    cout << endl;
//
////    cv::Sobel(test_mat, cv_sobel_out, -1, 1, 0);
////    cout << "dx" << endl;
////    print(cv_sobel_out);
////    cout << endl;
////    cv::Sobel(test_mat, cv_sobel_out, -1, 0, 1);
////    cout << "dy" << endl;
////    print(cv_sobel_out);
////    cout << endl;
//
//    cv::resize(test_mat, test_mat, cv::Size(900, 900), 0, 0, cv::INTER_NEAREST);
//
//    imshow("Test", test_mat);
//
//    NMS(mag_test, grad_test);
//
//    cout << "NMS mag" << endl;
//    print(mag_test);
//    cout << endl;
//
//    cv::waitKey(0);
//
//    return 0;

    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_PAIR);

    cout << "Binding to address" << endl;
    socket.bind("tcp://*:8001");

    cout << "Awaiting reply" << endl;
    zmq::message_t reply, request(7);
    zmq::recv_result_t recvResult = socket.recv(reply);

    cout << "Received reply" << endl;

    memcpy(request.data(), "Success", 7);

    socket.send(request, zmq::send_flags::none);

    cv::namedWindow("original");
    cv::namedWindow("sobel mag");
    cv::namedWindow("nms mag");
//    cv::namedWindow("dbl thresh");
    cv::namedWindow("final");
    cv::namedWindow("truth");

    deque<double> frame_times;
    double start = (double) chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;

    zmq::message_t request_message1(1);
    memcpy(request_message1.data(), "D", 1);
    socket.send(request_message1, zmq::send_flags::none);

    while (true) {
//        cout << "Transmitted request message" << endl;

        recvResult = socket.recv(reply);
        zmq::message_t request_message(1);
        memcpy(request_message.data(), "D", 1);
        socket.send(request_message, zmq::send_flags::none);
//        cout << "Received message data" << endl;

        char *buf = static_cast<char *>(reply.data());

        cv::Mat img = cv::imdecode(cv::Mat(1, reply.size(), CV_8UC1, buf),
                                   cv::IMREAD_UNCHANGED), grayscale, disp_mag;

        cv::imshow("original", img);

        cv::cvtColor(img, grayscale, cv::COLOR_BGR2GRAY);
        grayscale.convertTo(grayscale, CV_32F);
        grayscale /= 255.0;

        cv::Mat gaussian, mag, phase;
        Gaussian(grayscale, gaussian, 5, 1);

        Sobel(grayscale, mag, phase);

        cv::imshow("sobel mag", mag);

        NMS(mag, phase);

        cv::imshow("nms mag", mag);

        auto q = DoubleThresholding(mag);

//        cv::imshow("dbl thresh", mag);
        Hysteresis(mag, q);

        cv::threshold(mag, mag, 100, 1, cv::THRESH_BINARY);

        cv::imshow("final", mag);

        cv::Mat canny_truth;
        cv::cvtColor(img, grayscale, cv::COLOR_BGR2GRAY);
        cv::Canny(grayscale, canny_truth, 100, 200);
        cv::imshow("truth", canny_truth);

        frame_times.push_back((double) chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0);
        while (frame_times.front() < frame_times.back() - 5)
            frame_times.pop_front();

        printf("FPS: %f\n", (double) frame_times.size() / min(frame_times.back() - start, 5.0));

        if (cv::waitKey(10) == 27) {
            break;
        }
    }
}
