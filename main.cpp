#include <iostream>
#include <Eigen/Dense>
#include <zmq.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>

using Eigen::MatrixXd;
using namespace std;

cv::Mat Gaussian(const cv::Mat &mat, int kernel_size = 5, int sigma = 1.0) {
    assert(kernel_size % 2 == 1 && kernel_size > 0);
    int k = (kernel_size - 1) / 2;

    double kernel[kernel_size][kernel_size];
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j <= i; j++) {
            double val =
                    1.0 / (2 * M_PI * sigma * sigma) * exp(-((i - k) * (i - k) + (j - k) * (j - k))) / 2 / sigma /
                    sigma;
            kernel[i][j] = val;
            kernel[j][i] = val;
        }
    }

    cv::Mat kernel_mat(kernel_size, kernel_size, CV_64F, kernel), result;
    cv::filter2D(mat, result, CV_32F, kernel_mat, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    return result;
}

pair<cv::Mat, cv::Mat> Sobel(const cv::Mat &mat) {
    float x_kernel_data[3][3] = {{-1, 0, 1},
                                 {-2, 0, 2},
                                 {-1, 0, 1}};
    float y_kernel_data[3][3] = {{1,  2,  1},
                                 {0,  0,  0},
                                 {-1, -2, -1}};
    cv::Mat x_res, y_res, x_kernel(3, 3, CV_32F, x_kernel_data), y_kernel(3, 3, CV_32F, y_kernel_data);

    cv::filter2D(mat, x_res, CV_32F, x_kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(mat, y_res, CV_32F, y_kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

//    print(x_res);
//    cout << endl;
//    print(y_res);
//    cout << endl;

    cv::Mat mag, grad;
    cv::magnitude(x_res, y_res, mag);
    cv::phase(x_res, y_res, grad, true);
    cv::subtract(grad, 360, grad, (grad > 180));

    return make_pair(mag, grad);
}

void NMS(cv::Mat mag, cv::Mat grad) {
    for (int i = 1; i < mag.size().height - 1; i++) {
        for (int j = 1; j < mag.size().width - 1; j++) {
            float val = mag.at<float>(i, j), dir = grad.at<float>(i, j);
            if ((dir < -67.5 || dir > 67.5) && (val <= mag.at<float>(i + 1, j) || val <= mag.at<float>(i - 1, j)))
                mag.at<float>(i, j) = 0;
            else if (dir < -22.5 && (val <= mag.at<float>(i + 1, j + 1) || val <= mag.at<float>(i - 1, j - 1)))
                mag.at<float>(i, j) = 0;
            else if (dir < 22.5 && (val <= mag.at<float>(i, j + 1) || val <= mag.at<float>(i, j - 1)))
                mag.at<float>(i, j) = 0;
            else if (val <= mag.at<float>(i + 1, j - 1) || val <= mag.at<float>(i - 1, j + 1))
                mag.at<float>(i, j) = 0;
        }
    }
}

queue<pair<int, int>> DoubleThresholding(cv::Mat mat) {
    queue<pair<int, int>> q;
    for (int i = 0; i < mat.size().height; i++) {
        for (int j = 0; j < mat.size().width; j++) {
            float val = mat.at<float>(i, j);
            if(val > 100) {
                mat.at<float>(i, j) = 255;
                q.push(make_pair(i, j));
            }
            else if(val > 30)
                mat.at<float>(i, j) = 1;
            else
                mat.at<float>(i, j) = 0;
        }
    }

    return q;
}

void Hysteresis(cv::Mat mat, queue<pair<int, int>> q) {
    int directions[8][2] = {{-1, 1}, {-1, 0}, {-1, -1}, {0, 1}, {0, -1}, {1, 1}, {1, 0}, {1, -1}};
    while(!q.empty()) {
        pair<int, int> point = q.front();
        q.pop();
        for(auto dir : directions) {
            int nx = point.second + dir[1], ny = point.first + dir[0];
            if(0 <= ny < mat.size().height && 0 <= nx < mat.size().width && mat.at<float>(ny, nx) == 1) {
                mat.at<float>(ny, nx) = 255;
                q.push(make_pair(ny, nx));
            }
        }
    }
}

int main() {
//    int test[3][3] = {{0, 0, 255},
//                      {0, 1, 0},
//                      {0, 0, 0}};
//    cv::Mat test_mat(3, 3, CV_8U, test);
//
//    print(test_mat);
//    cout << endl;
//
//    auto[mag_test, grad_test] = Sobel(test_mat);
//
//    print(mag_test);
//    cout << endl;
//    print(grad_test);
//    cout << endl;
//
//    cv::resize(test_mat, test_mat, cv::Size(900, 900));
//
//    imshow("Test", test_mat);
//
//    cv::waitKey(0);

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
    cv::namedWindow("final");

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

        cv::Mat img = cv::imdecode(cv::Mat(1, reply.size(), CV_8UC1, buf), cv::IMREAD_UNCHANGED), grayscale, disp_mag;

        cv::imshow("original", img);

        cv::cvtColor(img, grayscale, cv::COLOR_BGR2GRAY);

//        cv::Mat gaussian = Gaussian(grayscale);
        cv::Mat gaussian;
        cv::GaussianBlur(grayscale, gaussian, cv::Size(5, 5), 1);

        auto[mag, grad] = Sobel(gaussian);

        cv::normalize(mag, disp_mag, 0, 1, cv::NORM_MINMAX);

        cv::imshow("sobel mag", disp_mag);

        NMS(mag, grad);

        cv::normalize(mag, disp_mag, 0, 1, cv::NORM_MINMAX);

        cv::imshow("nms mag", disp_mag);

        auto q = DoubleThresholding(mag);
        Hysteresis(mag, q);

        cv::threshold(mag, mag, 100, 1, cv::THRESH_BINARY);

        cv::imshow("final", mag);

        frame_times.push_back((double) chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0);
        while (frame_times.front() < frame_times.back() - 5)
            frame_times.pop_front();

        printf("FPS: %f\n", (double) frame_times.size() / min(frame_times.back() - start, 5.0));

        if (cv::waitKey(10) == 27) {
            break;
        }
    }

//    MatrixXd m(2, 2);
//    m(0, 0) = 3;
//    m(1, 0) = 2.5;
//    m(0, 1) = -1;
//    m(1, 1) = m(1, 0) + m(0, 1);
//    std::cout << m << std::endl;
}
