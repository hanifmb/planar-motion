#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include<iostream>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <random>

using PointPairs3f = std::vector<std::pair<cv::Point3f, cv::Point3f>>;

void calculate2PtEssential(std::vector<cv::Mat>& Es, const std::pair<cv::Point3f, cv::Point3f>& c1, 
                            const std::pair<cv::Point3f, cv::Point3f>& c2);

void calcEpilines(const std::vector<cv::Point2f>& points, cv::Mat F, std::vector<cv::Vec3f>& lines);

void findMatchesSIFT(const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::KeyPoint>& keypoints1, 
                    std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& good_matches, const float& thresh);

void drawEpipolarLines(const cv::Mat& img1, const cv::Mat& img2, const std::vector<cv::Point2f>& points1, 
                    const std::vector<cv::Point2f>& points2, const cv::Mat& F);

void draw_cross(const cv::Point& pt, const cv::Mat& image, const int& size);

float calcDistance(cv::Point2f point, float a, float b, float c);


int main(int argc, char** argv)
{
	if (argc != 5)
	{
		std::cout << "Usage: ./main img1 img2\n";
		return -1;
	}
    
    //--loading images
    cv::Mat img1 = cv::imread(argv[1]);
    cv::Mat img2 = cv::imread(argv[2]);
    
    std::string err = "Could not open or find image\n"; 
	if (!img1.data || !img2.data) {std::cout << err; return -1;}

    std::vector<cv::DMatch> good_matches;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    float ratio_thresh = atof(argv[3]);
    findMatchesSIFT(img1, img2, keypoints1, keypoints2, good_matches, ratio_thresh);
    
    std::vector<cv::Point2f> points1, points2;
    
    //--intrinsic matrix
    double K1_data[9] = {1280.7, 0.0, 969.4257, 0.0, 1281.2, 639.7227, 0.0, 0.0, 1.0};
    cv::Mat K1(3, 3, CV_64F, K1_data); 

    double K2_data[9] = {1276.1 , 0.0, 965.9650, 0.0, 1275.4, 618.2222, 0.0, 0.0, 1.0};
    cv::Mat K2(3, 3, CV_64F, K2_data); 
    
    std::string cam_param = argv[4];
    cv::Mat K = K1;

    if(cam_param == "dev1") K = K1;
    else if(cam_param == "dev2") K = K2;
    
    PointPairs3f pp;
    for(int i = 0; i<good_matches.size(); ++i){
        cv::Point2f pt1(keypoints1[good_matches[i].queryIdx].pt.x, keypoints1[good_matches[i].queryIdx].pt.y);
        cv::Point2f pt2(keypoints2[good_matches[i].trainIdx].pt.x, keypoints2[good_matches[i].trainIdx].pt.y);

        cv::Mat pts_query = (cv::Mat_<double>(3,1) << pt1.x, pt1.y, 1.0f);
        cv::Mat pts_train = (cv::Mat_<double>(3,1) << pt2.x, pt2.y, 1.0f);
        
        //--converting image points to camera points
        cv::Mat mt_3d_query = K.inv() * pts_query;  
        cv::Mat mt_3d_train = K.inv() * pts_train;  

        cv::Point3f pt_3d_query(mt_3d_query.at<double>(0, 0), mt_3d_query.at<double>(1, 0), mt_3d_query.at<double>(2, 0));
        cv::Point3f pt_3d_train(mt_3d_train.at<double>(0, 0), mt_3d_train.at<double>(1, 0), mt_3d_train.at<double>(2, 0));
        
        pp.push_back(std::pair(pt_3d_query, pt_3d_train));

        points1.emplace_back(pt1);
        points2.emplace_back(pt2);
    }

    //--acquiring two random correspondences
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, pp.size()-1);

    cv::Mat best_F;
    cv::Mat best_E;
    int most_num_inliers = 0;
    
    /* RANSAC iteration */
    int t = 2000;
    while(t--){
        /* pick two random correspondences */
        int index1 = dis(gen);
        int index2 = dis(gen);
        while(index1 == index2){
            index2 = dis(gen);
        }

        float x1_l = pp[index1].first.x;
        float y1_l = pp[index1].first.y;
        float z1_l = pp[index1].first.z;

        float x1_r = pp[index1].second.x;
        float y1_r = pp[index1].second.y;
        float z1_r = pp[index1].second.z; 

        float x2_l = pp[index2].first.x;
        float y2_l = pp[index2].first.y;
        float z2_l = pp[index2].first.z;

        float x2_r = pp[index2].second.x;
        float y2_r = pp[index2].second.y;
        float z2_r = pp[index2].second.z;

        std::pair<cv::Point3f, cv::Point3f> c1(cv::Point3f(x1_l, y1_l, z1_l), cv::Point3f(x1_r, y1_r, z1_r));
        std::pair<cv::Point3f, cv::Point3f> c2(cv::Point3f(x2_l, y2_l, z2_l), cv::Point3f(x2_r, y2_r, z2_r));

        std::vector<cv::Mat> Es;
        calculate2PtEssential(Es, c1, c2);

        for(auto & E: Es){
            E.convertTo(E, CV_64F);
            
            cv::Mat F = K.t().inv() * E * K.inv();
            F = (1/F.at<double>(2, 2)) * F;

            std::vector<cv::Vec3f> lines1;
            calcEpilines(points1, F, lines1);

            std::vector<cv::Vec3f> lines2;
            calcEpilines(points2, F, lines2);

            float error = 0;
            int curr_num_inliers = 0;
            for(int i = 0; i<lines1.size(); i++){

                float dist1 = calcDistance(points1[i], lines2[i][0], lines2[i][1], lines2[i][2]);
                float dist2 = calcDistance(points2[i], lines1[i][0], lines1[i][1], lines1[i][2]);
                 
                error = (dist1*dist1 + dist2*dist2);
                if(error < 200) curr_num_inliers++;
            }
            
            if(curr_num_inliers > most_num_inliers){
                most_num_inliers = curr_num_inliers; 
                best_E = E;
                best_F = F;
            }

        }

    }

    std::cout << "Most num of inliers\n" << most_num_inliers << "\n\n";
    std::cout << "Num of points\n" << pp.size() << "\n\n";
    std::cout << "Best E\n" << best_E << "\n\n";
    std::cout << "Best F\n" << best_F << "\n\n";

    
    drawEpipolarLines(img1, img2, points1, points2, best_F);
    
	cv::namedWindow("img2", cv::WINDOW_NORMAL); 
    cv::imshow("img2", img2); 

	cv::namedWindow("img1", cv::WINDOW_NORMAL); 
    cv::imshow("img1", img1); 

    imwrite("image1.jpg", img1); 
    imwrite("image2.jpg", img2);

    cv::waitKey(0);
}


void findMatchesSIFT(const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& good_matches, const float& thresh){
    int minHessian = 400;
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }    
}

void calculate2PtEssential(std::vector<cv::Mat>& Es, const std::pair<cv::Point3f, cv::Point3f>& c1, const std::pair<cv::Point3f, cv::Point3f>& c2){

    float x1_l = c1.first.x;
    float y1_l = c1.first.y;
    float z1_l = c1.first.z;

    float x1_r = c1.second.x;
    float y1_r = c1.second.y;
    float z1_r = c1.second.z; 

    float x2_l = c2.first.x;
    float y2_l = c2.first.y;
    float z2_l = c2.first.z;

    float x2_r = c2.second.x;
    float y2_r = c2.second.y;
    float z2_r = c2.second.z;

    /* from the epipolar constrain A1v1 = A2v2 */
    float A1_data[4] = {x1_l * y1_r,  -z1_l * y1_r, 
                       x2_l * y2_r,  -z2_l * y2_r};
    float A2_data[4] = {x1_r * y1_l,  z1_r * y1_l, 
                       x2_r * y2_l,  z2_r * y2_l};

    /* unit circle and ellipse are v1.t() * v = 1 */
    /* and v1.t()*C.t()*C*v1 = 1 */
    
    cv::Mat A1(2, 2, CV_32FC1, A1_data); 
    cv::Mat A2(2, 2, CV_32FC1, A2_data); 

    cv::Mat C = A2.inv() * A1;

    cv::Mat S, U, Vt;

    cv::Mat B = C.t() * C;
    cv::SVDecomp(B, S, U, Vt, cv::SVD::FULL_UV);
    
    float s1 = S.at<float>(0, 0);
    float s2 = S.at<float>(1, 0);
    
    /* ellipse represented by r = U.t() * v1 */
    /* v1 = U*r */
    
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            float rx = pow(-1, i)*sqrt((1 - s2)/(s1 - s2));
            float ry = pow(-1, j)*sqrt((s1 - 1)/(s1 - s2));

            float r_data[2] = {rx, ry};
            cv::Mat r(2, 1, CV_32FC1, r_data); 
            
            cv::Mat v2 = U*r;
            cv::Mat v1 = C*v2;

            /* E = */ 
            /* [0              ,cos(alpha-beta)    ,0          ] */
            /* [-cos(alpha)    ,0                  ,sin(alpha) ] */
            /* [0              ,sin(alpha-beta)    ,0          ] */

            float E_data[9] = {0, v1.at<float>(0, 0), 0,
                            -v2.at<float>(0, 0), 0, v2.at<float>(1, 0),
                            0, v1.at<float>(1, 0), 0};
            cv::Mat E(3, 3, CV_32FC1, E_data); 
            Es.emplace_back(E);
        }
    }
}

void drawEpipolarLines(const cv::Mat& img1, const cv::Mat& img2, 
                        const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2, 
                        const cv::Mat& F){

    std::vector<cv::Vec3f> lines1;
    calcEpilines(points1, F, lines1);

    std::vector<cv::Vec3f> lines2;
    calcEpilines(points2, F, lines2);


    for(int i = 0; i<lines1.size(); i++){

        cv::line(img2, cv::Point(0, -lines1[i][2] / lines1[i][1]),
                 cv::Point(img2.cols, -(lines1[i][2] + lines1[i][0] * img2.cols) / lines1[i][1]),
                 cv::Scalar(255, 255, 255));
        draw_cross(points2[i], img2, 10);

        cv::line(img1, cv::Point(0, -lines2[i][2] / lines2[i][1]),
                 cv::Point(img1.cols, -(lines2[i][2] + lines2[i][0] * img1.cols) / lines2[i][1]),
                 cv::Scalar(255, 255, 255));
        draw_cross(points1[i], img1, 10);

    }
}

void draw_cross(const cv::Point& pt, const cv::Mat& image, const int& size){

    cv::Point starting1(pt.x - size, pt.y-size);
    cv::Point ending1(pt.x + size, pt.y+size);

    cv::Point starting2(pt.x + size, pt.y-size);
    cv::Point ending2(pt.x-size, pt.y+size);

    cv::Scalar line_Color(0, 255, 0);
    int thickness = 2;
       
    cv::line(image, starting1, ending1, line_Color, thickness);
    cv::line(image, starting2, ending2, line_Color, thickness);
}

float calcDistance(cv::Point2f point, float a, float b, float c)
{
    return abs(a * point.x + b * point.y + c) / sqrt(a * a + b * b);
}

void calcEpilines(const std::vector<cv::Point2f>& points, cv::Mat F, std::vector<cv::Vec3f>& lines){
    for(auto & e: points){
        cv::Mat my_point = (cv::Mat_<float>(3,1) 
                << e.x, e.y, 1.0f);

        /* cv::Mat temp_F = F; */
        F.convertTo(F, CV_32F);

        cv::Mat line = F * my_point;
        float k = 1/sqrt(pow(line.at<float>(0, 0), 2) + 
                    pow(line.at<float>(1, 0), 2));
        line = line * k;

        cv::Vec3f temp_line(line.reshape(3).at<cv::Vec3f>());
        lines.emplace_back(temp_line);
    }
}
