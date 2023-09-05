// std
#include <iostream>
#include <vector>
#include <string>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <random>


using namespace cv;
using namespace std;



////Functions for Core 3

void corePart3(Mat img1, Mat img2) {

    // Convert the images to grayscale
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    // Detect keypoints and compute descriptors using ORB
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    Ptr<ORB> sift = ORB::create();
    sift->detectAndCompute(gray1, noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(gray2, noArray(), keypoints2, descriptors2);

    // Match the descriptors using Brute-Force Matcher
    BFMatcher matcher(NORM_HAMMING);
    std::vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Find the homography matrix using RANSAC
    std::vector<Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }


    //add 100px padding to each image
    Mat padded1, padded2;
    copyMakeBorder(img1, padded1, 100, 100, 100, 100, BORDER_CONSTANT, Scalar(0, 0, 0));
    copyMakeBorder(img2, padded2, 100, 100, 100, 100, BORDER_CONSTANT, Scalar(0, 0, 0));

    //find homography
    Mat H = findHomography(points1, points2, RANSAC);


    Mat warped1;
    warpPerspective(padded1, warped1, H, padded1.size());

    //add the two images together
    Mat stitched = Mat::zeros(padded1.size(), CV_8UC3);
    warped1.copyTo(stitched);
    for (int y = 0; y < padded2.rows; y++)
    {
        for (int x = 0; x < padded2.cols; x++)
        {
            if (padded2.at<Vec3b>(y, x) != Vec3b(0, 0, 0))
            {
                stitched.at<Vec3b>(y, x) = padded2.at<Vec3b>(y, x);
            }
        }
    }



    //show the result for Core 3
    imshow("Core 3", stitched);
    waitKey(0);
    destroyAllWindows();
}

//// Completion Functions
vector<Mat> loadImages(int numFrames, const String prefix) {
    std::vector<Mat> images;
    for (int i = 0; i < numFrames; i++) {
        std::ostringstream frameNumber;
        frameNumber << std::setfill('0') << std::setw(3) << i;
        String filename = prefix + frameNumber.str() + ".jpg";
        Mat image = imread(filename);
        images.push_back(image);
    }

    std::cout << "...Image loading complete \n";
    return images;
}

void exportImages(const std::vector<Mat>& images, const String prefix, const String outputFolder) {
    for (int i = 0; i < images.size(); i++) {
        std::ostringstream frameNumber;
        frameNumber << std::setw(3) << std::setfill('0') << i;
        String filename = outputFolder + "/" + prefix + frameNumber.str() + ".png";
        imwrite(filename, images[i]);
    }
}


void generate1DGaussian(double mean, double stddev, int size, std::vector<double>& gaussian) {

    // Create the 1D Gaussian kernel
    Mat kernel = getGaussianKernel(size, stddev, CV_64F);
    gaussian.resize(size);
    for (int i = 0; i < size; ++i)
    {
        gaussian[i] = kernel.at<double>(i);
    }
}

Mat extraction(Mat img_1, Mat img_2) {

    // Detect keypoints and compute descriptors using ORB
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    Ptr<ORB> sift = ORB::create();
    sift->detectAndCompute(img_1, noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img_2, noArray(), keypoints2, descriptors2);

    // Match the descriptors using Brute-Force Matcher
    BFMatcher matcher(NORM_HAMMING);
    std::vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Find the homography matrix using RANSAC
    std::vector<Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    Mat H = findHomography(points1, points2, RANSAC);
    //std::cout << "...Extraction complete\n";
    return H;

}

void VideoStabilzing(std::vector<Mat>& frames) {

    // Generate vector of matrix differences between frames (movement performed between each frame)
    std::vector<Mat> differences(frames.size() - 1);
    for (int i = 0; i < frames.size() - 1; ++i) {
        differences[i] = frames[i + 1] - frames[i];
    }


    // Generate vector of cumulative matrices
    std::vector<Mat> cumulativeMatrices(frames.size());
    cumulativeMatrices[0] = Mat::eye(3, 3, CV_64F);

    for (int i = 1; i < frames.size(); ++i) {
        // get the frames
        Mat initialFrame = frames[i];
        Mat prevFrame = frames[i - 1];

        // extract the h matrix from the frames
        Mat h = extraction(initialFrame, prevFrame);

        // Update to store the transformation matrix
        cumulativeMatrices[i] = cumulativeMatrices[i - 1] * h; // TODO needs fixing
        //std::cout<<"->vs - computing analysis "<<i<<"\n";
    }


    // Generate 1D Gaussian filter
    double mean = 0.0;
    double stddev = 5;
    int filterSize = 9;  // Window size for the Gaussian filter
    std::vector<double> gaussian;
    generate1DGaussian(mean, stddev, filterSize, gaussian);


    // Create vector of cumulative matrices with Gaussian smoothing
    std::vector<Mat> smoothedMatrices(frames.size());
    for (int i = 0; i < frames.size(); ++i) {
        smoothedMatrices[i] = Mat::zeros(3, 3, CV_64F); //g transformation matrix
        for (int j = -filterSize / 2; j <= filterSize / 2; ++j) {
            int index = i + j;
            if (index >= 0 && index < frames.size()) {
                double weight = gaussian[j + filterSize / 2];
                smoothedMatrices[i] += weight * cumulativeMatrices[index];
            }
        }
    }

    // Create vector of stabilization matrices by getting movement and subtracting smooth
    std::vector<Mat> stabilizationMatrices(frames.size());
    for (int i = 0; i < frames.size(); ++i) {
        stabilizationMatrices[i] = smoothedMatrices[i].inv() * cumulativeMatrices[i];
    }


    // Apply stabilization transforms to each frame
    for (int i = 0; i < frames.size(); ++i) {
        Mat stabilizedFrame;
        warpPerspective(frames[i], stabilizedFrame, stabilizationMatrices[i], frames[i].size());
        frames[i] = stabilizedFrame;
    }

}

// main program


int main(int argc, char** argv) {



    // check we have exactly one additional argument
    // eg. res/vgc-logo.png
    if (argc != 2) {
        cerr << "Usage: Cgra352 <Image>" << endl;
        abort();
    }

    //Read the file
    Mat img1 = cv::imread("/home/chawangray/Downloads/CGRA352_A4/A4/work/res/Frame039.jpg", cv::IMREAD_COLOR);
    imshow("img1", img1);

    Mat img2 = cv::imread("/home/chawangray/Downloads/CGRA352_A4/A4/work/res/Frame041.jpg", cv::IMREAD_COLOR);
    imshow("img2", img2);


    //core 1 , Detect the keypoints using sift, then compute the descriptors
    Ptr<SIFT> sift = sift->create();
    vector<KeyPoint> kpoints_1, kpoints_2;
    Mat descriptors_1;
    Mat descriptors_2;
    sift->detectAndCompute(img1, noArray(), kpoints_1, descriptors_1);
    sift->detectAndCompute(img2, noArray(), kpoints_2, descriptors_2);

    BFMatcher matcher(NORM_L2, true);
    vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);

    Mat core1_matched;
    vconcat(img1, img2, core1_matched);

    cout << "matches: " << matches.size() << "\n";
    for (int i = 0; i < matches.size(); i++) {
        Point2f p1 = kpoints_1[matches[i].queryIdx].pt;
        Point2f p2 = kpoints_2[matches[i].trainIdx].pt + Point2f(0, img1.rows);
        line(core1_matched, p1, p2, Scalar(0, 255, 0), 1);
    }
    //Core 1 Output
    imshow("Core 1", core1_matched);


    //Core 2
    std::random_device rd;
    std::mt19937 rng(rd());
    std::vector<DMatch> edges = matches;

    std::vector<DMatch> bestInliers;
    Mat bestHomography;
    srand(time(NULL));
    for (int i = 0; i < 100; i++) {
        std::cout << i << std::endl;

        // create the inlier list
        std::vector<DMatch> inlierEdges;
        std::vector<DMatch> outlierEdges;

        // select 4 random pairs
        int rand1 = rand() % edges.size();
        int rand2 = rand() % edges.size();
        int rand3 = rand() % edges.size();
        int rand4 = rand() % edges.size();


        while (rand2 == rand1 || rand2 == rand3 || rand2 == rand4) {

            rand2 = rand() % edges.size();
        }
        while (rand3 == rand1 || rand3 == rand2 || rand3 == rand4) {

            rand3 = rand() % edges.size();
        }
        while (rand4 == rand1 || rand4 == rand2 || rand4 == rand3) {

            rand4 = rand() % edges.size();
        }


        std::vector<Point2f> src;
        src.push_back(kpoints_1[edges[rand1].queryIdx].pt);
        src.push_back(kpoints_1[edges[rand2].queryIdx].pt);
        src.push_back(kpoints_1[edges[rand3].queryIdx].pt);
        src.push_back(kpoints_1[edges[rand4].queryIdx].pt);
        std::vector<Point2f> dst;
        dst.push_back(kpoints_2[edges[rand1].trainIdx].pt);
        dst.push_back(kpoints_2[edges[rand2].trainIdx].pt);
        dst.push_back(kpoints_2[edges[rand3].trainIdx].pt);
        dst.push_back(kpoints_2[edges[rand4].trainIdx].pt);



        //compute homography for pairs exactly
        Mat h = findHomography(src, dst, 0);
        cv::Mat_<float> hMat = cv::Mat::eye(3, 3, CV_32FC1);

        // epsilon
        int epsilon = 10;

        //compute inliers and outliers
        for (DMatch e : edges) {
            if (!h.empty()) {
                hMat = h;
                Vec3f point(kpoints_1[e.queryIdx].pt.x, kpoints_1[e.queryIdx].pt.y, 1);
                Mat p(point);
                Mat q = hMat * p;
                Vec3f qVec = q;
                Point2f pointQ = Point(qVec[0], qVec[1]);
                float error = norm((Mat)kpoints_2[e.trainIdx].pt, (Mat)pointQ);


                // check if error is smaller than the epsilon
                if (error < epsilon) {
                    inlierEdges.push_back(e);
                }
            }
        }
        std::cout << "--> inter-computing analysis 4\n";
        std::cout << "inlier edge side " << inlierEdges.size() << "  best inlier size " << bestInliers.size()
            << std::endl;

        if (inlierEdges.size() > bestInliers.size()) {
            bestInliers = inlierEdges;
            bestHomography = h.clone();
        }
    }
    std::cout << "    ...finished computing \n";


    std::vector<DMatch> finalInliers;
    std::vector<DMatch> finalOutliers;

    for (DMatch e : edges) {
        Mat_<float> homography = Mat::eye(3, 3, CV_32FC1);
        homography = bestHomography;
        Vec3f point(kpoints_1[e.queryIdx].pt.x, kpoints_1[e.queryIdx].pt.y, 1);
        Mat p(point);
        Mat q = homography * p;
        Vec3f qVec = q;
        Point2f pointQ = Point(qVec[0], qVec[1]);
        float error = norm((Mat)kpoints_2[e.trainIdx].pt, (Mat)pointQ);
        int epsilon = 10;

        if (error < epsilon) {
            finalInliers.push_back(e);
        }
        else {
            finalOutliers.push_back(e);
        }
    }

    std::cout << "inlier size " << finalInliers.size() << "  outlier size " << finalOutliers.size() << std::endl;
    Mat result;
    vconcat(img1, img2, result);

    //draw inliners in green
    for (DMatch in : finalInliers) {
        line(result, Point(kpoints_1[in.queryIdx].pt), Point(kpoints_2[in.trainIdx].pt) + Point(0, img1.rows),
            Scalar(0, 255, 0), 1);
    }
    //draw outliers in red
    for (DMatch out : finalOutliers) {
        line(result, Point(kpoints_1[out.queryIdx].pt), Point(kpoints_2[out.trainIdx].pt) + Point(0, img1.rows),
            Scalar(0, 0, 255), 1);
    }

    imshow("Core 2", result);

    //Core 3
    corePart3(img1, img2);


    //Completion

    // Load images
    vector<Mat>  loadImgs = loadImages(102,"/home/chawangray/Downloads/CGRA352_A4/A4/work/res/Frame");

    VideoStabilzing(loadImgs);
    exportImages(loadImgs, "Stable", "/home/chawangray/Downloads/CGRA352_A4/A4/work/Completion");
}

