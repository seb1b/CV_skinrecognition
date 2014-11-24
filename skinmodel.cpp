#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

cv::Mat1b skin_model;

cv::Mat hist_skin;
cv::Mat hist_no_skin;
cv::Mat decision_rule;
int hist_counter;

/// Constructor
SkinModel::SkinModel()
{
}


/// Destructor
SkinModel::~SkinModel()
{
}

/// Start the training.  This resets/initializes the model.
///
/// Implementation hint:
/// Use this function to initialize/clear data structures used for training the skin model.
void SkinModel::startTraining()
{
    skin_model = cv::Mat1b::zeros(0, 0);
    hist_counter = 0;
    //hist_skin;
    //hist_noskin;
    
    cout<<skin_model<<endl;
    //--- IMPLEMENT THIS ---//
}

/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
void SkinModel::train(const cv::Mat3b& img, const cv::Mat1b& mask)
{
    
    //get HSV
    Mat hsv;
    cvtColor(img, hsv, CV_BGR2HSV);
    //get Normalized RGB (aka rgb)
    
    
    //imshow("no ssdfsfkin", hsv);
    /// Using 50 bins for hue and 60 for saturation
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    
    const float* ranges[] = { h_ranges, s_ranges };
    
    // Use the o-th and 1-st channels
    int channels[] = { 0, 1 };
    
    /// Calculate the histograms for the HSV images
    calcHist( &hsv, 1, channels, mask, hist_skin, 2, histSize, ranges, true, true );
    
    calcHist( &hsv, 1, channels, ~mask, hist_no_skin, 2, histSize, ranges, true, true );
    
    
    //imshow("asd", temp_hist_skin);
    //imshow("no skin", temp_hist_no_skin);
    
    //cout << temp_hist_skin <<endl;
    //waitKey(0);
    
    // normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );
    
    //--- IMPLEMENT THIS ---//
    
    //train with mixture of gaussians
    hist_counter++;
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
///
/// Implementation hint:
/// e.g normalize w.r.t. the number of training images etc.
void SkinModel::finishTraining()
{
    
    //cout << hist_skin.size() << endl;
    //cout << hist_skin << endl;
    
    
    //normalize histogramm
    normalize(hist_skin,  hist_skin, 1, 1, NORM_L1, -1, Mat());
    normalize(hist_no_skin,  hist_no_skin, 1, 1, NORM_L1, -1, Mat());
    
    decision_rule = (hist_skin > hist_no_skin);
    cout << "Normalized: " << hist_skin.size() << endl;
    //cout << hist_skin << endl;
    
    
    //cout << temp_hist_skin <<endl;
    //waitKey(0);
    
    //cvNormalizeHist(hist_skin,1);
    
    /* for (int ubin=0; ubin < hist_bins[0]; ubin++) {
     for (int vbin = 0; vbin < hist_bins[1]; vbin++) {
     if (skin_hist.at<float>(ubin,vbin) > 0) {
     skin_hist.at<float>(ubin,vbin) /= skin_pixels;
     }
     if (non_skin_hist.at<float>(ubin,vbin) > 0) {
     non_skin_hist.at<float>(ubin,vbin) /= non_skin_pixels;
     }
     }
     */
    //--- IMPLEMENT THIS ---//
}


/// Classify an unknown test image.  The result is a probability
/// mask denoting for each pixel how likely it is of skin color.
///
/// @param img: unknown test image
/// @return:    probability mask of skin color likelihood
cv::Mat1b SkinModel::classify(const cv::Mat3b& img)
{
    cv::Mat1b skin = cv::Mat1b::zeros(img.rows, img.cols);
    //get HSV
    Mat hsv;
    cvtColor(img, hsv, CV_BGR2HSV);
    //--- IMPLEMENT THIS ---//
    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            //Martina
            if (true) {
                cv::Vec3b pixel = hsv.at<cv::Vec3b>(row, col);
                float h_bin_size = 180.0/50.0;
                float s_bin_size = 256.0/60.0;
                int h = pixel[0]/h_bin_size;
                int s = pixel[1]/s_bin_size;
                if (decision_rule.at<int>(s, h) > 1) {
                    skin(row, col) = 255;
                }
            }
            if (false)
                skin(row, col) = rand()%256;
            
            if (false)
                skin(row, col) = img(row,col)[2];
            
            if (false) {
                
                cv::Vec3b bgr = img(row, col);
                if (bgr[2] > bgr[1] && bgr[1] > bgr[0])
                    skin(row, col) = 2*(bgr[2] - bgr[0]);
            }
        }
    }
    
    //by seb
    //modify in order to change impact of opening closing
    cv::Mat element = getStructuringElement( 1, cv::Size( 13, 13 ), cv::Point( 6, 6 ) );
    
    //erosion and dilation to reduce mistakes
    //first morphological closing (delatation -> erosion) and then opening (erosion -> delatation)
    //MORPH_CLOSE 3     MORPH_OPEN 2
    //cv::morphologyEx( skin, skin, MORPH_CLOSE, element);
    //cv::morphologyEx( skin, skin, MORPH_OPEN, element);
    
    
    
    return skin;
}