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
double prior_skin;
double prior_no_skin;
int hist_counter;




cv::Mat skin_Covar, skin_Mu;
Mat skin_gauss;
Mat no_skin_gauss;

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
    
    int blur_size = 3;
    //get HSV
    Mat hsv;
    cvtColor(img, hsv, CV_BGR2HSV);
    
    //smooth image
    GaussianBlur(hsv, hsv, Size(blur_size, blur_size), 0, 0);
    
    //Seb with gauss
    for (int row = 0; row < hsv.rows; ++row) {
        for (int col = 0; col < hsv.cols; ++col) {
            
            cv::Vec3b hsv_ = hsv.at<cv::Vec3b>(row, col);

            
            double h__ = (double)hsv_[0];
            double s__ = (double)hsv_[1];
           
            double temp [] = {h__, s__};
            Mat hs_row = Mat(1,2,  CV_64F,temp);
            
            if(mask.at<int>(row, col) >0){
                
                skin_gauss.push_back(hs_row);
            
            }else{
                no_skin_gauss.push_back(hs_row);
            }
            
            
            
        }
    
    }
    /*
    vector<Mat> hsv_planes;
    split( hsv, hsv_planes );
    hsv_planes[0].mul(mask/255);
    */
    
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
    
    hist_counter++;
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
///
/// Implementation hint:
/// e.g normalize w.r.t. the number of training images etc.
void SkinModel::finishTraining()
{
    
    //calcualte cov and mean of skin
    cv::calcCovarMatrix(skin_gauss,skin_Covar, skin_Mu, CV_COVAR_NORMAL | CV_COVAR_ROWS |CV_COVAR_SCALE );

    
    
    //calcualte cov and mean of no _skin
    cv::Mat Covar_, Mu_;
    cv::calcCovarMatrix(no_skin_gauss,Covar_, Mu_, CV_COVAR_NORMAL | CV_COVAR_ROWS |CV_COVAR_SCALE );
   // waitKey(0);

    

    double skin_hist_sum = cv::sum(hist_skin)[0];
    double no_skin_hist_sum = cv::sum(hist_no_skin)[0];
    prior_skin = skin_hist_sum / (skin_hist_sum + no_skin_hist_sum);
    prior_no_skin = no_skin_hist_sum / (skin_hist_sum + no_skin_hist_sum);
    
    //normalize histogramm
    normalize(hist_skin,  hist_skin, 1, 1, NORM_L1, -1, Mat());
    normalize(hist_no_skin,  hist_no_skin, 1, 1, NORM_L1, -1, Mat());
    
    decision_rule = (hist_skin > hist_no_skin);

 
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
            if (false) {
                cv::Vec3b pixel = hsv.at<cv::Vec3b>(row, col);
                float h_bin_size = 180.0/50.0;
                float s_bin_size = 256.0/60.0;
                int h = ((int)pixel[0])/h_bin_size;
                int s = ((int)pixel[1])/s_bin_size;
               // cout << "prior no skin"<< << endl;
                if (((hist_skin.at<int>(s, h)*prior_skin) / prior_no_skin) > ((hist_no_skin.at<int>(s, h)*prior_no_skin) / prior_skin)) {
                    skin(row, col) = 255;
                }
            }
            
            //Seb gauss
            if (true) {
                //for skin use formular slide 31
                cv::Vec3b pixel = hsv.at<cv::Vec3b>(row, col);
                double h = pixel[0];
                double s = pixel[1];
                double temp [] = {h, s};
                Mat x = Mat(1,2,  CV_64F,temp);
     
               
                Mat x_mu = x - skin_Mu;
                Mat x_mu_t = x_mu.t();
                
                
                double det_cov = determinant(skin_Covar);
                Mat cov_inv = skin_Covar.inv();
                
                double first_ = 0.5 * sqrt(det_cov);
                Mat exp_ = -0.5 * x_mu * cov_inv * x_mu_t;
                double exponent = exp_.at<double>(0,0);

                
                
                double probab = first_ * exp(exponent);
   
               // waitKey(0);
               
                if (probab > 0) {
                    skin(row, col) = 1;
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
    cv::morphologyEx( skin, skin, MORPH_CLOSE, element);
    cv::morphologyEx( skin, skin, MORPH_OPEN, element);
    
    
    
    return skin;
}