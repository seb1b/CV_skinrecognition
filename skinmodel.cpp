#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include <math.h>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;


//for hist
cv::Mat hist_skin;
cv::Mat hist_no_skin;



//for gauss
Mat skin_gauss;
Mat no_skin_gauss;
Mat gauss_values_skin;
Mat gauss_values_no_skin;




//parameters
int morph_elem = 0;
int morph_size = 3;
int blur_size =9;
int counter = 0;

//method selection
bool gauss_on = false;
bool hist_on = true;
bool hist_on_there = false;
bool normalize_on =false;


int bin_h_size = 90;
int bin_s_size = 110;

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
    gauss_values_skin = Mat(180, 255,  CV_64F);
    gauss_values_no_skin = Mat(180, 255,  CV_64F);
    hist_skin = Mat(bin_h_size,bin_s_size, CV_64F);
    hist_no_skin = Mat(bin_h_size,bin_s_size, CV_64F);
    
    
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
    
    //smooth image
    GaussianBlur(hsv, hsv, Size(blur_size, blur_size), 0, 0);
    
    
    //Seb with gauss
    if(gauss_on){
        for (int row = 0; row < hsv.rows; ++row) {
            for (int col = 0; col < hsv.cols; ++col) {
                
                cv::Vec3b hsv_ = hsv.at<cv::Vec3b>(row, col);
                float h__ = (float)hsv_[0];
                float s__ = (float)hsv_[1];
                
                float temp [] = {h__, s__};
                Mat hs_row = Mat(1, 2,  CV_32F, temp);
                
                if(mask(row, col) == 255){
                    
                    skin_gauss.push_back(hs_row);
                }
                else{
                    if(counter%7 ==0){
                        no_skin_gauss.push_back(hs_row);
                    }
                    counter++;
                    
                }
            }
        }
    }
    
    if(hist_on_there){
        /// Using 50 bins for hue and 60 for saturation
        int h_bins = 50; int s_bins = 60;
        //int h_bins = 10; int s_bins = 10;
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
      
    }
    
    if(hist_on){
        for (int row = 0; row < hsv.rows; ++row) {
            for (int col = 0; col < hsv.cols; ++col) {
                
                cv::Vec3b hsv_ = hsv.at<cv::Vec3b>(row, col);
                
                double h__ = hsv_[0];
                double s__ =  hsv_[1];
                int bin1  = (int) (h__/180 * hist_skin.rows);
                int bin2  = (int) (s__/256 * hist_skin.cols);
                
                
                if(mask(row,col) != 0){
                    hist_skin.at<double>(bin1,bin2)++;
                }else{
                    if(counter%2 ==0){
                        hist_no_skin.at<double>(bin1,bin2)++;
                    }
                    counter++;
                    if(counter == 32000){
                        counter = 0;
                    }
                    
                }
            }
        }
    }
}
/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
///
/// Implementation hint:
/// e.g normalize w.r.t. the number of training images etc.
void SkinModel::finishTraining()
{
    
    if(normalize_on){
        double hist_skin_sum = sum(hist_skin)[0];
        double hist_no_skin_sum = sum(hist_no_skin)[0];
        // cout << hist_skin_sum<<hist_no_skin_sum <<endl;
        hist_skin = (1.0/hist_skin_sum) * hist_skin;
        hist_no_skin = (1.0/hist_no_skin_sum) * hist_no_skin;
        
        normalize(hist_skin, hist_skin, 0, 255, NORM_MINMAX, -1, Mat() );
        normalize(hist_no_skin, hist_no_skin, 0, 255, NORM_MINMAX, -1, Mat() );
    }
 
    
      // GaussianBlur(hist_skin, hist_skin, Size(blur_size, blur_size), 0, 0);
    //cout<<hist_skin<<endl;
    //cout<<sum(hist_skin)[0]<<endl;
    //cv::namedWindow("gauss", WINDOW_AUTOSIZE);
    //cv::imshow("gauss", hist_skin);
    // waitKey(0);
    
    if(gauss_on){
        
        cv::Mat skin_Covar, skin_Mu;
        cv::Mat no_skin_Covar, no_skin_Mu;
        //calcualte cov and mean of skin
        cv::calcCovarMatrix(skin_gauss,skin_Covar, skin_Mu, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE, CV_32F );
        
        
        //calcualte cov and mean of no _skin
        cv::calcCovarMatrix(no_skin_gauss, no_skin_Covar, no_skin_Mu, CV_COVAR_NORMAL | CV_COVAR_ROWS |CV_COVAR_SCALE,CV_32F );
        cout<<no_skin_Covar << no_skin_Mu<<endl;
        cout<<skin_Covar << skin_Mu<<endl;
        double det_cov_skin = determinant(skin_Covar);
        Mat cov_inv_skin = skin_Covar.inv();
        
        double det_cov_no_skin = determinant(no_skin_Covar);
        Mat cov_inv_no_skin = no_skin_Covar.inv();
        
        for (int row = 0; row < gauss_values_skin.rows; ++row) {
            for (int col = 0; col < gauss_values_skin.cols; ++col) {
                
                
                float temp [] = {(float)row, (float)col};
                Mat x = Mat(1, 2,  CV_32F, temp);
                
                
                Mat x_mu_skin = x - skin_Mu;
                Mat x_mu_skin_t = x_mu_skin.t();
                
                double first_skin = 1/(2 * M_PI) * 1/sqrt(det_cov_skin);
                Mat exp_skin = -0.5 * x_mu_skin * cov_inv_skin * x_mu_skin_t;
                double exponent_skin = (double)exp_skin.at<float>(0,0);
                
                double prob_skin = first_skin * exp(exponent_skin);
                gauss_values_skin.at<double>(row,col) = prob_skin;
                
                
                
                Mat x_mu_no_skin = x - no_skin_Mu;
                Mat x_mu_no_skin_t = x_mu_no_skin.t();
                
                double first_no_skin = 1/(2 * M_PI) * 1/sqrt(det_cov_no_skin);
                Mat exp_no_skin = -0.5 * x_mu_no_skin * cov_inv_no_skin * x_mu_no_skin_t;
                double exponent_no_skin = (double)exp_no_skin.at<float>(0,0);
                
                
                double prob_no_skin = first_no_skin * exp(exponent_no_skin);
                gauss_values_no_skin.at<double>(row,col) = prob_no_skin;
                
                
            }
        }
        
        
        
    }
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
    GaussianBlur(hsv, hsv, Size(blur_size, blur_size), 0, 0);
    //--- IMPLEMENT THIS ---//
    // cout << hist_skin<< endl;
    //  cout << hist_no_skin<< endl;
    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            //Martina
            if (hist_on) {
                cv::Vec3b pixel = hsv.at<cv::Vec3b>(row, col);
                float h_bin_size = 180.0/(double)bin_h_size;
                float s_bin_size = 256.0/(double)bin_s_size;
                int h = ((int)pixel[0])/h_bin_size;
                int s = ((int)pixel[1])/s_bin_size;
                
                if (hist_skin.at<double>(h, s) > hist_no_skin.at<double>(h, s)) {
                    skin(row,col) = (int)(hist_skin.at<double>(h, s)/hist_no_skin.at<double>(h, s) * 255);
                    //skin(row, col) = (int)hist_skin.at<double>(h, s);
                }
            }
            
            //Seb gauss
            if (gauss_on) {
                //for skin use formular slide 31
                cv::Vec3b pixel = hsv.at<cv::Vec3b>(row, col);
                int h = (int)pixel[0];
                int s = (int)pixel[1];
                
                double prob_skin = gauss_values_skin.at<double>(h,s);
                double prob_no_skin = gauss_values_no_skin.at<double>(h,s);
                
                if (prob_skin > prob_no_skin) {
                    skin(row, col) = prob_skin/prob_no_skin * 128;
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
    
    
    //modify in order to change impact of opening closing
    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    
    
    //erosion and dilation to reduce mistakes
    //first morphological closing (delatation -> erosion) and then opening (erosion -> delatation)
    //MORPH_CLOSE 3     MORPH_OPEN 2
    cv::morphologyEx( skin, skin, MORPH_CLOSE, element);
    cv::morphologyEx( skin, skin, MORPH_OPEN, element);
    
    
    return skin;
}