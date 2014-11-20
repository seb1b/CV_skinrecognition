///
///  Assignment 1
///  Skin Color Classification
///
///  Group Number:
///  Authors:
///
#define _OPENCV_FLANN_HPP_
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <iostream>
#include <fstream>
using namespace std;

#include "skinmodel.h"
#include "ROC.h"
using namespace std;



int main(int argc, char* argv[]) {
	
	/// parse command line options
	boost::program_options::variables_map pom;
	{
		namespace po = boost::program_options;
		po::options_description pod(string("Allowed options for ")+argv[0]);
		pod.add_options() 
			("help,h", "produce this help message")
			("gui,s", "Enable the GUI")
			("out,o", po::value<string>(), "Path where to write the results")
			("path", po::value<string>()->default_value("."), "Path where to read input data");

		po::positional_options_description pop;
		pop.add("path", -1);

		po::store(po::command_line_parser( argc, argv ).options(pod).positional(pop).run(), pom);
		po::notify(pom);

		if (pom.count("help")) {
			cout << "Usage:" << endl <<  pod << "\n";
			return 0;
		}
	}
	
	/// get image filenames
	string path = pom["path"].as<string>();
	vector<string> trainImgs, testImgs;
	{
		namespace fs = boost::filesystem; 
		for (fs::directory_iterator it(fs::path(path+"/train")); it!=fs::directory_iterator(); it++)
			if (is_regular_file(*it) and it->path().filename().string().substr(0,5)!="mask-")
				trainImgs.push_back(it->path().filename().string());

		for (fs::directory_iterator it(fs::path(path+"/test")); it!=fs::directory_iterator(); it++)
			if (is_regular_file(*it) and it->path().filename().string().substr(0,5)!="mask-")
				testImgs.push_back(it->path().filename().string());
    } 
    
    /// create skin color model instance
    SkinModel model;

    /// train model with all images in the train folder
	model.startTraining();
	
	for (auto &f:trainImgs) {
		cout << "Training on Image " << path+"/train/"+f << endl;
		cv::Mat3b img = cv::imread(path+"/train/"+f);
		cv::Mat1b mask = cv::imread(path+"/train/mask-"+f,0);
		
		cv::threshold( mask, mask, 127, 255, cv::THRESH_BINARY); 
		model.train( img, mask );
	}
	
	model.finishTraining();
	
    /// test model with all images in the test folder, 
	ROC<int> roc;
	for (auto &f:testImgs) {
		cout << "Testing Image " << path+"/train/"+f << endl;
		cv::Mat3b img = cv::imread(path+"/test/"+f);
		cv::Mat1b hyp = model.classify(img);

		cv::Mat1b mask = cv::imread(path+"/test/mask-"+f,0);

		for (int i=0; i<hyp.rows; i++)
			for (int j=0; j<hyp.cols; j++)
				roc.add(mask(i,j)>127, hyp(i,j));
	}
	
	/// After training, update statistics and show results
	roc.update();
	
	cout << "Overall F1 score: " << roc.F1 << endl;
	
	/// Display final result if desired
	if (pom.count("gui")) {
		cv::imshow("ROC", roc.draw());
		cv::waitKey(0);
	}

	/// Ouput a summary of the data if required
	if (pom.count("out")) {
		
		string p = pom["out"].as<string>();
		
		/// GRAPH format with one FPR and TPR coordinates per line
		ofstream graph(p+"/graph.txt");
		for (auto &dot : roc.graph)
			graph << dot.first << " " << dot.second << endl;
		
		/// Single output of the F1 score
		ofstream score(p+"/score.txt");
		score << roc.F1 << endl;
		/// Ouput of the obtained ROC figure
		cv::imwrite(p+"/ROC.png", roc.draw());
	}
}

