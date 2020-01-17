#include "preprocess.h"
#include <iostream>
#include <string>
using namespace cv;
using namespace std;
int main() {
	//const char* Dir = "I:\tmp_test";
	//SetCurrentDirectory(Dir);
	//string filename("test.tif");
	char * string_read = "C:/Users/WeiOffice/Documents/Visual Studio 2015/Projects/opencv_test/opencv_test/test.jpg";
	char * string_save = "C:/Users/WeiOffice/Documents/Visual Studio 2015/Projects/opencv_test/opencv_test/test.bin";
	int width = 512, height = 512;// resize image to 512x512 
	int data_len = width*height;
	bool save_flag = 1;// wether save ouput 
	int nlevels = 3; // levels of wavelet denoise
	float i_thre = 20;// treshold 

	Mat onechannel;
	Mat img = imread(string_read);
	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", img);
	waitKey(0);

	img.convertTo(img, CV_32FC1);
	cvtColor(img, onechannel, CV_BGR2GRAY);
	Mat roriginal(width, height, CV_32FC1, Scalar::all(0));
	resize(onechannel, roriginal, roriginal.size(), 0, 0, INTER_LINEAR);
	float *input = new float[data_len];
	for (int j = 0; j < height; ++j) {
		for (int i = 0; i <width; ++i) {
			input[j*width + i] = onechannel.at<float>(j, i);
		}
	}
 	gpu_wavelet(input, width, height, save_flag, nlevels, i_thre, string_save);
	//waitKey(0);
	//system("pause");
	return 0;
}