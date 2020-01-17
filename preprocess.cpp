#include "preprocess.h"
#include "wavelet.h"
#include "other.h"
void preprocess(Mat &oringal, Mat &output, int ratio) {
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat raw = oringal.clone();
	Mat sobel_response, mask, tmp_mask;
	int threshold_value = 0;
	double max, min;
	Sobel(oringal, sobel_response, ddepth, 1, 1, 3, scale, delta, BORDER_DEFAULT);
	Size s = sobel_response.size();
	// set border to zero
	sobel_response.rowRange(0, 11).setTo(Scalar(0));
	sobel_response.rowRange(s.height - 10, s.height).setTo(Scalar(0));
	sobel_response = sobel_response *(float)-1;
	minMaxLoc(sobel_response, &min, &max);
	//sobel_response = (sobel_response - min) / max * 255;
	raw.rowRange(0, 11).setTo(Scalar(0));
	raw.rowRange(s.height - 10, s.height).setTo(Scalar(0));
	threshold(raw, mask, 15000, 65535, 0);
	mask.convertTo(mask, CV_8U);
	Canny(mask, mask, 150, 255);
	//bwareaopen(mask, 10);
	mask.convertTo(mask, CV_16S);
	// looking for the first layer or last layer 
	Mat filter_edge = Mat(mask.rows, mask.cols, CV_16S, cvScalar(0.));
	int width = mask.cols;
	int img_size = mask.cols*mask.rows;
	float max_val = 255;
	float *idx_output = new float[width];
	float *img = (float *)mask.data;
	void find_location(float *img, float *idx_output, int width, float max_val);
	for (int i = 0; i < width; ++i)
	{
		filter_edge.at<float>(i, (int)idx_output[i]) = 255;
	}
	output = ratio*filter_edge + sobel_response;
}



void gpu_wavelet(float *input, int width, int height,bool nlevels,int i_thre,bool save_flag, char* save_path)
{
	int array_len = width*height;
	float *tmp_output = new float[array_len];
	char wname[] = { "db1" };
	wavelet W(input, width, height, nlevels, i_thre, save_flag);
	W.dualtree();
	W.load_img(tmp_output);
	if(save_flag)
		writefile(tmp_output, (width)*(height),save_path);
	delete[]tmp_output;
}