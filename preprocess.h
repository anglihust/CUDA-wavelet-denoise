#pragma once
#ifndef PREPROCESS_H
#define PREPROCESS_H
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
void preprocess(Mat &oringal, Mat &output, int ratio);
void gpu_wavelet(float *input, int width, int height, bool nlevels, int i_thre, bool save_flag, char* save_path);

#endif