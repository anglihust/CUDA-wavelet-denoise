#pragma once
#ifndef WAVELET_H
#define WAVELET_H
#define NULL 0
#include "cublas_v2.h"

class wavelet
{
private:
	float *img;
	int width;
	int height;
	int levels;
	int save_img;
	float thre;
	float **d_coeffs;
	float **d_coeffs_b;
	float *d_img_pad_1;
	float *d_img_pad_2;
	float *d_img; // space in gpu to restore image
	float *d_output;
	float *d_tmp;
	float * d_tmp_b;
	cublasHandle_t handle;
public:
	wavelet();
	wavelet(float *img, int width, int row, int levels, float i_thre, int save_flag);
	~wavelet();
	void load_img(float *l_img);
	void dualtree();
	void runtest(float *l_img);
};

#endif