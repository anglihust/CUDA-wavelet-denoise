#pragma once
#ifndef SEPERABLE_H
#define SEPERABLE_H
#include <stdio.h> 
#include <math.h>
#include <stdlib.h>
#include "other.h" 
#include "cuda_runtime.h" 
#include "cublas_v2.h"

#define NULL 0


// func list///
// create buffer for wavelet transform
float **create_buffer_coeffcient(int img_width, int img_row, int levels);// create buffer for wavelet coeffcients on CUDA
void free_buffer_coeffcient(float **buffer, int levels); // free wavelet coeffcients created on cuda 

// load filter to cuda memory
int compute_sperable_filter(int option);// select wavelet type:0/1 for seperable wavelet; 2 for dual tree wavelet. return length of filter(hlen)

//other operation
__global__ void up_sampling_width(float* ary1, float* ary2, float*output_ary1, float*output_ary2, int height, int width);
__global__ void down_sampling_width(float* ary1, float* ary2, float*output_ary1, float*output_ary2, int height, int width);
__global__ void up_sampling_height(float* ary1, float* ary2, float* ary3, float* ary4, float*output_ary1, float*output_ary2, float*output_ary3, float*output_ary4, int height, int width);
__global__ void down_sampling_height(float* ary1, float* ary2, float* ary3, float* ary4, float*output_ary1, float*output_ary2, float*output_ary3, float*output_ary4, int height, int width);
__global__ void padding_width(float* ary, float*output_ary, int height, int width, int pad_num);
__global__ void padding_height(float* ary1, float* ary2, float*output_ary1, float*output_ary2, int height, int width, int pad_num);

//forward pass
void w_forward(float *d_img, float **d_coeffs, float **dd_coeffs, float* d_img_pad_1, float* d_img_pad_2, int width, int height, int levels, cublasHandle_t handle);
__global__ void w_kernel_foward_pass1_1st(float *d_image, float *d_tmp1, float *d_tmp2, int height, int width, float*tmpout);// dual tree kernel 1 
__global__ void w_kernel_foward_pass2_1st(float *d_tmp1, float *d_tmp2, float *o_a, float *o_dv, float *o_dh, float *o_dd, int height, int width, float*tmpout);

__global__ void w_kernel_foward_pass1(float *d_image, float *d_tmp1, float *d_tmp2, int height, int width, float*tmpout);// dual tree kernel 1 
__global__ void w_kernel_foward_pass2(float *d_tmp1, float *d_tmp2, float *o_a, float *o_dv, float *o_dh, float *o_dd, int height, int width, float*tmpout);

__global__ void w_kerneld_foward_pass1_1st(float *d_image, float *d_tmp1, float *d_tmp2, int height, int width, float*tmpout);// dual tree kernel 2
__global__ void w_kerneld_foward_pass2_1st(float *d_tmp1, float *d_tmp2, float *o_a, float *o_dv, float *o_dh, float *o_dd, int height, int width, float*tmpout);

__global__ void w_kerneld_foward_pass1(float *d_image, float *d_tmp1, float *d_tmp2, int height, int width, float*tmpout);// dual tree kernel 2
__global__ void w_kerneld_foward_pass2(float *d_tmp1, float *d_tmp2, float *o_a, float *o_dv, float *o_dh, float *o_dd, int height, int width, float*tmpout);

//sum up the coeffcients and then normalized
void w_coeffcients_sum(float **d_coeffs_a, float ** d_coeffs_b, float*d_img_pad_1, int width, int row, int levels, cublasHandle_t handle);

//apply soft filter
void w_shrinkage(float **d_coeffs_a, float **d_coeffs_b, float thre, int width, int row, int levels);
__global__ void w_kernel_shrinkage_single(float *o_a, float thre, int height, int width);
__global__ void w_kernel_shrinkage_multi(float *o_dv, float *o_dh, float *o_dd, float thre, int height, int width);

//inverse wavelet
// sum up the coeffcients and then normalized
void w_inverse(float **d_coeffs, float **dd_coeffs, float *d_img, float* d_img_pad_1, float*d_img_pad_2, int width, int height, int levels, cublasHandle_t handle);
__global__ void w_kernel_inverse_pass1_1st(float *o_a, float *o_dv, float *o_dh, float *o_dd, float *d_tmp1, float *d_tmp2, int width, int height);
__global__ void w_kernel_inverse_pass2_1st(float* d_tmp1, float* d_tmp2, float* img, int width, int height);

__global__ void w_kernel_inverse_pass1(float *o_a, float *o_dv, float *o_dh, float *o_dd, float *d_tmp1, float *d_tmp2, int width, int height);
__global__ void w_kernel_inverse_pass2(float* d_tmp1, float* d_tmp2, float* img, int width, int height);

__global__ void w_kerneld_inverse_pass1_1st(float *o_a, float *o_dv, float *o_dh, float *o_dd, float *d_tmp1, float *d_tmp2, int width, int height);
__global__ void w_kerneld_inverse_pass2_1st(float* d_tmp1, float* d_tmp2, float* img, int width, int height);

__global__ void w_kerneld_inverse_pass1(float *o_a, float *o_dv, float *o_dh, float *o_dd, float *d_tmp1, float *d_tmp2, int width, int height);
__global__ void w_kerneld_inverse_pass2(float* d_tmp1, float* d_tmp2, float* img, int width, int height);

struct filter
{
	char name[16];
	int length;
	float *f_l; //foward lowpass filter
	float *f_h; //foward highpass filter
	float *i_l; //inverse lowpass filter
	float *i_h; //imverse highpass filter
};

//others
void w_image_normalization(float *d_img, int width, int row, cublasHandle_t handle);
void w_coeff_normalization(float **d_coeffs, int width, int row, int levels, cublasHandle_t handle);
//void w_call_circshift(float *input, float* output, int img_width, int img_row, int sr, int sc);//shift image. used in seperable filter
__global__ void w_kern_circshift(float* d_image, float* d_out, int Nr, int Nc, int sr, int sc);// cuda kernel for shift image 

#endif