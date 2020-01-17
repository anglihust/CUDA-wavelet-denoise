#include <cuda.h>
#include "cuda_runtime.h" 
#include <cuComplex.h> 
#include "wavelet.h"
#include "seperable.h"

wavelet::wavelet(void) :img(NULL), width(0), height(0), levels(0), save_img(0), thre(0){}
wavelet::wavelet(float *img, int width, int row, int levels, float i_thre, int save_flag) : img(img), width(width),height(row), levels(levels), thre(i_thre), d_coeffs(NULL), d_coeffs_b(NULL), d_img(NULL), d_output(NULL), d_tmp(NULL), d_tmp_b(NULL), save_img(save_flag)
{
	
	int hlen;
	float *d_arr_in;
	cudaMalloc(&d_arr_in, sizeof(float)* width* height);
	cudaMemcpy(d_arr_in, img, sizeof(float)*width*height, cudaMemcpyHostToDevice);
	d_img = d_arr_in;

	float *d_output_new;
	cudaMalloc(&d_output_new, sizeof(float)*width*height);
	d_output = d_output_new;

	float *d_tmp_new;
	cudaMalloc(&d_tmp_new, sizeof(float)*height*(width + 2 * 4) * 6);
	d_tmp = d_tmp_new;

	float *d_tmp_new_b;
	cudaMalloc(&d_tmp_new_b, sizeof(float)*height*(width + 2 * 4) * 6);
	d_tmp_b = d_tmp_new_b;

	int filter_option = 2;	
	hlen = compute_sperable_filter(filter_option);

	int N = __min(width, height);
	int max_level = log2(N / hlen);
	if (max_level<levels)
	{
		printf("input level is over max level\n");
		levels = max_level;
	}

	float **d_coeffs_temp;
	d_coeffs_temp = create_buffer_coeffcient(width, height, levels);
	d_coeffs = d_coeffs_temp;

	// create coefficient space for second tree 
	float **d_coeffs_tmp_b;
	d_coeffs_tmp_b = create_buffer_coeffcient(width, height, levels);
	d_coeffs_b = d_coeffs_tmp_b;
	cublasCreate(&handle);
}

wavelet::~wavelet(void)
{
	cudaFree(d_img);
	cudaFree(d_tmp);
	cudaFree(d_output);
	free_buffer_coeffcient(d_coeffs, levels);
	cudaFree(d_tmp_b);
	free_buffer_coeffcient(d_coeffs_b, levels);
	cublasDestroy(handle);
}

void wavelet::load_img(float *l_img) {
	cudaError_t cudaStatus;
	cudaStatus=cudaMemcpy(l_img, d_output, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
	}
}


void wavelet::dualtree(void)
{

	w_forward(d_img, d_coeffs, d_coeffs_b, d_tmp, d_tmp_b, width, height, levels, handle);

	w_coeffcients_sum(d_coeffs, d_coeffs_b, d_tmp, width, height, levels, handle);// d_coeffs_b:A+B  d_coeffs:A-B
	w_shrinkage(d_coeffs, d_coeffs_b, thre, width, height, levels);

	w_coeffcients_sum(d_coeffs_b, d_coeffs, d_tmp, width, height, levels, handle);// d_coeffs_b:B d_coeffs:A
	w_inverse(d_coeffs, d_coeffs_b, d_output, d_tmp, d_tmp_b, width, height, levels, handle);
}

void wavelet::runtest(float *l_img) {

}