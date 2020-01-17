// seperate_cuda.cu
// implementation of 
#include "seperable.h" 
#include "device_launch_parameters.h"

#define Thread 32
#define Border 4 
#define h_left 4 
#define h_right 5
#define hlen 10
// first stage for tree 1
__constant__ float fdevice_L[40];
__constant__ float fdevice_H[40];
__constant__ float fdevice_IL[40];
__constant__ float fdevice_IH[40];
//following stages for tree 1
__constant__ float device_L[40];
__constant__ float device_H[40];
__constant__ float device_IL[40];
__constant__ float device_IH[40];

// first stage for tree 2
__constant__ float fdevice_L_1[40];
__constant__ float fdevice_H_1[40];
__constant__ float fdevice_IL_1[40];
__constant__ float fdevice_IH_1[40];

//following stages for tree 2
__constant__ float device_L_1[40];
__constant__ float device_H_1[40];
__constant__ float device_IL_1[40];
__constant__ float device_IH_1[40];

float FDUALTREE_1_L[10] = {
	0.0351638400000000,
	0,
	-0.0883294200000000,
	0.233890320000000,
	0.760272370000000,
	0.587518300000000,
	0,
	-0.114301840000000,
	0,
	0
};

float FDUALTREE_1_H[10] = {
	0,
	0,
	-0.114301840000000,
	0,
	0.587518300000000,
	-0.760272370000000,
	0.233890320000000,
	0.0883294200000000,
	0,
	-0.0351638400000000
};

float FDUALTREE_1_IL[10] = {
	0,
	0,
	-0.114301840000000,
	0,
	0.587518300000000,
	0.760272370000000,
	0.233890320000000,
	-0.0883294200000000,
	0,
	0.0351638400000000
};

float FDUALTREE_1_IH[10] = {
	-0.0351638400000000,
	0,
	0.0883294200000000,
	0.233890320000000,
	-0.760272370000000,
	0.587518300000000,
	0,
	-0.114301840000000,
	0,
	0
};

float DUALTREE_1_L[10] = {
	0,
	-0.0883883476483200,
	0.0883883476483200,
	0.695879989034000,
	0.695879989034000,
	0.0883883476483200,
	-0.0883883476483200,
	0.0112267921525400,
	0.0112267921525400,
	0
};

float DUALTREE_1_H[10] = {
	0,
	-0.0112267921525400,
	0.0112267921525400,
	0.0883883476483200,
	0.0883883476483200,
	-0.695879989034000,
	0.695879989034000,
	-0.0883883476483200,
	-0.0883883476483200,
	0
};

float DUALTREE_1_IL[10] = {
	0,
	0.0112267921525400,
	0.0112267921525400,
	-0.0883883476483200,
	0.0883883476483200,
	0.695879989034000,
	0.695879989034000,
	0.0883883476483200,
	-0.0883883476483200,
	0
};

float DUALTREE_1_IH[10] = {
	0,
	-0.0883883476483200,
	-0.0883883476483200,
	0.695879989034000,
	-0.695879989034000,
	0.0883883476483200,
	0.0883883476483200,
	0.0112267921525400,
	-0.0112267921525400,
	0
};

float FDUALTREE_2_L[10] = {
	0,
	0,
	-0.114301840000000,
	0,
	0.587518300000000,
	0.760272370000000,
	0.233890320000000,
	-0.0883294200000000,
	0,
	0.0351638400000000
};

float FDUALTREE_2_H[10] = {
	-0.0351638400000000,
	0,
	0.0883294200000000,
	0.233890320000000,
	-0.760272370000000,
	0.587518300000000,
	0,
	-0.114301840000000,
	0,
	0
};

float FDUALTREE_2_IL[10] = {
	0.0351638400000000,
	0,
	-0.0883294200000000,
	0.233890320000000,
	0.760272370000000,
	0.587518300000000,
	0,
	-0.114301840000000,
	0,
	0
};

float FDUALTREE_2_IH[10] = {
	0,
	0,
	-0.114301840000000,
	0,
	0.587518300000000,
	-0.760272370000000,
	0.233890320000000,
	0.0883294200000000,
	0,
	-0.0351638400000000
};

float DUALTREE_2_L[10] = {
	0.0112267921525400,
	0.0112267921525400,
	-0.0883883476483200,
	0.0883883476483200,
	0.695879989034000,
	0.695879989034000,
	0.0883883476483200,
	-0.0883883476483200,
	0,
	0
};

float DUALTREE_2_H[10] = {
	0,
	0,
	-0.0883883476483200,
	-0.0883883476483200,
	0.695879989034000,
	-0.695879989034000,
	0.0883883476483200,
	0.0883883476483200,
	0.0112267921525400,
	-0.0112267921525400
};

float DUALTREE_2_IL[10] = {
	0,
	0,
	-0.0883883476483200,
	0.0883883476483200,
	0.695879989034000,
	0.695879989034000,
	0.0883883476483200,
	-0.0883883476483200,
	0.0112267921525400,
	0.0112267921525400
};

float DUALTREE_2_IH[10] = {
	-0.0112267921525400,
	0.0112267921525400,
	0.0883883476483200,
	0.0883883476483200,
	-0.695879989034000,
	0.695879989034000,
	-0.0883883476483200,
	-0.0883883476483200,
	0,
	0
};


filter filter_list[4] = {
	{ "dualtree_1_f", 10,FDUALTREE_1_L, FDUALTREE_1_H, FDUALTREE_1_IL,FDUALTREE_1_IH },
	{ "dualtree_1", 10,DUALTREE_1_L, DUALTREE_1_H, DUALTREE_1_IL,DUALTREE_1_IH },
	{ "dualtree_2_f", 10,FDUALTREE_2_L, FDUALTREE_2_H, FDUALTREE_2_IL,FDUALTREE_2_IH },
	{ "dualtree_2",10,DUALTREE_2_L, DUALTREE_2_H, DUALTREE_2_IL,DUALTREE_2_IH }
};

// initial cuda and allocate space 
void initial_cuda() {

}


// create buffer for wavelet transform
float **create_buffer_coeffcient(int img_width, int img_row, int levels)
{
	int img_width_temp = div2(img_width);
	int img_row_temp = div2(img_row);
	float **buffer = (float **)calloc(levels * 3 + 1, sizeof(float *));
	for (int i = 1; i <= levels; ++i)
	{
		img_width = div2(img_width);
		img_row = div2(img_row);
		cudaMalloc(&(buffer[(i - 1) * 3 + 1]), img_width*img_row * sizeof(float));
		cudaMemset(buffer[(i - 1) * 3 + 1], 0, img_width*img_row * sizeof(float));
		cudaMalloc(&(buffer[(i - 1) * 3 + 2]), img_width*img_row * sizeof(float));
		cudaMemset(buffer[(i - 1) * 3 + 2], 0, img_width*img_row * sizeof(float));
		cudaMalloc(&(buffer[(i - 1) * 3 + 3]), img_width*img_row * sizeof(float));
		cudaMemset(buffer[(i - 1) * 3 + 3], 0, img_width*img_row * sizeof(float));
	}
	cudaMalloc(&(buffer[0]), img_width_temp*img_row_temp * sizeof(float));
	cudaMemset(buffer[0], 0, img_width_temp*img_row_temp * sizeof(float));
	return buffer;
}

void free_buffer_coeffcient(float **buffer, int levels)
{
	for (int i = 0; i<3 * levels + 1; ++i)
	{
		cudaFree(buffer[i]);
	}
	free(buffer);
}
//


// load filter to cuda memory
int compute_sperable_filter(int option)
{
	int length;
	float *fw_l, *fw_h, *in_l, *in_h;
	float *fw_l_1 = NULL, *fw_h_2 = NULL, *in_l_1 = NULL, *in_h_2 = NULL;
	if (option == 0)
	{
		length = filter_list[0].length;
		fw_l = filter_list[0].f_l;
		fw_h = filter_list[0].f_h;
		in_l = filter_list[0].i_l;
		in_h = filter_list[0].i_h;

		cudaMemcpyToSymbol(device_L, fw_l, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(device_H, fw_h, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(device_IL, in_l, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(device_IH, in_h, length * sizeof(float), 0, cudaMemcpyHostToDevice);
	}
	else
	{
		float *ffw_l, *ffw_h, *fin_l, *fin_h;//first stage filter for tree 1 
		float *ffw_l_1, *ffw_h_2, *fin_l_1, *fin_h_2;//first stage filter for tree 2
		length = filter_list[0].length;
		ffw_l = filter_list[1].f_l;
		ffw_h = filter_list[1].f_h;
		fin_l = filter_list[1].i_l;
		fin_h = filter_list[1].i_h;

		fw_l = filter_list[0].f_l;
		fw_h = filter_list[0].f_h;
		in_l = filter_list[0].i_l;
		in_h = filter_list[0].i_h;

		ffw_l_1 = filter_list[3].f_l;
		ffw_h_2 = filter_list[3].f_h;
		fin_l_1 = filter_list[3].i_l;
		fin_h_2 = filter_list[3].i_h;

		fw_l_1 = filter_list[2].f_l;
		fw_h_2 = filter_list[2].f_h;
		in_l_1 = filter_list[2].i_l;
		in_h_2 = filter_list[2].i_h;

		cudaMemcpyToSymbol(fdevice_L, ffw_l, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(fdevice_H, ffw_h, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(fdevice_IL, fin_l, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(fdevice_IH, fin_h, length * sizeof(float), 0, cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(device_L, fw_l, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(device_H, fw_h, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(device_IL, in_l, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(device_IH, in_h, length * sizeof(float), 0, cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(fdevice_L_1, ffw_l_1, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(fdevice_H_1, ffw_h_2, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(fdevice_IL_1, fin_l_1, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(fdevice_IH_1, fin_h_2, length * sizeof(float), 0, cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(device_L_1, fw_l_1, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(device_H_1, fw_h_2, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(device_IL_1, in_l_1, length * sizeof(float), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(device_IH_1, in_h_2, length * sizeof(float), 0, cudaMemcpyHostToDevice);
	}

	return length;
}
//

//other operation
__global__ void up_sampling_width(float* ary1, float* ary2, float*output_ary1, float*output_ary2, int height, int width) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int idx;
	if (gidx < 2 * width && gidy < height) {
		idx = gidx / 2;
		output_ary1[2 * gidy*width + gidx] = ary1[gidy*width + idx];
		output_ary2[2 * gidy*width + gidx] = ary2[gidy*width + idx];
	}
}

__global__ void down_sampling_width(float* ary1, float* ary2, float*output_ary1, float*output_ary2, int height, int width) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int idx;
	if (gidx < width / 2 && gidy < height) {
		idx = gidx * 2;
		output_ary1[gidy*width / 2 + gidx] = ary1[gidy*width + idx];
		output_ary2[gidy*width / 2 + gidx] = ary2[gidy*width + idx];
	}
}

__global__ void up_sampling_height(float* ary1, float* ary2, float* ary3, float* ary4, float*output_ary1, float*output_ary2, float*output_ary3, float*output_ary4, int height, int width) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int idy;
	if (gidx < width && gidy <2 * height) {
		idy = gidy / 2;
		output_ary1[gidy*width + gidx] = ary1[idy*width + gidx];
		output_ary2[gidy*width + gidx] = ary2[idy*width + gidx];
		output_ary3[gidy*width + gidx] = ary3[idy*width + gidx];
		output_ary4[gidy*width + gidx] = ary4[idy*width + gidx];
	}
}

__global__ void down_sampling_height(float* ary1, float* ary2, float* ary3, float* ary4, float*output_ary1, float*output_ary2, float*output_ary3, float*output_ary4, int height, int width) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int idy;
	if (gidx < width && gidy <height / 2) {
		idy = gidy * 2;
		output_ary1[gidy*width + gidx] = ary1[idy*width + gidx];
		output_ary2[gidy*width + gidx] = ary2[idy*width + gidx];
		output_ary3[gidy*width + gidx] = ary3[idy*width + gidx];
		output_ary4[gidy*width + gidx] = ary4[idy*width + gidx];
	}
}

__global__ void padding_width(float* ary, float*output_ary, int height, int width, int pad_num) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	if (gidx < (width + 2 * pad_num + 1) && gidy < height) {
		if (gidx<pad_num) {
			output_ary[gidy*(width + 2 * pad_num + 1) + gidx] = ary[gidy*width + width - pad_num + gidx];
		}
		else if (gidx >= (width + pad_num)) {
			output_ary[gidy*(width + 2 * pad_num + 1) + gidx] = ary[gidy*width + gidx - width - pad_num];
		}
		else {
			output_ary[gidy*(width + 2 * pad_num + 1) + gidx] = ary[gidy*width + (gidx - pad_num)];
		}
	}
}

__global__ void padding_height(float* ary1, float* ary2, float*output_ary1, float*output_ary2, int height, int width, int pad_num) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	if (gidx < width && gidy < (height + 2 * pad_num + 1)) {
		if (gidy<pad_num) {
			output_ary1[gidy*width + gidx] = ary1[(height - pad_num + gidy)*width + gidx];
			output_ary2[gidy*width + gidx] = ary2[(height - pad_num + gidy)*width + gidx];
		}
		else if (gidy >= (height + pad_num)) {
			output_ary1[gidy*width + gidx] = ary1[(gidy - height - pad_num)*width + gidx];
			output_ary2[gidy*width + gidx] = ary2[(gidy - height - pad_num)*width + gidx];
		}
		else {
			output_ary1[gidy *width + gidx] = ary1[(gidy - pad_num)*width + gidx];
			output_ary2[gidy *width + gidx] = ary2[(gidy - pad_num)*width + gidx];

		}
	}
}

//forward pass
void w_forward(float *d_img, float **d_coeffs, float **dd_coeffs, float* d_img_pad_1, float* d_img_pad_2, int width, int height, int levels, cublasHandle_t handle)
{
	// change to multi-stream in next version
	float*d_tmp1, *d_tmp2, *d_tmp3, *d_tmp4, *d_tmp5, *d_tmp6, *tmpout;//temporal cuda space for tree 1
	float *dd_tmp1, *dd_tmp2, *dd_tmp3, *dd_tmp4, *dd_tmp5, *dd_tmp6;//temporal cuda space for tree 2
	int pad_num = Border;
	int threadnum = 32;
	int pad_length = height*(width + 2 * pad_num);
	float alpha = 1 / sqrt(2);
	cudaError_t err = cudaSuccess;
	//err = cudaMalloc(&d_img_pad_1, sizeof(float)*6*height*(width + 2 * pad_num));
	//err = cudaMalloc(&d_img_pad_2, sizeof(float)*6*height*(width + 2 * pad_num));
	//if (err != cudaSuccess) {
	//	fprintf(stderr, "Failed to allocate (error code %s)!\n", cudaGetErrorString(err));
	//}
	d_tmp6 = d_img_pad_1;
	d_tmp1 = d_tmp6 + pad_length;
	d_tmp2 = d_tmp1 + pad_length;
	d_tmp3 = d_tmp2 + pad_length;
	d_tmp4 = d_tmp3 + pad_length;
	d_tmp5 = d_tmp4 + pad_length;

	dd_tmp6 = d_img_pad_2;
	dd_tmp1 = dd_tmp6 + pad_length;
	dd_tmp2 = dd_tmp1 + pad_length;
	dd_tmp3 = dd_tmp2 + pad_length;
	dd_tmp4 = dd_tmp3 + pad_length;
	dd_tmp5 = dd_tmp4 + pad_length;
	tmpout = d_tmp6;
	dim3 threadperblock(threadnum, threadnum);

	cublasSscal(handle, width*height, &alpha, d_img, 1);

	for (int i = 0; i < levels; ++i) {
		if (i == 0)
		{
			dim3 blockpergrid1((int(width + 2 * pad_num + 1 - 0.5) / threadnum) + 1, int((height - 0.5) / threadnum) + 1);
			padding_width << <blockpergrid1, threadperblock >> > (d_img, d_tmp6, height, width, pad_num);
			padding_width << <blockpergrid1, threadperblock >> > (d_img, dd_tmp6, height, width, pad_num);

			dim3 blockpergrid2((int(width - 0.5) / threadnum) + 1, int((height - 0.5) / threadnum) + 1);
			w_kernel_foward_pass1_1st << <blockpergrid2, threadperblock >> > (d_tmp6, d_tmp1, d_tmp2, height, width, tmpout);
			w_kerneld_foward_pass1_1st << <blockpergrid2, threadperblock >> > (dd_tmp6, dd_tmp1, dd_tmp2, height, width, tmpout);

			dim3 blockpergrid3((int(width / 2 - 0.5) / threadnum) + 1, int((height - 0.5) / threadnum) + 1);
			down_sampling_width << <blockpergrid3, threadperblock >> > (d_tmp1, d_tmp2, d_tmp3, d_tmp4, height, width);
			down_sampling_width << <blockpergrid3, threadperblock >> > (dd_tmp1, dd_tmp2, dd_tmp3, dd_tmp4, height, width);

			dim3 blockpergrid4((int(width / 2 - 0.5) / threadnum) + 1, int((height + 2 * pad_num + 1 - 0.5) / threadnum) + 1);
			padding_height << <blockpergrid4, threadperblock >> > (d_tmp3, d_tmp4, d_tmp1, d_tmp2, height, width / 2, pad_num);
			padding_height << <blockpergrid4, threadperblock >> > (dd_tmp3, dd_tmp4, dd_tmp1, dd_tmp2, height, width / 2, pad_num);

			w_kernel_foward_pass2_1st << <blockpergrid3, threadperblock >> > (d_tmp1, d_tmp2, d_tmp3, d_tmp4, d_tmp5, d_tmp6, height, width / 2, tmpout);
			w_kerneld_foward_pass2_1st << <blockpergrid3, threadperblock >> > (dd_tmp1, dd_tmp2, dd_tmp3, dd_tmp4, dd_tmp5, dd_tmp6, height, width / 2, tmpout);

			dim3 blockpergrid6((int(width / 2 - 0.5) / threadnum) + 1, int((height / 2 - 0.5) / threadnum) + 1);
			down_sampling_height << <blockpergrid6, threadperblock >> > (d_tmp3, d_tmp4, d_tmp5, d_tmp6, d_coeffs[0], d_coeffs[3 * i + 1], d_coeffs[3 * i + 2], d_coeffs[3 * i + 3], height, width / 2);
			down_sampling_height << <blockpergrid6, threadperblock >> > (dd_tmp3, dd_tmp4, dd_tmp5, dd_tmp6, dd_coeffs[0], dd_coeffs[3 * i + 1], dd_coeffs[3 * i + 2], dd_coeffs[3 * i + 3], height, width / 2);

		}
		else
		{
			height = div2(height);
			width = div2(width);
			dim3 blockpergrid1((int(width + 2 * pad_num + 1 - 0.5) / threadnum) + 1, int((height - 0.5) / threadnum) + 1);
			padding_width << <blockpergrid1, threadperblock >> > (d_coeffs[0], d_tmp6, height, width, pad_num);
			padding_width << <blockpergrid1, threadperblock >> > (dd_coeffs[0], dd_tmp6, height, width, pad_num);

			dim3 blockpergrid2((int(width - 0.5) / threadnum) + 1, int((height - 0.5) / threadnum) + 1);
			w_kernel_foward_pass1 << <blockpergrid2, threadperblock >> > (d_tmp6, d_tmp1, d_tmp2, height, width, tmpout);
			w_kerneld_foward_pass1 << <blockpergrid2, threadperblock >> > (dd_tmp6, dd_tmp1, dd_tmp2, height, width, tmpout);

			dim3 blockpergrid3((int(width / 2 - 0.5) / threadnum) + 1, int((height - 0.5) / threadnum) + 1);
			down_sampling_width << <blockpergrid3, threadperblock >> > (d_tmp1, d_tmp2, d_tmp3, d_tmp4, height, width);
			down_sampling_width << <blockpergrid3, threadperblock >> > (dd_tmp1, dd_tmp2, dd_tmp3, dd_tmp4, height, width);

			dim3 blockpergrid4((int(width / 2 - 0.5) / threadnum) + 1, int((height + 2 * pad_num + 1 - 0.5) / threadnum) + 1);
			padding_height << <blockpergrid4, threadperblock >> > (d_tmp3, d_tmp4, d_tmp1, d_tmp2, height, width / 2, pad_num);
			padding_height << <blockpergrid4, threadperblock >> > (dd_tmp3, dd_tmp4, dd_tmp1, dd_tmp2, height, width / 2, pad_num);

			w_kernel_foward_pass2 << <blockpergrid3, threadperblock >> > (d_tmp1, d_tmp2, d_tmp3, d_tmp4, d_tmp5, d_tmp6, height, width / 2, tmpout);
			w_kerneld_foward_pass2 << <blockpergrid3, threadperblock >> > (dd_tmp1, dd_tmp2, dd_tmp3, dd_tmp4, dd_tmp5, dd_tmp6, height, width / 2, tmpout);

			dim3 blockpergrid6((int(width / 2 - 0.5) / threadnum) + 1, int((height / 2 - 0.5) / threadnum) + 1);
			down_sampling_height << <blockpergrid6, threadperblock >> > (d_tmp3, d_tmp4, d_tmp5, d_tmp6, d_coeffs[0], d_coeffs[3 * i + 1], d_coeffs[3 * i + 2], d_coeffs[3 * i + 3], height, width / 2);
			down_sampling_height << <blockpergrid6, threadperblock >> > (dd_tmp3, dd_tmp4, dd_tmp5, dd_tmp6, dd_coeffs[0], dd_coeffs[3 * i + 1], dd_coeffs[3 * i + 2], dd_coeffs[3 * i + 3], height, width / 2);
		}
	}
}

// tree 1 forward
__global__ void w_kernel_foward_pass1_1st(float *d_image, float *d_tmp1, float *d_tmp2, int height, int width, float*tmpout) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;


	if (gidx<width && gidy<height) {
		__shared__ float sdata[Thread*(Thread + 2 * Border + 1)];
		int ii = tx + ty * Thread;
		int I = ii % (Thread + 2 * Border + 1);
		int J = ii / (Thread + 2 * Border + 1);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata[J*(Thread + 2 * Border + 1) + I] = d_image[s_global_idy*(width + 2 * Border + 1) + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		int I2 = ii2 % (Thread + 2 * Border + 1);
		int J2 = ii2 / (Thread + 2 * Border + 1);
		s_global_idx = blockIdx.x*blockDim.x + I2;
		s_global_idy = blockIdx.y*blockDim.y + J2;

		if (I2<(Thread + 2 * Border + 1) && J2<Thread && ii2<(Thread + 2 * Border + 1)*(Thread)) {
			sdata[J2*(Thread + 2 * Border + 1) + I2] = d_image[s_global_idy*(width + 2 * Border + 1) + s_global_idx];
		}

		__syncthreads();

		//if (gidx ==width-3 && gidy ==0) {
		//	for (int idi = 0; idi<(Thread+2* Border)*Thread; idi++)
		//		tmpout[idi] = sdata[idi];
		//}


		float tmp_store_1 = 0, tmp_store_2 = 0;
		float img_array;

		for (int i = 0; i <= h_left + h_right; ++i)
		{
			int idx_x = tx + i;
			img_array = sdata[ty*(Thread + 2 * Border + 1) + idx_x];
			tmp_store_1 += img_array*fdevice_L[hlen - 1 - i];
			tmp_store_2 += img_array*fdevice_H[hlen - 1 - i];
		}
		d_tmp1[gidy*width + gidx] = tmp_store_1;
		d_tmp2[gidy*width + gidx] = tmp_store_2;
	}
}

__global__ void w_kernel_foward_pass2_1st(float *d_tmp1, float *d_tmp2, float *o_a, float *o_dv, float *o_dh, float *o_dd, int height, int width, float*tmpout) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	if (gidx<width && gidy< height) {
		__shared__ float sdata1[Thread*(Thread + 2 * Border + 1)];
		__shared__ float sdata2[Thread*(Thread + 2 * Border + 1)];
		int ii = tx + ty * Thread;
		int I = ii % (Thread);
		int J = ii / (Thread);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata1[J*Thread + I] = d_tmp1[s_global_idy *width + s_global_idx];
		sdata2[J*Thread + I] = d_tmp2[s_global_idy *width + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		I = ii2 % (Thread);
		J = ii2 / (Thread);
		s_global_idx = blockIdx.x*blockDim.x + I;
		s_global_idy = blockIdx.y*blockDim.y + J;

		if (I<(Thread) && J<(Thread + 2 * Border + 1) && ii2<(Thread + 2 * Border + 1)*(Thread)) {
			sdata1[J*Thread + I] = d_tmp1[s_global_idy *width + s_global_idx];
			sdata2[J*Thread + I] = d_tmp2[s_global_idy *width + s_global_idx];
		}
		__syncthreads();

		//if (gidx == 0 && gidy == 0) {
		//	for (int idi = 0; idi<(Thread + 2 * Border)*Thread; idi++)
		//		tmpout[idi] = sdata1[idi];
		//}

		float tmp_store_1 = 0, tmp_store_2 = 0, tmp_store_3 = 0, tmp_store_4 = 0;
		float img_array_1, img_array_2;

		for (int i = 0; i <= h_left + h_right; ++i)
		{
			int idy = ty + i;
			img_array_1 = sdata1[idy*Thread + tx];
			img_array_2 = sdata2[idy*Thread + tx];
			tmp_store_1 += img_array_1 * fdevice_L[hlen - 1 - i];
			tmp_store_2 += img_array_1 * fdevice_H[hlen - 1 - i];
			tmp_store_3 += img_array_2 * fdevice_L[hlen - 1 - i];
			tmp_store_4 += img_array_2 * fdevice_H[hlen - 1 - i];
		}

		o_a[gidy *width + gidx] = tmp_store_1;
		o_dv[gidy*width + gidx] = tmp_store_2;
		o_dh[gidy*width + gidx] = tmp_store_3;
		o_dd[gidy*width + gidx] = tmp_store_4;
	}
}


__global__ void w_kernel_foward_pass1(float *d_image, float *d_tmp1, float *d_tmp2, int height, int width, float*tmpout) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;


	if (gidx<width && gidy<height) {
		__shared__ float sdata[Thread*(Thread + 2 * Border + 1)];
		int ii = tx + ty * Thread;
		int I = ii % (Thread + 2 * Border + 1);
		int J = ii / (Thread + 2 * Border + 1);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata[J*(Thread + 2 * Border + 1) + I] = d_image[s_global_idy*(width + 2 * Border + 1) + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		int I2 = ii2 % (Thread + 2 * Border + 1);
		int J2 = ii2 / (Thread + 2 * Border + 1);
		s_global_idx = blockIdx.x*blockDim.x + I2;
		s_global_idy = blockIdx.y*blockDim.y + J2;

		if (I2<(Thread + 2 * Border + 1) && J2<Thread && ii2<(Thread + 2 * Border + 1)*(Thread)) {
			sdata[J2*(Thread + 2 * Border + 1) + I2] = d_image[s_global_idy*(width + 2 * Border + 1) + s_global_idx];
		}

		__syncthreads();

		//if (gidx ==width-3 && gidy ==0) {
		//	for (int idi = 0; idi<(Thread+2* Border)*Thread; idi++)
		//		tmpout[idi] = sdata[idi];
		//}


		float tmp_store_1 = 0, tmp_store_2 = 0;
		float img_array;

		for (int i = 0; i <= h_left + h_right; ++i)
		{
			int idx_x = tx + i;
			img_array = sdata[ty*(Thread + 2 * Border + 1) + idx_x];
			tmp_store_1 += img_array*device_L[hlen - 1 - i];
			tmp_store_2 += img_array*device_H[hlen - 1 - i];
		}
		d_tmp1[gidy*width + gidx] = tmp_store_1;
		d_tmp2[gidy*width + gidx] = tmp_store_2;
	}
}

__global__ void w_kernel_foward_pass2(float *d_tmp1, float *d_tmp2, float *o_a, float *o_dv, float *o_dh, float *o_dd, int height, int width, float*tmpout) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	if (gidx<width && gidy< height) {
		__shared__ float sdata1[Thread*(Thread + 2 * Border + 1)];
		__shared__ float sdata2[Thread*(Thread + 2 * Border + 1)];
		int ii = tx + ty * Thread;
		int I = ii % (Thread);
		int J = ii / (Thread);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata1[J*Thread + I] = d_tmp1[s_global_idy *width + s_global_idx];
		sdata2[J*Thread + I] = d_tmp2[s_global_idy *width + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		I = ii2 % (Thread);
		J = ii2 / (Thread);
		s_global_idx = blockIdx.x*blockDim.x + I;
		s_global_idy = blockIdx.y*blockDim.y + J;

		if (I<(Thread) && J<(Thread + 2 * Border + 1) && ii2<(Thread + 2 * Border + 1)*(Thread)) {
			sdata1[J*Thread + I] = d_tmp1[s_global_idy *width + s_global_idx];
			sdata2[J*Thread + I] = d_tmp2[s_global_idy *width + s_global_idx];
		}
		__syncthreads();

		//if (gidx == 0 && gidy == 0) {
		//	for (int idi = 0; idi<(Thread + 2 * Border)*Thread; idi++)
		//		tmpout[idi] = sdata1[idi];
		//}

		float tmp_store_1 = 0, tmp_store_2 = 0, tmp_store_3 = 0, tmp_store_4 = 0;
		float img_array_1, img_array_2;

		for (int i = 0; i <= h_left + h_right; ++i)
		{
			int idy = ty + i;
			img_array_1 = sdata1[idy*Thread + tx];
			img_array_2 = sdata2[idy*Thread + tx];
			tmp_store_1 += img_array_1 * device_L[hlen - 1 - i];
			tmp_store_2 += img_array_1 * device_H[hlen - 1 - i];
			tmp_store_3 += img_array_2 * device_L[hlen - 1 - i];
			tmp_store_4 += img_array_2 * device_H[hlen - 1 - i];
		}

		o_a[gidy *width + gidx] = tmp_store_1;
		o_dv[gidy*width + gidx] = tmp_store_2;
		o_dh[gidy*width + gidx] = tmp_store_3;
		o_dd[gidy*width + gidx] = tmp_store_4;
	}
}
// tree 2 forward 
__global__ void w_kerneld_foward_pass1_1st(float *d_image, float *d_tmp1, float *d_tmp2, int height, int width, float*tmpout) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;


	if (gidx<width && gidy<height) {
		__shared__ float sdata[Thread*(Thread + 2 * Border + 1)];
		int ii = tx + ty * Thread;
		int I = ii % (Thread + 2 * Border + 1);
		int J = ii / (Thread + 2 * Border + 1);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata[J*(Thread + 2 * Border + 1) + I] = d_image[s_global_idy*(width + 2 * Border + 1) + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		int I2 = ii2 % (Thread + 2 * Border + 1);
		int J2 = ii2 / (Thread + 2 * Border + 1);
		s_global_idx = blockIdx.x*blockDim.x + I2;
		s_global_idy = blockIdx.y*blockDim.y + J2;

		if (I2<(Thread + 2 * Border + 1) && J2<Thread && ii2<(Thread + 2 * Border + 1)*(Thread)) {
			sdata[J2*(Thread + 2 * Border + 1) + I2] = d_image[s_global_idy*(width + 2 * Border + 1) + s_global_idx];
		}

		__syncthreads();

		//if (gidx ==width-3 && gidy ==0) {
		//	for (int idi = 0; idi<(Thread+2* Border)*Thread; idi++)
		//		tmpout[idi] = sdata[idi];
		//}


		float tmp_store_1 = 0, tmp_store_2 = 0;
		float img_array;

		for (int i = 0; i <= h_left + h_right; ++i)
		{
			int idx_x = tx + i;
			img_array = sdata[ty*(Thread + 2 * Border + 1) + idx_x];
			tmp_store_1 += img_array*fdevice_L_1[hlen - 1 - i];
			tmp_store_2 += img_array*fdevice_H_1[hlen - 1 - i];
		}
		d_tmp1[gidy*width + gidx] = tmp_store_1;
		d_tmp2[gidy*width + gidx] = tmp_store_2;
	}
}

__global__ void w_kerneld_foward_pass2_1st(float *d_tmp1, float *d_tmp2, float *o_a, float *o_dv, float *o_dh, float *o_dd, int height, int width, float*tmpout) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	if (gidx<width && gidy< height) {
		__shared__ float sdata1[Thread*(Thread + 2 * Border + 1)];
		__shared__ float sdata2[Thread*(Thread + 2 * Border + 1)];
		int ii = tx + ty * Thread;
		int I = ii % (Thread);
		int J = ii / (Thread);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata1[J*Thread + I] = d_tmp1[s_global_idy *width + s_global_idx];
		sdata2[J*Thread + I] = d_tmp2[s_global_idy *width + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		I = ii2 % (Thread);
		J = ii2 / (Thread);
		s_global_idx = blockIdx.x*blockDim.x + I;
		s_global_idy = blockIdx.y*blockDim.y + J;

		if (I<(Thread) && J<(Thread + 2 * Border + 1) && ii2<(Thread + 2 * Border + 1)*(Thread)) {
			sdata1[J*Thread + I] = d_tmp1[s_global_idy *width + s_global_idx];
			sdata2[J*Thread + I] = d_tmp2[s_global_idy *width + s_global_idx];
		}
		__syncthreads();

		//if (gidx == 0 && gidy == 0) {
		//	for (int idi = 0; idi<(Thread + 2 * Border)*Thread; idi++)
		//		tmpout[idi] = sdata1[idi];
		//}

		float tmp_store_1 = 0, tmp_store_2 = 0, tmp_store_3 = 0, tmp_store_4 = 0;
		float img_array_1, img_array_2;

		for (int i = 0; i <= h_left + h_right; ++i)
		{
			int idy = ty + i;
			img_array_1 = sdata1[idy*Thread + tx];
			img_array_2 = sdata2[idy*Thread + tx];
			tmp_store_1 += img_array_1 * fdevice_L_1[hlen - 1 - i];
			tmp_store_2 += img_array_1 * fdevice_H_1[hlen - 1 - i];
			tmp_store_3 += img_array_2 * fdevice_L_1[hlen - 1 - i];
			tmp_store_4 += img_array_2 * fdevice_H_1[hlen - 1 - i];
		}

		o_a[gidy *width + gidx] = tmp_store_1;
		o_dv[gidy*width + gidx] = tmp_store_2;
		o_dh[gidy*width + gidx] = tmp_store_3;
		o_dd[gidy*width + gidx] = tmp_store_4;
	}
}


__global__ void w_kerneld_foward_pass1(float *d_image, float *d_tmp1, float *d_tmp2, int height, int width, float*tmpout) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;


	if (gidx<width && gidy<height) {
		__shared__ float sdata[Thread*(Thread + 2 * Border + 1)];
		int ii = tx + ty * Thread;
		int I = ii % (Thread + 2 * Border + 1);
		int J = ii / (Thread + 2 * Border + 1);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata[J*(Thread + 2 * Border + 1) + I] = d_image[s_global_idy*(width + 2 * Border + 1) + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		int I2 = ii2 % (Thread + 2 * Border + 1);
		int J2 = ii2 / (Thread + 2 * Border + 1);
		s_global_idx = blockIdx.x*blockDim.x + I2;
		s_global_idy = blockIdx.y*blockDim.y + J2;

		if (I2<(Thread + 2 * Border + 1) && J2<Thread && ii2<(Thread + 2 * Border + 1)*(Thread)) {
			sdata[J2*(Thread + 2 * Border + 1) + I2] = d_image[s_global_idy*(width + 2 * Border + 1) + s_global_idx];
		}

		__syncthreads();

		//if (gidx ==width-3 && gidy ==0) {
		//	for (int idi = 0; idi<(Thread+2* Border)*Thread; idi++)
		//		tmpout[idi] = sdata[idi];
		//}


		float tmp_store_1 = 0, tmp_store_2 = 0;
		float img_array;

		for (int i = 0; i <= h_left + h_right; ++i)
		{
			int idx_x = tx + i;
			img_array = sdata[ty*(Thread + 2 * Border + 1) + idx_x];
			tmp_store_1 += img_array*device_L_1[hlen - 1 - i];
			tmp_store_2 += img_array*device_H_1[hlen - 1 - i];
		}
		d_tmp1[gidy*width + gidx] = tmp_store_1;
		d_tmp2[gidy*width + gidx] = tmp_store_2;
	}
}

__global__ void w_kerneld_foward_pass2(float *d_tmp1, float *d_tmp2, float *o_a, float *o_dv, float *o_dh, float *o_dd, int height, int width, float*tmpout) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	if (gidx<width && gidy< height) {
		__shared__ float sdata1[Thread*(Thread + 2 * Border + 1)];
		__shared__ float sdata2[Thread*(Thread + 2 * Border + 1)];
		int ii = tx + ty * Thread;
		int I = ii % (Thread);
		int J = ii / (Thread);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata1[J*Thread + I] = d_tmp1[s_global_idy *width + s_global_idx];
		sdata2[J*Thread + I] = d_tmp2[s_global_idy *width + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		I = ii2 % (Thread);
		J = ii2 / (Thread);
		s_global_idx = blockIdx.x*blockDim.x + I;
		s_global_idy = blockIdx.y*blockDim.y + J;

		if (I<(Thread) && J<(Thread + 2 * Border + 1) && ii2<(Thread + 2 * Border + 1)*(Thread)) {
			sdata1[J*Thread + I] = d_tmp1[s_global_idy *width + s_global_idx];
			sdata2[J*Thread + I] = d_tmp2[s_global_idy *width + s_global_idx];
		}
		__syncthreads();

		//if (gidx == 0 && gidy == 0) {
		//	for (int idi = 0; idi<(Thread + 2 * Border)*Thread; idi++)
		//		tmpout[idi] = sdata1[idi];
		//}

		float tmp_store_1 = 0, tmp_store_2 = 0, tmp_store_3 = 0, tmp_store_4 = 0;
		float img_array_1, img_array_2;

		for (int i = 0; i <= h_left + h_right; ++i)
		{
			int idy = ty + i;
			img_array_1 = sdata1[idy*Thread + tx];
			img_array_2 = sdata2[idy*Thread + tx];
			tmp_store_1 += img_array_1 * device_L_1[hlen - 1 - i];
			tmp_store_2 += img_array_1 * device_H_1[hlen - 1 - i];
			tmp_store_3 += img_array_2 * device_L_1[hlen - 1 - i];
			tmp_store_4 += img_array_2 * device_H_1[hlen - 1 - i];
		}

		o_a[gidy *width + gidx] = tmp_store_1;
		o_dv[gidy*width + gidx] = tmp_store_2;
		o_dh[gidy*width + gidx] = tmp_store_3;
		o_dd[gidy*width + gidx] = tmp_store_4;
	}
}

void w_coeffcients_sum(float **d_coeffs_a, float ** d_coeffs_b, float*d_img_pad_1, int width, int row, int levels, cublasHandle_t handle) {
	// d_coeffs_b =d_coeffs_a+  d_coeffs_b
	// d_coeffs_a =d_coeffs_a-  d_coeffs_b
	float alpha1 = 1;
	float alpha2 = -1;
	float alpha3 = 1 / sqrt(2);
	int length = div2(width)*div2(row);
	float *tmp_1, *tmp_2, *tmp_3;

	tmp_1 = d_img_pad_1;
	tmp_2 = tmp_1 + length;
	tmp_3 = tmp_2 + length;

	for (int i = 0; i<levels; ++i) {
		width = div2(width);
		row = div2(row);
		// store d_coeffs_b in tmp space 
		cudaMemcpy(tmp_1, d_coeffs_b[3 * i + 1], sizeof(float)*width*row, cudaMemcpyDeviceToDevice);
		cudaMemcpy(tmp_2, d_coeffs_b[3 * i + 2], sizeof(float)*width*row, cudaMemcpyDeviceToDevice);
		cudaMemcpy(tmp_3, d_coeffs_b[3 * i + 3], sizeof(float)*width*row, cudaMemcpyDeviceToDevice);
		//y=ax+y
		cublasSaxpy(handle, width*row, &alpha1, d_coeffs_a[3 * i + 1], 1, d_coeffs_b[3 * i + 1], 1);
		cublasSaxpy(handle, width*row, &alpha1, d_coeffs_a[3 * i + 2], 1, d_coeffs_b[3 * i + 2], 1);
		cublasSaxpy(handle, width*row, &alpha1, d_coeffs_a[3 * i + 3], 1, d_coeffs_b[3 * i + 3], 1);

		cublasSscal(handle, width*row, &alpha3, d_coeffs_b[3 * i + 1], 1);
		cublasSscal(handle, width*row, &alpha3, d_coeffs_b[3 * i + 2], 1);
		cublasSscal(handle, width*row, &alpha3, d_coeffs_b[3 * i + 3], 1);

		//y=ax-y
		cublasSaxpy(handle, width*row, &alpha2, tmp_1, 1, d_coeffs_a[3 * i + 1], 1);
		cublasSaxpy(handle, width*row, &alpha2, tmp_2, 1, d_coeffs_a[3 * i + 2], 1);
		cublasSaxpy(handle, width*row, &alpha2, tmp_3, 1, d_coeffs_a[3 * i + 3], 1);

		cublasSscal(handle, width*row, &alpha3, d_coeffs_a[3 * i + 1], 1);
		cublasSscal(handle, width*row, &alpha3, d_coeffs_a[3 * i + 2], 1);
		cublasSscal(handle, width*row, &alpha3, d_coeffs_a[3 * i + 3], 1);
	}
	// last low-low 
	//cudaMemcpy(tmp_1, d_coeffs_b[0], sizeof(float)*width*row, cudaMemcpyDeviceToDevice);
	//cublasSaxpy(handle, width*row, &alpha1, d_coeffs_a[0], 1, d_coeffs_b[0], 1);
	//cublasSaxpy(handle, width*row, &alpha2, tmp_1, 1, d_coeffs_a[0], 1);
	//cublasSscal(handle, width*row, &alpha3, d_coeffs_a[0], 1);
	//cublasSscal(handle, width*row, &alpha3, d_coeffs_b[0], 1);
}

__global__ void w_kernel_shrinkage_single(float *o_a, float thre, int height, int width) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	float val = 0.0f, val_1 = 0.0f;
	if (gidx<width && gidy<height) {
		val = o_a[gidy*width + gidx];
		val_1 = fmaxf((fabsf(val) - thre), 0.0f);
		o_a[gidy*width + gidx] = val_1 / (val_1 + thre)*val;
	}
}

__global__ void w_kernel_shrinkage_multi(float *o_dv, float *o_dh, float *o_dd, float thre, int height, int width) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	float val = 0.0f, val_1 = 0.0f;
	if (gidx<width && gidy<height) {
		val = o_dv[gidy*width + gidx];
		val_1 = fmaxf((fabsf(val) - thre), 0.0f);
		o_dv[gidy*width + gidx] = val_1 / (val_1 + thre)*val;

		val = o_dh[gidy*width + gidx];
		val_1 = fmaxf((fabsf(val) - thre), 0.0f);
		o_dh[gidy*width + gidx] = val_1 / (val_1 + thre)*val;

		val = o_dd[gidy*width + gidx];
		val_1 = fmaxf((fabsf(val) - thre), 0.0f);
		o_dd[gidy*width + gidx] = val_1 / (val_1 + thre)*val;
	}
}

void w_shrinkage(float **d_coeffs_a, float **d_coeffs_b, float thre, int width, int row, int levels) {
	int *tmp_width = new int[levels + 1];
	int *tmp_row = new int[levels + 1];
	tmp_width[0] = width;
	tmp_row[0] = row;
	for (int i = 1; i <= levels; ++i) {
		tmp_width[i] = div2(tmp_width[i - 1]);
		tmp_row[i] = div2(tmp_row[i - 1]);
	}
	int blocksx, blocksy, dwidth, dheight;
	dim3 threadperblock(Thread, Thread);
	for (int i = 1; i <= levels; ++i) {
		dheight = tmp_row[i];
		dwidth = tmp_width[i];
		blocksx = int((dheight - 0.5) / Thread) + 1;
		blocksy = int((dwidth - 0.5) / Thread) + 1;
		dim3 blockpergrid(blocksx, blocksy);
		w_kernel_shrinkage_multi << <blockpergrid, threadperblock >> >(d_coeffs_a[3 * (i - 1) + 1], d_coeffs_a[3 * (i - 1) + 2], d_coeffs_a[3 * (i - 1) + 3], thre, dheight, dwidth);
		w_kernel_shrinkage_multi << <blockpergrid, threadperblock >> >(d_coeffs_b[3 * (i - 1) + 1], d_coeffs_b[3 * (i - 1) + 2], d_coeffs_b[3 * (i - 1) + 3], thre, dheight, dwidth);
	}
	delete[]tmp_width;
	delete[]tmp_row;
}

//inverse wavelet
void w_inverse(float **d_coeffs, float **dd_coeffs, float *d_img, float* d_img_pad_1, float*d_img_pad_2, int width, int height, int levels, cublasHandle_t handle)
{
	int dou_height, dou_width;
	int *tmp_width = new int[levels + 1];
	int *tmp_row = new int[levels + 1];
	tmp_width[0] = width;
	tmp_row[0] = height;
	for (int i = 1; i <= levels; ++i) {
		tmp_width[i] = div2(tmp_width[i - 1]);
		tmp_row[i] = div2(tmp_row[i - 1]);
	}
	// change to multi-stream in next version
	float*d_tmp1, *d_tmp2, *d_tmp3, *d_tmp4, *d_tmp5, *d_tmp6, *tmpout;//temporal cuda space for tree 1
	float *dd_tmp1, *dd_tmp2, *dd_tmp3, *dd_tmp4, *dd_tmp5, *dd_tmp6;//temporal cuda space for tree 2
	int pad_num = Border;
	int threadnum = 32;
	int pad_length = height*(width + 2 * pad_num);
	cudaError_t err = cudaSuccess;
	//err = cudaMalloc(&d_img_pad_1, sizeof(float) * 6 * height*(width + 2 * pad_num));
	//err = cudaMalloc(&d_img_pad_2, sizeof(float) * 6 * height*(width + 2 * pad_num));
	//if (err != cudaSuccess) {
	//	fprintf(stderr, "Failed to allocate (error code %s)!\n", cudaGetErrorString(err));
	//}
	d_tmp6 = d_img_pad_1;
	d_tmp1 = d_tmp6 + pad_length;
	d_tmp2 = d_tmp1 + pad_length;
	d_tmp3 = d_tmp2 + pad_length;
	d_tmp4 = d_tmp3 + pad_length;
	d_tmp5 = d_tmp4 + pad_length;

	dd_tmp6 = d_img_pad_2;
	dd_tmp1 = dd_tmp6 + pad_length;
	dd_tmp2 = dd_tmp1 + pad_length;
	dd_tmp3 = dd_tmp2 + pad_length;
	dd_tmp4 = dd_tmp3 + pad_length;
	dd_tmp5 = dd_tmp4 + pad_length;
	tmpout = d_tmp6;
	dim3 threadperblock(threadnum, threadnum);

	for (int i = levels - 1; i >= 1; --i) {
		width = tmp_width[i + 1];
		height = tmp_row[i + 1];
		dou_width = tmp_width[i];
		dou_height = tmp_row[i];
		dim3 blockpergrid1((int(width - 0.5) / threadnum) + 1, int((height + pad_num + 1 - 0.5) / threadnum) + 1);
		padding_height << <blockpergrid1, threadperblock >> >(d_coeffs[0], d_coeffs[3 * i + 1], d_tmp1, d_tmp2, height, width, pad_num / 2);
		padding_height << <blockpergrid1, threadperblock >> >(d_coeffs[3 * i + 2], d_coeffs[3 * i + 3], d_tmp3, d_tmp4, height, width, pad_num / 2);
		padding_height << <blockpergrid1, threadperblock >> >(dd_coeffs[0], dd_coeffs[3 * i + 1], dd_tmp1, dd_tmp2, height, width, pad_num / 2);
		padding_height << <blockpergrid1, threadperblock >> >(dd_coeffs[3 * i + 2], dd_coeffs[3 * i + 3], dd_tmp3, dd_tmp4, height, width, pad_num / 2);

		dim3 blockpergrid2((int(width - 0.5) / threadnum) + 1, int((height - 0.5) / threadnum) + 1);
		w_kernel_inverse_pass1 << <blockpergrid2, threadperblock >> >(d_tmp1, d_tmp2, d_tmp3, d_tmp4, d_tmp5, d_tmp6, width, height);
		w_kerneld_inverse_pass1 << <blockpergrid2, threadperblock >> >(dd_tmp1, dd_tmp2, dd_tmp3, dd_tmp4, dd_tmp5, dd_tmp6, width, height);

		dim3 blockpergrid3((int(width + pad_num + 1 - 0.5) / threadnum) + 1, int((dou_height - 0.5) / threadnum) + 1);
		padding_width << <blockpergrid3, threadperblock >> > (d_tmp5, d_tmp1, dou_height, width, pad_num / 2);
		padding_width << <blockpergrid3, threadperblock >> >(d_tmp6, d_tmp2, dou_height, width, pad_num / 2);
		padding_width << <blockpergrid3, threadperblock >> > (dd_tmp5, dd_tmp1, dou_height, width, pad_num / 2);
		padding_width << <blockpergrid3, threadperblock >> >(dd_tmp6, dd_tmp2, dou_height, width, pad_num / 2);

		dim3 blockpergrid4((int(width - 0.5) / threadnum) + 1, int((dou_height - 0.5) / threadnum) + 1);
		w_kernel_inverse_pass2 << <blockpergrid4, threadperblock >> >(d_tmp1, d_tmp2, d_coeffs[0], width, dou_height);
		w_kerneld_inverse_pass2 << <blockpergrid4, threadperblock >> >(dd_tmp1, dd_tmp2, dd_coeffs[0], width, dou_height);
	}
	width = tmp_width[1];
	height = tmp_row[1];
	dou_width = tmp_width[0];
	dou_height = tmp_row[0];
	dim3 blockpergrid1((int(width - 0.5) / threadnum) + 1, int((height + pad_num + 1 - 0.5) / threadnum) + 1);

	padding_height << <blockpergrid1, threadperblock >> >(d_coeffs[0], d_coeffs[1], d_tmp1, d_tmp2, height, width, pad_num / 2);
	padding_height << <blockpergrid1, threadperblock >> >(d_coeffs[2], d_coeffs[3], d_tmp3, d_tmp4, height, width, pad_num / 2);
	padding_height << <blockpergrid1, threadperblock >> >(dd_coeffs[0], dd_coeffs[1], dd_tmp1, dd_tmp2, height, width, pad_num / 2);
	padding_height << <blockpergrid1, threadperblock >> >(dd_coeffs[2], dd_coeffs[3], dd_tmp3, dd_tmp4, height, width, pad_num / 2);

	dim3 blockpergrid2((int(width - 0.5) / threadnum) + 1, int((height - 0.5) / threadnum) + 1);
	w_kernel_inverse_pass1_1st << <blockpergrid2, threadperblock >> >(d_tmp1, d_tmp2, d_tmp3, d_tmp4, d_tmp5, d_tmp6, width, height);
	w_kerneld_inverse_pass1_1st << <blockpergrid2, threadperblock >> >(dd_tmp1, dd_tmp2, dd_tmp3, dd_tmp4, dd_tmp5, dd_tmp6, width, height);

	dim3 blockpergrid3((int(width + pad_num + 1 - 0.5) / threadnum) + 1, int((dou_height - 0.5) / threadnum) + 1);
	padding_width << <blockpergrid3, threadperblock >> > (d_tmp5, d_tmp1, dou_height, width, pad_num / 2);
	padding_width << <blockpergrid3, threadperblock >> >(d_tmp6, d_tmp2, dou_height, width, pad_num / 2);
	padding_width << <blockpergrid3, threadperblock >> > (dd_tmp5, dd_tmp1, dou_height, width, pad_num / 2);
	padding_width << <blockpergrid3, threadperblock >> >(dd_tmp6, dd_tmp2, dou_height, width, pad_num / 2);

	dim3 blockpergrid4((int(width - 0.5) / threadnum) + 1, int((dou_height - 0.5) / threadnum) + 1);
	w_kernel_inverse_pass2_1st << <blockpergrid4, threadperblock >> >(d_tmp1, d_tmp2, d_tmp3, width, dou_height);
	w_kerneld_inverse_pass2_1st << <blockpergrid4, threadperblock >> >(dd_tmp1, dd_tmp2, d_img, width, dou_height);

	float alpha1 = 1;
	float alpha2 = 1 / sqrt(2);
	cublasSaxpy(handle, dou_width*dou_height, &alpha1, d_tmp3, 1, d_img, 1);
	cublasSscal(handle, dou_width*dou_height, &alpha1, d_img, 1);
	delete[]tmp_width;
	delete[]tmp_row;
}

// tree 1 inverse 

// tree 1 inverse 

__global__ void w_kernel_inverse_pass1_1st(float *o_a, float *o_dv, float *o_dh, float *o_dd, float *d_tmp1, float *d_tmp2, int width, int height) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	if (gidy<height && gidx<width) {

		__shared__ float sdata_o_a[Thread*(Thread + Border + 1)];
		__shared__ float sdata_o_dv[Thread*(Thread + Border + 1)];
		__shared__ float sdata_o_dh[Thread*(Thread + Border + 1)];
		__shared__ float sdata_o_dd[Thread*(Thread + Border + 1)];

		int ii = tx + ty * Thread;
		int I = ii % (Thread);
		int J = ii / (Thread);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata_o_a[J*Thread + I] = o_a[s_global_idy *width + s_global_idx];
		sdata_o_dv[J*Thread + I] = o_dv[s_global_idy *width + s_global_idx];
		sdata_o_dh[J*Thread + I] = o_dh[s_global_idy *width + s_global_idx];
		sdata_o_dd[J*Thread + I] = o_dd[s_global_idy *width + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		I = ii2 % (Thread);
		J = ii2 / (Thread);
		s_global_idx = blockIdx.x*blockDim.x + I;
		s_global_idy = blockIdx.y*blockDim.y + J;

		if (I<(Thread) && J<(Thread + Border + 1) && ii2<(Thread + Border + 1)*(Thread)) {
			sdata_o_a[J*Thread + I] = o_a[s_global_idy *width + s_global_idx];
			sdata_o_dv[J*Thread + I] = o_dv[s_global_idy *width + s_global_idx];
			sdata_o_dh[J*Thread + I] = o_dh[s_global_idy *width + s_global_idx];
			sdata_o_dd[J*Thread + I] = o_dd[s_global_idy *width + s_global_idx];
		}
		__syncthreads();

		float tmp_store_1 = 0, tmp_store_2 = 0, tmp_store_3 = 0, tmp_store_4 = 0;
		float otmp_store_1 = 0, otmp_store_2 = 0, otmp_store_3 = 0, otmp_store_4 = 0;
		int off = 1 - (gidy & 1);
		for (int i = 0; i <= h_left; ++i) {
			int idy = ty + i;
			tmp_store_1 += sdata_o_a[idy*Thread + tx] * fdevice_IL[hlen - 1 - i * 2 - 1];
			tmp_store_2 += sdata_o_dv[idy*Thread + tx] * fdevice_IH[hlen - 1 - i * 2 - 1];
			tmp_store_3 += sdata_o_dh[idy*Thread + tx] * fdevice_IL[hlen - 1 - i * 2 - 1];
			tmp_store_4 += sdata_o_dd[idy*Thread + tx] * fdevice_IH[hlen - 1 - i * 2 - 1];

			otmp_store_1 += sdata_o_a[idy*Thread + tx] * fdevice_IL[hlen - 1 - i * 2];
			otmp_store_2 += sdata_o_dv[idy*Thread + tx] * fdevice_IH[hlen - 1 - i * 2];
			otmp_store_3 += sdata_o_dh[idy*Thread + tx] * fdevice_IL[hlen - 1 - i * 2];
			otmp_store_4 += sdata_o_dd[idy*Thread + tx] * fdevice_IH[hlen - 1 - i * 2];

		}

		d_tmp1[2 * gidy*width + gidx] = tmp_store_1 + tmp_store_2;//L
		d_tmp2[2 * gidy*width + gidx] = tmp_store_3 + tmp_store_4;//H

		d_tmp1[(2 * gidy + 1)*width + gidx] = otmp_store_1 + otmp_store_2;//L
		d_tmp2[(2 * gidy + 1)*width + gidx] = otmp_store_3 + otmp_store_4;//H
	}
}

__global__ void w_kernel_inverse_pass2_1st(float* d_tmp1, float* d_tmp2, float* img, int width, int height) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	if (gidy < height && gidx < width) {
		__shared__ float sdata1[Thread*(Thread + Border + 1)];
		__shared__ float sdata2[Thread*(Thread + Border + 1)];

		int ii = tx + ty * Thread;
		int I = ii % (Thread + Border + 1);
		int J = ii / (Thread + Border + 1);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata1[J*(Thread + Border + 1) + I] = d_tmp1[s_global_idy*(width + Border + 1) + s_global_idx];
		sdata2[J*(Thread + Border + 1) + I] = d_tmp2[s_global_idy*(width + Border + 1) + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		int I2 = ii2 % (Thread + Border + 1);
		int J2 = ii2 / (Thread + Border + 1);
		s_global_idx = blockIdx.x*blockDim.x + I2;
		s_global_idy = blockIdx.y*blockDim.y + J2;

		if (I2<(Thread + Border + 1) && J2<Thread && ii2<(Thread + Border + 1)*(Thread)) {
			sdata1[J2*(Thread + Border + 1) + I2] = d_tmp1[s_global_idy*(width + Border + 1) + s_global_idx];
			sdata2[J2*(Thread + Border + 1) + I2] = d_tmp2[s_global_idy*(width + Border + 1) + s_global_idx];
		}

		__syncthreads();

		float tmp_store_1 = 0, tmp_store_2 = 0;
		float otmp_store_1 = 0, otmp_store_2 = 0;

		for (int i = 0; i <= h_left; ++i) {
			int idx_x = tx + i;
			tmp_store_1 += sdata1[ty*(Thread + Border + 1) + idx_x] * fdevice_IL[hlen - 1 - 2 * i - 1];
			tmp_store_2 += sdata2[ty*(Thread + Border + 1) + idx_x] * fdevice_IH[hlen - 1 - 2 * i - 1];

			otmp_store_1 += sdata1[ty*(Thread + Border + 1) + idx_x] * fdevice_IL[hlen - 1 - 2 * i];
			otmp_store_2 += sdata2[ty*(Thread + Border + 1) + idx_x] * fdevice_IH[hlen - 1 - 2 * i];

		}
		img[gidy*width * 2 + gidx * 2] = tmp_store_1 + tmp_store_2;
		img[gidy*width * 2 + gidx * 2 + 1] = otmp_store_1 + otmp_store_2;
	}
}

__global__ void w_kernel_inverse_pass1(float *o_a, float *o_dv, float *o_dh, float *o_dd, float *d_tmp1, float *d_tmp2, int width, int height) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	if (gidy<height && gidx<width) {

		__shared__ float sdata_o_a[Thread*(Thread + Border + 1)];
		__shared__ float sdata_o_dv[Thread*(Thread + Border + 1)];
		__shared__ float sdata_o_dh[Thread*(Thread + Border + 1)];
		__shared__ float sdata_o_dd[Thread*(Thread + Border + 1)];

		int ii = tx + ty * Thread;
		int I = ii % (Thread);
		int J = ii / (Thread);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata_o_a[J*Thread + I] = o_a[s_global_idy *width + s_global_idx];
		sdata_o_dv[J*Thread + I] = o_dv[s_global_idy *width + s_global_idx];
		sdata_o_dh[J*Thread + I] = o_dh[s_global_idy *width + s_global_idx];
		sdata_o_dd[J*Thread + I] = o_dd[s_global_idy *width + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		I = ii2 % (Thread);
		J = ii2 / (Thread);
		s_global_idx = blockIdx.x*blockDim.x + I;
		s_global_idy = blockIdx.y*blockDim.y + J;

		if (I<(Thread) && J<(Thread + Border + 1) && ii2<(Thread + Border + 1)*(Thread)) {
			sdata_o_a[J*Thread + I] = o_a[s_global_idy *width + s_global_idx];
			sdata_o_dv[J*Thread + I] = o_dv[s_global_idy *width + s_global_idx];
			sdata_o_dh[J*Thread + I] = o_dh[s_global_idy *width + s_global_idx];
			sdata_o_dd[J*Thread + I] = o_dd[s_global_idy *width + s_global_idx];
		}
		__syncthreads();

		float tmp_store_1 = 0, tmp_store_2 = 0, tmp_store_3 = 0, tmp_store_4 = 0;
		float otmp_store_1 = 0, otmp_store_2 = 0, otmp_store_3 = 0, otmp_store_4 = 0;
		int off = 1 - (gidy & 1);
		for (int i = 0; i <= h_left; ++i) {
			int idy = ty + i;
			tmp_store_1 += sdata_o_a[idy*Thread + tx] * device_IL[hlen - 1 - i * 2 - 1];
			tmp_store_2 += sdata_o_dv[idy*Thread + tx] * device_IH[hlen - 1 - i * 2 - 1];
			tmp_store_3 += sdata_o_dh[idy*Thread + tx] * device_IL[hlen - 1 - i * 2 - 1];
			tmp_store_4 += sdata_o_dd[idy*Thread + tx] * device_IH[hlen - 1 - i * 2 - 1];

			otmp_store_1 += sdata_o_a[idy*Thread + tx] * device_IL[hlen - 1 - i * 2];
			otmp_store_2 += sdata_o_dv[idy*Thread + tx] * device_IH[hlen - 1 - i * 2];
			otmp_store_3 += sdata_o_dh[idy*Thread + tx] * device_IL[hlen - 1 - i * 2];
			otmp_store_4 += sdata_o_dd[idy*Thread + tx] * device_IH[hlen - 1 - i * 2];

		}

		d_tmp1[2 * gidy*width + gidx] = tmp_store_1 + tmp_store_2;//L
		d_tmp2[2 * gidy*width + gidx] = tmp_store_3 + tmp_store_4;//H

		d_tmp1[(2 * gidy + 1)*width + gidx] = otmp_store_1 + otmp_store_2;//L
		d_tmp2[(2 * gidy + 1)*width + gidx] = otmp_store_3 + otmp_store_4;//H
	}
}

__global__ void w_kernel_inverse_pass2(float* d_tmp1, float* d_tmp2, float* img, int width, int height) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	if (gidy < height && gidx < width) {
		__shared__ float sdata1[Thread*(Thread + Border + 1)];
		__shared__ float sdata2[Thread*(Thread + Border + 1)];

		int ii = tx + ty * Thread;
		int I = ii % (Thread + Border + 1);
		int J = ii / (Thread + Border + 1);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata1[J*(Thread + Border + 1) + I] = d_tmp1[s_global_idy*(width + Border + 1) + s_global_idx];
		sdata2[J*(Thread + Border + 1) + I] = d_tmp2[s_global_idy*(width + Border + 1) + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		int I2 = ii2 % (Thread + Border + 1);
		int J2 = ii2 / (Thread + Border + 1);
		s_global_idx = blockIdx.x*blockDim.x + I2;
		s_global_idy = blockIdx.y*blockDim.y + J2;

		if (I2<(Thread + Border + 1) && J2<Thread && ii2<(Thread + Border + 1)*(Thread)) {
			sdata1[J2*(Thread + Border + 1) + I2] = d_tmp1[s_global_idy*(width + Border + 1) + s_global_idx];
			sdata2[J2*(Thread + Border + 1) + I2] = d_tmp2[s_global_idy*(width + Border + 1) + s_global_idx];
		}

		__syncthreads();

		float tmp_store_1 = 0, tmp_store_2 = 0;
		float otmp_store_1 = 0, otmp_store_2 = 0;

		for (int i = 0; i <= h_left; ++i) {
			int idx_x = tx + i;
			tmp_store_1 += sdata1[ty*(Thread + Border + 1) + idx_x] * device_IL[hlen - 1 - 2 * i - 1];
			tmp_store_2 += sdata2[ty*(Thread + Border + 1) + idx_x] * device_IH[hlen - 1 - 2 * i - 1];

			otmp_store_1 += sdata1[ty*(Thread + Border + 1) + idx_x] * device_IL[hlen - 1 - 2 * i];
			otmp_store_2 += sdata2[ty*(Thread + Border + 1) + idx_x] * device_IH[hlen - 1 - 2 * i];

		}
		img[gidy*width * 2 + gidx * 2] = tmp_store_1 + tmp_store_2;
		img[gidy*width * 2 + gidx * 2 + 1] = otmp_store_1 + otmp_store_2;
	}
}

// tree 2 inverse 
__global__ void w_kerneld_inverse_pass1_1st(float *o_a, float *o_dv, float *o_dh, float *o_dd, float *d_tmp1, float *d_tmp2, int width, int height) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	if (gidy<height && gidx<width) {

		__shared__ float sdata_o_a[Thread*(Thread + Border + 1)];
		__shared__ float sdata_o_dv[Thread*(Thread + Border + 1)];
		__shared__ float sdata_o_dh[Thread*(Thread + Border + 1)];
		__shared__ float sdata_o_dd[Thread*(Thread + Border + 1)];

		int ii = tx + ty * Thread;
		int I = ii % (Thread);
		int J = ii / (Thread);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata_o_a[J*Thread + I] = o_a[s_global_idy *width + s_global_idx];
		sdata_o_dv[J*Thread + I] = o_dv[s_global_idy *width + s_global_idx];
		sdata_o_dh[J*Thread + I] = o_dh[s_global_idy *width + s_global_idx];
		sdata_o_dd[J*Thread + I] = o_dd[s_global_idy *width + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		I = ii2 % (Thread);
		J = ii2 / (Thread);
		s_global_idx = blockIdx.x*blockDim.x + I;
		s_global_idy = blockIdx.y*blockDim.y + J;

		if (I<(Thread) && J<(Thread + Border + 1) && ii2<(Thread + Border + 1)*(Thread)) {
			sdata_o_a[J*Thread + I] = o_a[s_global_idy *width + s_global_idx];
			sdata_o_dv[J*Thread + I] = o_dv[s_global_idy *width + s_global_idx];
			sdata_o_dh[J*Thread + I] = o_dh[s_global_idy *width + s_global_idx];
			sdata_o_dd[J*Thread + I] = o_dd[s_global_idy *width + s_global_idx];
		}
		__syncthreads();

		float tmp_store_1 = 0, tmp_store_2 = 0, tmp_store_3 = 0, tmp_store_4 = 0;
		float otmp_store_1 = 0, otmp_store_2 = 0, otmp_store_3 = 0, otmp_store_4 = 0;
		int off = 1 - (gidy & 1);
		for (int i = 0; i <= h_left; ++i) {
			int idy = ty + i;
			tmp_store_1 += sdata_o_a[idy*Thread + tx] * fdevice_IL_1[hlen - 1 - i * 2 - 1];
			tmp_store_2 += sdata_o_dv[idy*Thread + tx] * fdevice_IH_1[hlen - 1 - i * 2 - 1];
			tmp_store_3 += sdata_o_dh[idy*Thread + tx] * fdevice_IL_1[hlen - 1 - i * 2 - 1];
			tmp_store_4 += sdata_o_dd[idy*Thread + tx] * fdevice_IH_1[hlen - 1 - i * 2 - 1];

			otmp_store_1 += sdata_o_a[idy*Thread + tx] * fdevice_IL_1[hlen - 1 - i * 2];
			otmp_store_2 += sdata_o_dv[idy*Thread + tx] * fdevice_IH_1[hlen - 1 - i * 2];
			otmp_store_3 += sdata_o_dh[idy*Thread + tx] * fdevice_IL_1[hlen - 1 - i * 2];
			otmp_store_4 += sdata_o_dd[idy*Thread + tx] * fdevice_IH_1[hlen - 1 - i * 2];

		}

		d_tmp1[2 * gidy*width + gidx] = tmp_store_1 + tmp_store_2;//L
		d_tmp2[2 * gidy*width + gidx] = tmp_store_3 + tmp_store_4;//H

		d_tmp1[(2 * gidy + 1)*width + gidx] = otmp_store_1 + otmp_store_2;//L
		d_tmp2[(2 * gidy + 1)*width + gidx] = otmp_store_3 + otmp_store_4;//H
	}
}

__global__ void w_kerneld_inverse_pass2_1st(float* d_tmp1, float* d_tmp2, float* img, int width, int height) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	if (gidy < height && gidx < width) {
		__shared__ float sdata1[Thread*(Thread + Border + 1)];
		__shared__ float sdata2[Thread*(Thread + Border + 1)];

		int ii = tx + ty * Thread;
		int I = ii % (Thread + Border + 1);
		int J = ii / (Thread + Border + 1);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata1[J*(Thread + Border + 1) + I] = d_tmp1[s_global_idy*(width + Border + 1) + s_global_idx];
		sdata2[J*(Thread + Border + 1) + I] = d_tmp2[s_global_idy*(width + Border + 1) + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		int I2 = ii2 % (Thread + Border + 1);
		int J2 = ii2 / (Thread + Border + 1);
		s_global_idx = blockIdx.x*blockDim.x + I2;
		s_global_idy = blockIdx.y*blockDim.y + J2;

		if (I2<(Thread + Border + 1) && J2<Thread && ii2<(Thread + Border + 1)*(Thread)) {
			sdata1[J2*(Thread + Border + 1) + I2] = d_tmp1[s_global_idy*(width + Border + 1) + s_global_idx];
			sdata2[J2*(Thread + Border + 1) + I2] = d_tmp2[s_global_idy*(width + Border + 1) + s_global_idx];
		}

		__syncthreads();

		float tmp_store_1 = 0, tmp_store_2 = 0;
		float otmp_store_1 = 0, otmp_store_2 = 0;

		for (int i = 0; i <= h_left; ++i) {
			int idx_x = tx + i;
			tmp_store_1 += sdata1[ty*(Thread + Border + 1) + idx_x] * fdevice_IL_1[hlen - 1 - 2 * i - 1];
			tmp_store_2 += sdata2[ty*(Thread + Border + 1) + idx_x] * fdevice_IH_1[hlen - 1 - 2 * i - 1];

			otmp_store_1 += sdata1[ty*(Thread + Border + 1) + idx_x] * fdevice_IL_1[hlen - 1 - 2 * i];
			otmp_store_2 += sdata2[ty*(Thread + Border + 1) + idx_x] * fdevice_IH_1[hlen - 1 - 2 * i];

		}
		img[gidy*width * 2 + gidx * 2] = tmp_store_1 + tmp_store_2;
		img[gidy*width * 2 + gidx * 2 + 1] = otmp_store_1 + otmp_store_2;
	}
}

__global__ void w_kerneld_inverse_pass1(float *o_a, float *o_dv, float *o_dh, float *o_dd, float *d_tmp1, float *d_tmp2, int width, int height) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	if (gidy<height && gidx<width) {

		__shared__ float sdata_o_a[Thread*(Thread + Border + 1)];
		__shared__ float sdata_o_dv[Thread*(Thread + Border + 1)];
		__shared__ float sdata_o_dh[Thread*(Thread + Border + 1)];
		__shared__ float sdata_o_dd[Thread*(Thread + Border + 1)];

		int ii = tx + ty * Thread;
		int I = ii % (Thread);
		int J = ii / (Thread);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata_o_a[J*Thread + I] = o_a[s_global_idy *width + s_global_idx];
		sdata_o_dv[J*Thread + I] = o_dv[s_global_idy *width + s_global_idx];
		sdata_o_dh[J*Thread + I] = o_dh[s_global_idy *width + s_global_idx];
		sdata_o_dd[J*Thread + I] = o_dd[s_global_idy *width + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		I = ii2 % (Thread);
		J = ii2 / (Thread);
		s_global_idx = blockIdx.x*blockDim.x + I;
		s_global_idy = blockIdx.y*blockDim.y + J;

		if (I<(Thread) && J<(Thread + Border + 1) && ii2<(Thread + Border + 1)*(Thread)) {
			sdata_o_a[J*Thread + I] = o_a[s_global_idy *width + s_global_idx];
			sdata_o_dv[J*Thread + I] = o_dv[s_global_idy *width + s_global_idx];
			sdata_o_dh[J*Thread + I] = o_dh[s_global_idy *width + s_global_idx];
			sdata_o_dd[J*Thread + I] = o_dd[s_global_idy *width + s_global_idx];
		}
		__syncthreads();

		float tmp_store_1 = 0, tmp_store_2 = 0, tmp_store_3 = 0, tmp_store_4 = 0;
		float otmp_store_1 = 0, otmp_store_2 = 0, otmp_store_3 = 0, otmp_store_4 = 0;
		int off = 1 - (gidy & 1);
		for (int i = 0; i <= h_left; ++i) {
			int idy = ty + i;
			tmp_store_1 += sdata_o_a[idy*Thread + tx] * device_IL_1[hlen - 1 - i * 2 - 1];
			tmp_store_2 += sdata_o_dv[idy*Thread + tx] * device_IH_1[hlen - 1 - i * 2 - 1];
			tmp_store_3 += sdata_o_dh[idy*Thread + tx] * device_IL_1[hlen - 1 - i * 2 - 1];
			tmp_store_4 += sdata_o_dd[idy*Thread + tx] * device_IH_1[hlen - 1 - i * 2 - 1];

			otmp_store_1 += sdata_o_a[idy*Thread + tx] * device_IL_1[hlen - 1 - i * 2];
			otmp_store_2 += sdata_o_dv[idy*Thread + tx] * device_IH_1[hlen - 1 - i * 2];
			otmp_store_3 += sdata_o_dh[idy*Thread + tx] * device_IL_1[hlen - 1 - i * 2];
			otmp_store_4 += sdata_o_dd[idy*Thread + tx] * device_IH_1[hlen - 1 - i * 2];

		}

		d_tmp1[2 * gidy*width + gidx] = tmp_store_1 + tmp_store_2;//L
		d_tmp2[2 * gidy*width + gidx] = tmp_store_3 + tmp_store_4;//H

		d_tmp1[(2 * gidy + 1)*width + gidx] = otmp_store_1 + otmp_store_2;//L
		d_tmp2[(2 * gidy + 1)*width + gidx] = otmp_store_3 + otmp_store_4;//H
	}
}

__global__ void w_kerneld_inverse_pass2(float* d_tmp1, float* d_tmp2, float* img, int width, int height) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	if (gidy < height && gidx < width) {
		__shared__ float sdata1[Thread*(Thread + Border + 1)];
		__shared__ float sdata2[Thread*(Thread + Border + 1)];

		int ii = tx + ty * Thread;
		int I = ii % (Thread + Border + 1);
		int J = ii / (Thread + Border + 1);
		int s_global_idx = blockIdx.x*blockDim.x + I;
		int s_global_idy = blockIdx.y*blockDim.y + J;
		sdata1[J*(Thread + Border + 1) + I] = d_tmp1[s_global_idy*(width + Border + 1) + s_global_idx];
		sdata2[J*(Thread + Border + 1) + I] = d_tmp2[s_global_idy*(width + Border + 1) + s_global_idx];

		__syncthreads();

		int ii2 = tx + ty * Thread + Thread*Thread;
		int I2 = ii2 % (Thread + Border + 1);
		int J2 = ii2 / (Thread + Border + 1);
		s_global_idx = blockIdx.x*blockDim.x + I2;
		s_global_idy = blockIdx.y*blockDim.y + J2;

		if (I2<(Thread + Border + 1) && J2<Thread && ii2<(Thread + Border + 1)*(Thread)) {
			sdata1[J2*(Thread + Border + 1) + I2] = d_tmp1[s_global_idy*(width + Border + 1) + s_global_idx];
			sdata2[J2*(Thread + Border + 1) + I2] = d_tmp2[s_global_idy*(width + Border + 1) + s_global_idx];
		}

		__syncthreads();

		float tmp_store_1 = 0, tmp_store_2 = 0;
		float otmp_store_1 = 0, otmp_store_2 = 0;

		for (int i = 0; i <= h_left; ++i) {
			int idx_x = tx + i;
			tmp_store_1 += sdata1[ty*(Thread + Border + 1) + idx_x] * device_IL_1[hlen - 1 - 2 * i - 1];
			tmp_store_2 += sdata2[ty*(Thread + Border + 1) + idx_x] * device_IH_1[hlen - 1 - 2 * i - 1];

			otmp_store_1 += sdata1[ty*(Thread + Border + 1) + idx_x] * device_IL_1[hlen - 1 - 2 * i];
			otmp_store_2 += sdata2[ty*(Thread + Border + 1) + idx_x] * device_IH_1[hlen - 1 - 2 * i];

		}
		img[gidy*width * 2 + gidx * 2] = tmp_store_1 + tmp_store_2;
		img[gidy*width * 2 + gidx * 2 + 1] = otmp_store_1 + otmp_store_2;
	}
}

__global__ void w_kern_circshift(float* d_image, float* d_out, int Nr, int Nc, int sr, int sc) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
	if (gidx < Nc && gidy < Nr) {
		int r = gidy - sr, c = gidx - sc;
		if (r < 0) r += Nr;
		if (c < 0) c += Nc;
		d_out[gidy*Nc + gidx] = d_image[r*Nc + c];
	}
}

void w_image_normalization(float *d_img, int width, int row, cublasHandle_t handle) {
	float al = 1 / sqrt(2);
	cublasSscal(handle, width*row, &al, d_img, 1);
}

void w_coeff_normalization(float **d_coeffs, int width, int row, int levels, cublasHandle_t handle) {
	int Nr = row, Nc = width;
	float al = 1 / sqrt(2);
	for (int i = 0; i<levels; ++i) {
		width = div2(width);
		row = div2(row);
		cublasSscal(handle, width*row, &al, d_coeffs[3 * i + 1], 1);
		cublasSscal(handle, width*row, &al, d_coeffs[3 * i + 2], 1);
		cublasSscal(handle, width*row, &al, d_coeffs[3 * i + 3], 1);
	}
	cublasSscal(handle, width*row, &al, d_coeffs[0], 1);
}
//__global__ void padding_width(float* ary, float*output_ary, int height, int width, int ori_width) {
//	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
//	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
//	if (gidx < (width + 2 * Border) && gidy < height) {
//		if (gidx<Border) {
//			output_ary[gidy*(width + 2 * Border) + gidx] = ary[gidy*ori_width + 2 * Border - 2 - 2 * gidx];
//		}
//		else if (gidx >= (width + Border)) {
//			output_ary[gidy*(width + 2 * Border) + gidx] = ary[gidy*ori_width + 2 * ori_width - 2 * gidx + 2 * Border - 2];
//		}
//		else {
//			output_ary[gidy*(width + 2 * Border) + gidx] = ary[gidy*ori_width + (gidx - Border) * 2];
//		}
//	}
//}
//
//__global__ void padding_height(float* ary, float*output_ary, int height, int width, int ori_height) {
//	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
//	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
//	if (gidx < width && gidy < (height + 2 * Border)) {
//		if (gidy<Border) {
//			output_ary[gidy*width + gidx] = ary[(2 * Border - 2 * gidy - 2)*width + gidx];
//		}
//		else if (gidy >= (height + Border)) {
//			output_ary[gidy*width + gidx] = ary[(2 * ori_height - 2 * gidy + 2 * Border - 2)*width + gidx];
//		}
//		else {
//			output_ary[gidy *width + gidx] = ary[2 * (gidy - Border)*width + gidx];
//		}
//	}
//}
//
//
//// input img d_img horizontally convolved with L and H filter. The result store in d_tmp1 and d_tmp2
//__global__ void w_kernel_foward_pass1(float *d_image, float *d_tmp1, float *d_tmp2, int height, int width, float*tmpout) {
//	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
//	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
//	int tx = threadIdx.x;
//	int ty = threadIdx.y;
//
//
//	if (gidx<width && gidy<height) {
//		__shared__ float sdata[Thread*(Thread + 2 * Border + 1)];
//		int ii = tx + ty * Thread;
//		int I = ii % (Thread + 2 * Border + 1);
//		int J = ii / (Thread + 2 * Border + 1);
//		int s_global_idx = blockIdx.x*blockDim.x + I;
//		int s_global_idy = blockIdx.y*blockDim.y + J;
//		sdata[J*(Thread + 2 * Border + 1) + I] = d_image[s_global_idy*(width + 2 * Border + 1) + s_global_idx];
//
//		__syncthreads();
//
//		int ii2 = tx + ty * Thread + Thread*Thread;
//		int I2 = ii2 % (Thread + 2 * Border + 1);
//		int J2 = ii2 / (Thread + 2 * Border + 1);
//		s_global_idx = blockIdx.x*blockDim.x + I2;
//		s_global_idy = blockIdx.y*blockDim.y + J2;
//
//		if (I2<(Thread + 2 * Border + 1) && J2<Thread && ii2<(Thread + 2 * Border + 1)*(Thread)) {
//			sdata[J2*(Thread + 2 * Border + 1) + I2] = d_image[s_global_idy*(width + 2 * Border + 1) + s_global_idx];
//		}
//
//		__syncthreads();
//
//		//if (gidx ==width-3 && gidy ==0) {
//		//	for (int idi = 0; idi<(Thread+2* Border)*Thread; idi++)
//		//		tmpout[idi] = sdata[idi];
//		//}
//
//
//		float tmp_store_1 = 0, tmp_store_2 = 0;
//		float img_array;
//
//		for (int i = 0; i <= h_left + h_right; ++i)
//		{
//			int idx_x = tx + i;
//			img_array = sdata[ty*(Thread + 2 * Border + 1) + idx_x];
//			tmp_store_1 += img_array*device_L[hlen - 1 - i];
//			tmp_store_2 += img_array*device_H[hlen - 1 - i];
//		}
//		d_tmp1[gidy*width + gidx] = tmp_store_1;
//		d_tmp2[gidy*width + gidx] = tmp_store_2;
//	}
//}
//
//
//// after pass1, d_tmp1 and d_tmp2 vertically convolved with L and H filter, resulting LL o_a, HL o_dv, LH o_dh, HH o_dd
//__global__ void w_kernel_foward_pass2(float *d_tmp1, float *d_tmp2, float *o_a, float *o_dv, float *o_dh, float *o_dd, int height, int width, float*tmpout) {
//	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
//	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
//	int tx = threadIdx.x;
//	int ty = threadIdx.y;
//
//	if (gidx<width && gidy< height) {
//		__shared__ float sdata1[Thread*(Thread + 2 * Border + 1)];
//		__shared__ float sdata2[Thread*(Thread + 2 * Border + 1)];
//		int ii = tx + ty * Thread;
//		int I = ii % (Thread);
//		int J = ii / (Thread);
//		int s_global_idx = blockIdx.x*blockDim.x + I;
//		int s_global_idy = blockIdx.y*blockDim.y + J;
//		sdata1[J*Thread + I] = d_tmp1[s_global_idy *width + s_global_idx];
//		sdata2[J*Thread + I] = d_tmp2[s_global_idy *width + s_global_idx];
//
//		__syncthreads();
//
//		int ii2 = tx + ty * Thread + Thread*Thread;
//		I = ii2 % (Thread);
//		J = ii2 / (Thread);
//		s_global_idx = blockIdx.x*blockDim.x + I;
//		s_global_idy = blockIdx.y*blockDim.y + J;
//
//		if (I<(Thread) && J<(Thread + 2 * Border + 1) && ii2<(Thread + 2 * Border + 1)*(Thread)) {
//			sdata1[J*Thread + I] = d_tmp1[s_global_idy *width + s_global_idx];
//			sdata2[J*Thread + I] = d_tmp2[s_global_idy *width + s_global_idx];
//		}
//		__syncthreads();
//
//		//if (gidx == 0 && gidy == 0) {
//		//	for (int idi = 0; idi<(Thread + 2 * Border)*Thread; idi++)
//		//		tmpout[idi] = sdata1[idi];
//		//}
//
//		float tmp_store_1 = 0, tmp_store_2 = 0, tmp_store_3 = 0, tmp_store_4 = 0;
//		float img_array_1, img_array_2;
//
//		for (int i = 0; i <= h_left + h_right; ++i)
//		{
//			int idy = ty + i;
//			img_array_1 = sdata1[idy*Thread + tx];
//			img_array_2 = sdata2[idy*Thread + tx];
//			tmp_store_1 += img_array_1 * device_L[hlen - 1 - i];
//			tmp_store_2 += img_array_1 * device_H[hlen - 1 - i];
//			tmp_store_3 += img_array_2 * device_L[hlen - 1 - i];
//			tmp_store_4 += img_array_2 * device_H[hlen - 1 - i];
//		}
//
//		o_a[gidy *width + gidx] = tmp_store_1;
//		o_dv[gidy*width + gidx] = tmp_store_2;
//		o_dh[gidy*width + gidx] = tmp_store_3;
//		o_dd[gidy*width + gidx] = tmp_store_4;
//	}
//}
//
//__global__ void w_inverse_pass1(float *o_a, float *o_dv, float *o_dh, float *o_dd, float *d_tmp1, float *d_tmp2, int Nr, int Nc, int Nr2, int hlen) {
//	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
//	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
//	//int Nr2 = Nr * 2;
//	if (gidy<Nr2 && gidx<Nc) {
//		int h_left, h_right;
//		int hlen2 = hlen / 2;
//		h_left = hlen2 / 2;
//		h_right = hlen2 / 2 - 1;
//		gidy += 1;
//
//		int jy1 = h_left - gidy / 2;
//		int jy2 = Nr - 1 - gidy / 2 + h_left;
//		int offset = 1 - (gidy & 1);
//		float tmp_store_1 = 0, tmp_store_2 = 0, tmp_store_3 = 0, tmp_store_4 = 0;
//
//		for (int i = 0; i <= h_left + h_right; ++i) {
//			int  idy = gidy / 2 - h_left + i;
//			if (i<jy1) idy += Nr;
//			if (i>jy2) idy -= Nr;
//
//			tmp_store_1 += o_a[idy*Nc + gidx] * device_IL[hlen - 1 - (2 * i + offset)];
//			tmp_store_2 += o_dv[idy*Nc + gidx] * device_IL[hlen - 1 - (2 * i + offset)];
//			tmp_store_3 += o_dh[idy*Nc + gidx] * device_IH[hlen - 1 - (2 * i + offset)];
//			tmp_store_4 += o_dd[idy*Nc + gidx] * device_IH[hlen - 1 - (2 * i + offset)];
//
//		}
//		if (hlen2 & 1) {
//			d_tmp1[gidy*Nc + gidx] = tmp_store_1 + tmp_store_3;//L
//			d_tmp2[gidy*Nc + gidx] = tmp_store_2 + tmp_store_4;//H
//		}
//		else {
//			d_tmp1[(gidy - 1)*Nc + gidx] = tmp_store_1 + tmp_store_3;
//			d_tmp2[(gidy - 1)*Nc + gidx] = tmp_store_2 + tmp_store_4;
//		}
//	}
//}
//
//__global__ void w_inverse_pass2(float *d_tmp1, float *d_tmp2, float *output, int Nr, int Nc, int Nc2, int hlen) {
//	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
//	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
//	//int Nc2 = Nc * 2;
//	if (gidx<Nc2 && gidy< Nr) {
//		int h_left, h_right;
//		int h_len2 = hlen / 2;
//		h_left = h_len2 / 2;
//		h_right = h_len2 / 2 - 1;
//		gidx += 1;
//
//
//		int jx1 = h_left - gidx / 2;
//		int jx2 = Nc - 1 - gidx / 2 + h_left;
//		int offset = 1 - (gidx & 1);
//		float tmp_store_1 = 0, tmp_store_2 = 0;
//
//		for (int i = 0; i< h_left + h_right; ++i) {
//			int idx = gidx / 2 - h_left + i;
//			if (i<jx1) idx += Nc;
//			if (i>jx2) idx -= Nc;
//			tmp_store_1 = d_tmp1[gidy*Nc + idx] * device_IL[hlen - 1 - (2 * i + offset)];
//			tmp_store_2 = d_tmp2[gidy*Nc + idx] * device_IH[hlen - 1 - (2 * i + offset)];
//		}
//		if ((h_len2 & 1)) {
//			output[gidy*Nc2 + gidx] = tmp_store_1 + tmp_store_2;
//		}
//		else {
//			output[gidy*Nc2 + gidx - 1] = tmp_store_1 + tmp_store_2;
//		}
//
//	}
//
//}
//
//
//
//
//
//
///////////////////////////// dual trea kernel///////////////////// 
//
//__global__ void w_kernel_foward_pass1_b(float *d_image, float *d_tmp1, float *d_tmp2, int Nr, int Nc) {
//	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
//	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
//	int Nc_odd = (Nc & 1);
//	int Nc2 = (Nc + Nc_odd) / 2;
//	if (gidx<Nc2 && gidy<Nr) {
//		int h_left = hlen / 2 - 1;
//		int h_right = h_left + 1;
//
//		float tmp_store_1 = 0, tmp_store_2 = 0;
//		float img_array;
//
//		for (int i = 0; i <= h_left + h_right; ++i)
//		{
//			int idx = gidx * 2 - h_left + i;
//			if (idx < 0) { idx += (Nc + Nc_odd); }
//
//			if (idx > Nc - 1)
//			{
//				if ((idx == Nc) && (Nc_odd)) { idx--; }
//				else { idx -= (Nc + Nc_odd); }
//			}
//
//			img_array = d_image[gidy*Nc + idx];
//			tmp_store_1 += img_array*device_L_1[hlen - 1 - i];
//			tmp_store_2 += img_array*device_H_1[hlen - 1 - i];
//		}
//		d_tmp1[gidy*Nc2 + gidx] = tmp_store_1;
//		d_tmp2[gidy*Nc2 + gidx] = tmp_store_2;
//	}
//}
//
//
//// after pass1, d_tmp1 and d_tmp2 vertically convolved with L and H filter, resulting LL o_a, HL o_dv, LH o_dh, HH o_dd
//__global__ void w_kernel_foward_pass2_b(float *d_tmp1, float *d_tmp2, float *o_a, float *o_dv, float *o_dh, float *o_dd, int Nr, int Nc, int hlen) {
//	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
//	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
//	int Nr_odd = (Nr & 1);
//	int Nr2 = (Nr + Nr_odd) / 2;
//	if (gidx<Nc && gidy< Nr2) {
//		int h_left = hlen / 2 - 1;
//		int h_right = h_left + 1;
//
//		float tmp_store_1 = 0, tmp_store_2 = 0, tmp_store_3 = 0, tmp_store_4 = 0;
//		float img_array_1, img_array_2;
//
//		for (int i = 0; i <= h_left + h_right; ++i)
//		{
//			int idy = 2 * gidy - h_left + i;
//			if (idy<0) { idy += (Nr + Nr_odd); }
//			if (idy>Nr - 1) {
//				if ((idy == Nr) && (Nr_odd)) { idy--; }
//				else { idy -= (Nr + Nr_odd); }
//			}
//			img_array_1 = d_tmp1[idy*Nc + gidx];
//			img_array_2 = d_tmp2[idy*Nc + gidx];
//			tmp_store_1 += img_array_1 * device_L_1[hlen - 1 - i];
//			tmp_store_2 += img_array_2 * device_L_1[hlen - 1 - i];
//			tmp_store_3 += img_array_1 * device_H_1[hlen - 1 - i];
//			tmp_store_4 += img_array_2 * device_H_1[hlen - 1 - i];
//		}
//
//		o_a[gidy*Nc + gidx] = tmp_store_1;
//		o_dv[gidy*Nc + gidx] = tmp_store_2;
//		o_dh[gidy*Nc + gidx] = tmp_store_3;
//		o_dd[gidy*Nc + gidx] = tmp_store_4;
//	}
//}







//void w_call_circshift(float *input, float* output, int img_width, int img_row, int sr, int sc)
//{
//	// input and output are already space on cuda device
//	if (sr<0) sr += img_row;
//	if (sc<0) sc += img_width;
//
//	sr = sr % img_row;
//	sc = sc % img_width;
//	int threadnum = 16;
//	int blocksx = int((img_width - 0.5) / threadnum) + 1;
//	int blocksy = int((img_row - 0.5) / threadnum) + 1;
//	dim3 blockpergrid(blocksx, blocksy);
//	dim3 threadperblock(threadnum, threadnum);
//	w_kern_circshift<<<blockpergrid, threadperblock>>>(input, output, img_row, img_width, sr, sc);
//}
//
//
//
//void w_inverse(float *d_output, float **d_coeffs, float *d_tmp, int width, int row, int levels, int hlen) {
//	// reset d_output to zeros
//
//	int threadnum = 16;
//	float *d_tmp1 = d_tmp;
//	float *d_tmp2 = d_tmp + width*row;
//	//int level_width= div2(width);// Nc on levels
//	//int level_row = div2(row);// Nr on levels
//	int *tmp_width = new int[levels + 1];
//	int *tmp_row = new int[levels + 1];
//	tmp_width[0] = width;
//	tmp_row[0] = row;
//	for (int i = 1; i <= levels; ++i) {
//		tmp_width[i] = div2(tmp_width[i - 1]);
//		tmp_row[i] = div2(tmp_row[i - 1]);
//	}
//
//	int blocksx, blocksy, Nr, Nc, Nr2, Nc2;
//	dim3 threadperblock(threadnum, threadnum);
//	for (int i = 0; i<levels - 1; ++i) {
//		Nr = tmp_row[levels - i];
//		Nc = tmp_width[levels - i];
//		Nr2 = tmp_row[levels - i - 1];
//		Nc2 = tmp_row[levels - i - 1];
//		blocksx = int((Nr2 - 0.5) / threadnum) + 1;
//		blocksy = int((Nc - 0.5) / threadnum) + 1;
//		dim3 blockpergrid(blocksx, blocksy);
//		w_inverse_pass1 << <blockpergrid, threadperblock >> >(d_coeffs[0], d_coeffs[3 * (levels - 1 - i) + 1], d_coeffs[3 * (levels - 1 - i) + 2], d_coeffs[3 * (levels - 1 - i) + 3], d_tmp1, d_tmp2, Nr, Nc, Nr2, hlen);
//
//		//blocksx = int((Nr2-0.5)/16)+1;
//		blocksx = int((Nc2 - 0.5) / threadnum) + 1;
//		w_inverse_pass2 << <blockpergrid, threadperblock >> >(d_tmp1, d_tmp2, d_coeffs[0], Nr2, Nc, Nc2, hlen);
//	}
//	Nr = tmp_row[1];
//	Nc = tmp_width[1];
//	Nr2 = tmp_row[0];
//	Nc2 = tmp_width[0];
//	blocksx = int((Nr2 - 0.5) / threadnum) + 1;
//	blocksy = int((Nc - 0.5) / threadnum) + 1;
//	dim3 blockpergrid(blocksx, blocksy);
//	w_inverse_pass1 << <blockpergrid, threadperblock >> >(d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], d_tmp1, d_tmp2, Nr, Nc, Nr2, hlen);
//
//	//blocksx = int((Nr2-0.5)/16)+1;
//	blocksx = int((Nc2 - 0.5) / threadnum) + 1;
//	w_inverse_pass2 << <blockpergrid, threadperblock >> >(d_tmp1, d_tmp2, d_output, Nr2, Nc, Nc2, hlen);
//
//	delete[] tmp_width;
//	delete[] tmp_row;
//}
//
//
//
//
//void w_dualtree_forward(float *d_img, float **d_coeffs_a, float **d_coeffs_b, float *d_tmp_a, float * d_tmp_b, int width, int row, int levels, int hlen) {
//	int threadnum = 16;
//	int img_width_tmp = div2(width);
//	int img_row_tmp = div2(row);
//
//	int old_row, old_width;
//
//	float *d_tmp_a1 = d_tmp_a;
//	float *d_tmp_a2 = d_tmp_a + img_width_tmp*row;
//
//	float *d_tmp_b1 = d_tmp_b;
//	float *d_tmp_b2 = d_tmp_b + img_width_tmp*row;
//	int blocksx, blocksy;
//	dim3 threadperblock(threadnum, threadnum);
//
//	for (int i = 0; i<levels; ++i) {
//		if (i == 0)
//		{
//			blocksx = int((img_width_tmp - 0.5) / threadnum) + 1;
//			blocksy = int((row - 0.5) / threadnum) + 1;
//			dim3 blockpergrid1(blocksx, blocksy);
//			w_kernel_foward_pass1 << <blockpergrid1, threadperblock >> >(d_img, d_tmp_a1, d_tmp_a2, row, width, hlen);
//			w_kernel_foward_pass1_b << <blockpergrid1, threadperblock >> >(d_img, d_tmp_b1, d_tmp_b2, row, width, hlen);
//			blocksy = int((img_row_tmp - 0.5) / threadnum) + 1;
//			dim3 blockpergrid2(blocksx, blocksy);
//			w_kernel_foward_pass2 << <blockpergrid2, threadperblock >> >(d_tmp_a1, d_tmp_a2, d_coeffs_a[0], d_coeffs_a[1], d_coeffs_a[2], d_coeffs_a[3], row, img_width_tmp, hlen);
//			w_kernel_foward_pass2_b << <blockpergrid2, threadperblock >> >(d_tmp_b1, d_tmp_b2, d_coeffs_b[0], d_coeffs_b[1], d_coeffs_b[2], d_coeffs_b[3], row, img_width_tmp, hlen);
//		}
//		else {
//			old_row = img_row_tmp;
//			old_width = img_width_tmp;
//			img_width_tmp = div2(img_width_tmp);
//			blocksx = int((img_width_tmp - 0.5) / threadnum) + 1;
//			blocksy = int((old_row - 0.5) / threadnum) + 1;
//			dim3 blockpergrid1(blocksx, blocksy);
//			w_kernel_foward_pass1 << <blockpergrid1, threadperblock >> >(d_coeffs_a[0], d_tmp_a1, d_tmp_a2, old_row, old_width, hlen);
//			w_kernel_foward_pass1_b << <blockpergrid1, threadperblock >> >(d_coeffs_b[0], d_tmp_b1, d_tmp_b2, old_row, old_width, hlen);
//
//			img_row_tmp = div2(img_row_tmp);
//			blocksy = int((img_row_tmp - 0.5) / threadnum) + 1;
//			dim3 blockpergrid2(blocksx, blocksy);
//			w_kernel_foward_pass2 << <blockpergrid2, threadperblock >> >(d_tmp_a1, d_tmp_a2, d_coeffs_a[0], d_coeffs_a[3 * i + 1], d_coeffs_a[3 * i + 2], d_coeffs_a[3 * i + 3], old_row, img_width_tmp, hlen);
//			w_kernel_foward_pass2_b << <blockpergrid2, threadperblock >> >(d_tmp_b1, d_tmp_b2, d_coeffs_b[0], d_coeffs_b[3 * i + 1], d_coeffs_b[3 * i + 2], d_coeffs_b[3 * i + 3], old_row, img_width_tmp, hlen);
//		}
//	}
//}

// sum coeffcients of tree A and tree B. A+B-->d_coeffs_b A-B-->d_coeffs_A
// set alpha =1 


// difference coeffcient of tree A and tree B. the result is store in d_coeffs_a 
// set alpha = -1 
//void w_coeffcients_dif(float **d_coeffs_a, float ** d_coeffs_b, float alpha, int width, int row, int levels){
//	int Nr=row, Nc=width;
//	cublasHandle_t handle;
//	cublasCreate(&handle);
//	float al = -1;
//	for(int i=0;i<levels;++i){
//		width =div2(width);
//		row = div2(row);
//cublasSscal(handle,width*row,&al,d_coeffs_b[3*i+1],1);
//cublasSscal(handle,width*row,&al,d_coeffs_b[3*i+2],1);
//cublasSscal(handle,width*row,&al,d_coeffs_b[3*i+3],1);

//		cublasSaxpy(handle,width*row,&alpha,d_coeffs_b[3*i+1],1,d_coeffs_a[3*i+1],1);
//		cublasSaxpy(handle,width*row,&alpha,d_coeffs_b[3*i+2],1,d_coeffs_a[3*i+2],1);
//		cublasSaxpy(handle,width*row,&alpha,d_coeffs_b[3*i+3],1,d_coeffs_a[3*i+3],1);
//	}
//	cublasSaxpy(handle,width*row,&alpha,d_coeffs_a[0],1,d_coeffs_b[0],1);
//	cublasDestroy(handle);
//}






//void w_dualtree_inverse(float *d_img, float *d_img_2, float **d_coeffs_a, float **d_coeffs_b, float *d_tmp_a, float * d_tmp_b, int width, int row, int levels, int hlen) {
//	// set d_img to zeros
//	cudaMemset(d_img, 0, width*row);
//	//////////
//	int threadnum = 16;
//	float *d_tmp_a1 = d_tmp_a;
//	float *d_tmp_a2 = d_tmp_a + width*row;
//	float *d_tmp_b1 = d_tmp_b;
//	float *d_tmp_b2 = d_tmp_a + width*row;
//	//int level_width= div2(width);// Nc on levels
//	//int level_row = div2(row);// Nr on levels
//	int *tmp_width = new int[levels + 1];
//	int *tmp_row = new int[levels + 1];
//	tmp_width[0] = width;
//	tmp_row[0] = row;
//	for (int i = 1; i <= levels; ++i) {
//		tmp_width[i] = div2(tmp_width[i - 1]);
//		tmp_row[i] = div2(tmp_row[i - 1]);
//	}
//	int blocksx, blocksy, Nr, Nc, Nr2, Nc2;
//	dim3 threadperblock(threadnum, threadnum);
//	for (int i = 0; i<levels - 1; ++i) {
//		Nr = tmp_row[levels - i];
//		Nc = tmp_width[levels - i];
//		Nr2 = tmp_row[levels - i - 1];
//		Nc2 = tmp_row[levels - i - 1];
//		blocksx = int((Nr2 - 0.5) / threadnum) + 1;
//		blocksy = int((Nc - 0.5) / threadnum) + 1;
//		dim3 blockpergrid(blocksx, blocksy);
//		w_inverse_pass1 << <blockpergrid, threadperblock >> >(d_coeffs_a[0], d_coeffs_a[3 * (levels - 1 - i) + 1], d_coeffs_a[3 * (levels - 1 - i) + 2], d_coeffs_a[3 * (levels - 1 - i) + 3], d_tmp_a1, d_tmp_a2, Nr, Nc, Nr2, hlen);
//		w_inverse_pass1 << <blockpergrid, threadperblock >> >(d_coeffs_b[0], d_coeffs_b[3 * (levels - 1 - i) + 1], d_coeffs_b[3 * (levels - 1 - i) + 2], d_coeffs_b[3 * (levels - 1 - i) + 3], d_tmp_b1, d_tmp_b2, Nr, Nc, Nr2, hlen);
//
//		//blocksx = int((Nr2-0.5)/16)+1;
//		blocksx = int((Nc2 - 0.5) / threadnum) + 1;
//		w_inverse_pass2 << <blockpergrid, threadperblock >> >(d_tmp_a1, d_tmp_a2, d_coeffs_a[0], Nr2, Nc, Nc2, hlen);
//		w_inverse_pass2 << <blockpergrid, threadperblock >> >(d_tmp_b1, d_tmp_b2, d_coeffs_b[0], Nr2, Nc, Nc2, hlen);
//	}
//	Nr = tmp_row[1];
//	Nc = tmp_width[1];
//	Nr2 = tmp_row[0];
//	Nc2 = tmp_width[0];
//	blocksx = int((Nr2 - 0.5) / threadnum) + 1;
//	blocksy = int((Nc - 0.5) / threadnum) + 1;
//	dim3 blockpergrid(blocksx, blocksy);
//	w_inverse_pass1 << <blockpergrid, threadperblock >> >(d_coeffs_a[0], d_coeffs_a[1], d_coeffs_a[2], d_coeffs_a[3], d_tmp_a1, d_tmp_a2, Nr, Nc, Nr2, hlen);
//	w_inverse_pass1 << <blockpergrid, threadperblock >> >(d_coeffs_b[0], d_coeffs_b[1], d_coeffs_b[2], d_coeffs_b[3], d_tmp_b1, d_tmp_b2, Nr, Nc, Nr2, hlen);
//	blocksx = int((Nc2 - 0.5) / threadnum) + 1;
//	w_inverse_pass2 << <blockpergrid, threadperblock >> >(d_tmp_a1, d_tmp_a2, d_img, Nr2, Nc, Nc2, hlen);
//	w_inverse_pass2 << <blockpergrid, threadperblock >> >(d_tmp_b1, d_tmp_b2, d_img_2, Nr2, Nc, Nc2, hlen);
//
//	float alpha = 1;
//	cublasHandle_t handle;
//	cublasCreate(&handle);
//	cublasSaxpy(handle, width*row, &alpha, d_img, 1, d_img_2, 1);
//	cublasDestroy(handle);
//	delete[]tmp_width;
//	delete[]tmp_row;
//}



//////////////////////////////
//void test(float *d_img, float *d_tmp, float **d_coeffs, int row, int width,float *HH, int hlen) {
//	int img_width_tmp = div2(width);
//	int img_row_tmp = div2(row);
//	int old_row, old_width;
//
//	float *d_tmp1 = d_tmp;
//	float *d_tmp2 = d_tmp + img_width_tmp*row;
//
//	int threadnum = 16;
//	int blocksx, blocksy;
//	dim3 threadperblock(threadnum, threadnum);
//	blocksx = int((img_width_tmp - 0.5) / threadnum) + 1;
//	blocksy = int((row - 0.5) / threadnum) + 1;
//	dim3 blockpergrid1(blocksx, blocksy);
//	w_kernel_foward_pass1 << <blockpergrid1, threadperblock >> >(d_img, d_tmp1, d_tmp2, row, width, hlen);
//	blocksy = int((img_row_tmp - 0.5) / threadnum) + 1;
//	dim3 blockpergrid2(blocksx, blocksy);
//	w_kernel_foward_pass2 << <blockpergrid2, threadperblock >> >(d_tmp1, d_tmp2, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], row, img_width_tmp, hlen);
//	
//	cudaMemcpy(HH, d_coeffs[3], sizeof(img_width_tmp*img_row_tmp), cudaMemcpyDeviceToHost);
//}