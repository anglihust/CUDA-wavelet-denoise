#include "other.h"
#include <cstdio>
int log2(int i)
{
	int times = 0;
	while (i >>= 1)
	{
		++times;
	}
	return times;
}

int div2(int N)
{
	if (N & 1)
	{
		N = (N + 1) / 2;
	}
	else
	{
		N = N / 2;
	}
	return N;
}

void writefile(float *wfile, int size_n, char *save_path) {
	FILE *fp;
	fp = fopen(save_path, "wb");
	fwrite(wfile, sizeof(float), size_n, fp);
	fclose(fp);
}