// This is a sample implementation of a connected component labeling algorithm
// Written by Naoki Shibata shibatch.sf.net@gmail.com http://ito-lab.naist.jp/~n-sibata/cclarticle/index.xhtml
// This program is in public domain. You can use and modify this code for any purpose without any obligation.

// Naoki Shibata, Shinya Yamamoto: GPGPU-Assisted Subpixel Tracking Method for Fiducial Markers,
// Journal of Information Processing, Vol.22(2014), No.1, pp.19-28, 2014-01. DOI:10.2197/ipsjjip.22.19

#include <pthread.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include <CL/cl.h>
#include <cv.h>
#include <highgui.h>

char strbuf[1030] = "\0";

void abortf(const char *mes, ...) {
  va_list ap;
  va_start(ap, mes);
  vfprintf(stderr, mes, ap);
  va_end(ap);
  abort();
}

cl_device_id simpleGetDevice() {
  cl_uint NumPlatforms;
  clGetPlatformIDs (0, NULL, &NumPlatforms);

  if (NumPlatforms == 0) abortf("No platform available");

  cl_platform_id platformID;

  clGetPlatformIDs(1, &platformID, NULL); // select first platform

  cl_int ret;
  cl_device_id device;

  ret = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  if (ret != CL_SUCCESS) {
    fprintf(stderr, "Could not get a device ID : %d\n", ret);
    if (ret == CL_DEVICE_NOT_FOUND) fprintf(stderr, "Device not found\n");
    abort();
  }

  clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, strbuf, NULL);
  printf("%s ", strbuf);

  clGetDeviceInfo(device, CL_DEVICE_VERSION, 1024, strbuf, NULL);
  printf("%s\n", strbuf);

  return device;
}

void openclErrorCallback(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
  fprintf(stderr, "\nError callback called, info = %s\n", errinfo);
}

cl_context simpleCreateContext(cl_device_id device) {
  cl_int ret;
  cl_context hContext;

  hContext = clCreateContext(NULL, 1, &device, openclErrorCallback, NULL, &ret);
  if (ret != CL_SUCCESS) abortf("Could not create context : %d\n", ret);

  return hContext;
}

char *readFileAsStr(char *fn) {
  FILE *fp = fopen(fn, "r");
  if (fp == NULL) abortf("Couldn't open file %s\n", fn);

  long size;

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  if (size > 1000000) abortf("readFileAsStr : file too large\n");

  char *buf = malloc(size+10);

  fread(buf, 1, size, fp);
  buf[size] = '\0';

  fclose(fp);

  return buf;
}

#define MAXPASS 10

int main(int argc, char **argv) {
  int i;

  if (argc < 2) abortf("Usage : %s <image file name>", argv[0]);

  //

  IplImage *img = 0;
  img = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
  if( !img ) abortf("Could not load %s\n", argv[1]);

  if (img->nChannels != 3) abortf("nChannels != 3\n");

  int iw = img->width, ih = img->height;
  uint8_t *data = (uint8_t *)img->imageData;

  //

  cl_int *bufPix = calloc(iw * ih, sizeof(cl_int)), *bufLabel = calloc(iw * ih, sizeof(cl_int)), *bufFlags = calloc(MAXPASS+1, sizeof(cl_int));

  {
    int x, y;
    for(y=0;y<ih;y++) {
      for(x=0;x<iw;x++) {
	bufPix[y * iw + x] = data[y * img->widthStep + x * 3 + 1] > 127 ? 1 : 0;
      }
    }
  }

  //

  cl_device_id device = simpleGetDevice();
  cl_context context = simpleCreateContext(device);

  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

  char *source = readFileAsStr("ccl.cl");
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, 0, NULL);

  cl_int ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (ret != CL_SUCCESS) {
    fprintf(stderr, "Could not build program : %d\n", ret);
    if (ret == CL_BUILD_PROGRAM_FAILURE) fprintf(stderr, "CL_BUILD_PROGRAM_FAILURE\n");
    if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 1024, strbuf, NULL) == CL_SUCCESS) {
      fprintf(stderr, "Build log follows\n");
      fprintf(stderr, "%s\n", strbuf);
    }
    abort();
  }

  cl_kernel kernel_prepare = clCreateKernel(program, "labelxPreprocess_int_int", NULL);
  cl_kernel kernel_propagate = clCreateKernel(program, "label8xMain_int_int", NULL);

  // By specifying CL_MEM_COPY_HOST_PTR, device buffers are cleared.
  cl_mem memPix   = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), bufPix, NULL);
  cl_mem memLabel = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), bufLabel, NULL);
  cl_mem memFlags = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (MAXPASS+1) * sizeof(cl_int), bufFlags, NULL);

  size_t work_size[2] = {iw, ih};

  //

  clSetKernelArg(kernel_prepare, 0, sizeof(cl_mem), (void *) &memLabel);
  clSetKernelArg(kernel_prepare, 1, sizeof(cl_mem), (void *) &memPix);
  clSetKernelArg(kernel_prepare, 2, sizeof(cl_mem), (void *) &memFlags);
  i = MAXPASS; clSetKernelArg(kernel_prepare, 3, sizeof(cl_int), (void *) &i);
  i = 0; clSetKernelArg(kernel_prepare, 4, sizeof(cl_int), (void *) &i);
  clSetKernelArg(kernel_prepare, 5, sizeof(cl_int), (int *) &iw);
  clSetKernelArg(kernel_prepare, 6, sizeof(cl_int), (int *) &ih);

  clEnqueueNDRangeKernel(queue, kernel_prepare, 2, NULL, work_size, NULL, 0, NULL, NULL);

  for(i=1;i<=MAXPASS;i++) {
    clSetKernelArg(kernel_propagate, 0, sizeof(cl_mem), (void *) &memLabel);
    clSetKernelArg(kernel_propagate, 1, sizeof(cl_mem), (void *) &memPix);
    clSetKernelArg(kernel_propagate, 2, sizeof(cl_mem), (void *) &memFlags);
    clSetKernelArg(kernel_propagate, 3, sizeof(cl_int), (void *) &i);
    clSetKernelArg(kernel_propagate, 4, sizeof(cl_int), (int *) &iw);
    clSetKernelArg(kernel_propagate, 5, sizeof(cl_int), (int *) &ih);

    clEnqueueNDRangeKernel(queue, kernel_propagate, 2, NULL, work_size, NULL, 0, NULL, NULL);
  }

  clEnqueueReadBuffer(queue, memLabel, CL_TRUE, 0, iw * ih * sizeof(cl_int), bufLabel, 0, NULL, NULL);

  clFinish(queue);

  clReleaseMemObject(memFlags);
  clReleaseMemObject(memLabel);
  clReleaseMemObject(memPix);
  clReleaseKernel(kernel_propagate);
  clReleaseKernel(kernel_prepare);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  //

  {
    int x, y;
    for(y=0;y<ih;y++) {
      for(x=0;x<iw;x++) {
	int rgb = bufLabel[y * iw + x] == 0 ? 0 : (bufLabel[y * iw + x]  * 1103515245 + 12345);
	data[y * img->widthStep + x * 3 + 0] = rgb & 0xff; rgb >>= 8;
	data[y * img->widthStep + x * 3 + 1] = rgb & 0xff; rgb >>= 8;
	data[y * img->widthStep + x * 3 + 2] = rgb & 0xff; rgb >>= 8;
      }
    }
  }

  cvSaveImage("output.png", img, NULL);

  free(bufFlags);
  free(bufLabel);
  free(bufPix);

  exit(0);
}
