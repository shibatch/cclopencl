// Written by Naoki Shibata shibatch.sf.net@gmail.com 
// http://ito-lab.naist.jp/~n-sibata/cclarticle/index.xhtml

// This program is in public domain. You can use and modify this code for any purpose without any obligation.

// This is an example implementation of a connected component labeling algorithm proposed in the following paper.
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

char strbuf[10010] = "\0";

void abortf(const char *mes, ...) {
  va_list ap;
  va_start(ap, mes);
  vfprintf(stderr, mes, ap);
  va_end(ap);
  exit(-1);
}

#define MAXPLATFORMS 10
#define MAXDEVICES 10

cl_device_id simpleGetDevice(int did) {
  cl_int ret;
  cl_uint nPlatforms, nTotalDevices=0;
  cl_platform_id platformIDs[MAXPLATFORMS];
  cl_device_id devices[MAXDEVICES];

  clGetPlatformIDs(MAXPLATFORMS, platformIDs, &nPlatforms); // select first platform
  if (nPlatforms == 0) abortf("No platform available");

  int p;
  for(p=0;p<nPlatforms;p++) {
    cl_uint nDevices;
    ret = clGetDeviceIDs(platformIDs[p], CL_DEVICE_TYPE_ALL, MAXDEVICES-nTotalDevices, &devices[nTotalDevices], &nDevices);
    if (ret != CL_SUCCESS) continue;
    nTotalDevices += nDevices;
  }

  if (did < 0 || did >= nTotalDevices) {
    if (did >= 0) fprintf(stderr, "Device %d does not exist\n", did);
    int i;
    for(i=0;i<nTotalDevices;i++) {
      clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 1024, strbuf, NULL);
      fprintf(stderr, "Device %d : %s\n", i, strbuf);
    }
    exit(-1);
  }

  clGetDeviceInfo(devices[did], CL_DEVICE_NAME, 1024, strbuf, NULL);
  printf("%s ", strbuf);

  clGetDeviceInfo(devices[did], CL_DEVICE_VERSION, 1024, strbuf, NULL);
  printf("%s\n", strbuf);

  return devices[did];
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

char *readFileAsStr(const char *fn) {
  FILE *fp = fopen(fn, "r");
  if (fp == NULL) abortf("Couldn't open file %s\n", fn);

  long size;

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  if (size > 1000000) abortf("readFileAsStr : file too large\n");

  char *buf = (char *)malloc(size+10);

  fread(buf, 1, size, fp);
  buf[size] = '\0';

  fclose(fp);

  return buf;
}

#define MAXPASS 10

int main(int argc, char **argv) {
  int i;

  if (argc < 2) {
    fprintf(stderr, "Usage : %s <image file name> [<device number>]\nThe program will threshold the image, apply CCL,\nand output the result to output.png.\n", argv[0]);
    fprintf(stderr, "\nAvailable OpenCL Devices :\n");
    simpleGetDevice(-1);
    exit(-1);
  }

  //

  IplImage *img = 0;
  img = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
  if( !img ) abortf("Could not load %s\n", argv[1]);

  if (img->nChannels != 3) abortf("nChannels != 3\n");

  int iw = img->width, ih = img->height;
  uint8_t *data = (uint8_t *)img->imageData;

  //

  cl_int *bufPix   = (cl_int *)calloc(iw * ih,   sizeof(cl_int));
  cl_int *bufLabel = (cl_int *)calloc(iw * ih,   sizeof(cl_int));
  cl_int *bufFlags = (cl_int *)calloc(MAXPASS+1, sizeof(cl_int));

  {
    int x, y;
    for(y=0;y<ih;y++) {
      for(x=0;x<iw;x++) {
	bufPix[y * iw + x] = data[y * img->widthStep + x * 3 + 1] > 127 ? 1 : 0;
      }
    }
  }

  //

  int did = 0;
  if (argc >= 3) did = atoi(argv[2]);

  cl_device_id device = simpleGetDevice(did);
  cl_context context = simpleCreateContext(device);

  cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

  char *source = readFileAsStr("ccl.cl");
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, 0, NULL);

  cl_int ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (ret != CL_SUCCESS) {
    fprintf(stderr, "Could not build program : %d\n", ret);
    if (ret == CL_BUILD_PROGRAM_FAILURE) fprintf(stderr, "CL_BUILD_PROGRAM_FAILURE\n");
    if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 10000, strbuf, NULL) == CL_SUCCESS) {
      fprintf(stderr, "Build log follows\n");
      fprintf(stderr, "%s\n", strbuf);
    }
    exit(-1);
  }

  cl_kernel kernel_prepare = clCreateKernel(program, "labelxPreprocess_int_int", NULL);
  cl_kernel kernel_propagate = clCreateKernel(program, "label8xMain_int_int", NULL);

  // By specifying CL_MEM_COPY_HOST_PTR, device buffers are cleared.
  cl_mem memPix   = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), bufPix, NULL);
  cl_mem memLabel = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), bufLabel, NULL);
  cl_mem memFlags = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (MAXPASS+1) * sizeof(cl_int), bufFlags, NULL);

  size_t work_size[2] = {(size_t)((iw + 31) & ~31), (size_t)((ih + 31) & ~31)};

  cl_event events[MAXPASS+1];
  for(i=0;i<=MAXPASS;i++) {
    events[i] = clCreateUserEvent(context, NULL);
  }

  //

  clSetKernelArg(kernel_prepare, 0, sizeof(cl_mem), (void *) &memLabel);
  clSetKernelArg(kernel_prepare, 1, sizeof(cl_mem), (void *) &memPix);
  clSetKernelArg(kernel_prepare, 2, sizeof(cl_mem), (void *) &memFlags);
  i = MAXPASS; clSetKernelArg(kernel_prepare, 3, sizeof(cl_int), (void *) &i);
  i = 0; clSetKernelArg(kernel_prepare, 4, sizeof(cl_int), (void *) &i);
  clSetKernelArg(kernel_prepare, 5, sizeof(cl_int), (int *) &iw);
  clSetKernelArg(kernel_prepare, 6, sizeof(cl_int), (int *) &ih);

  clEnqueueNDRangeKernel(queue, kernel_prepare, 2, NULL, work_size, NULL, 0, NULL, &events[0]);

  for(i=1;i<=MAXPASS;i++) {
    clSetKernelArg(kernel_propagate, 0, sizeof(cl_mem), (void *) &memLabel);
    clSetKernelArg(kernel_propagate, 1, sizeof(cl_mem), (void *) &memPix);
    clSetKernelArg(kernel_propagate, 2, sizeof(cl_mem), (void *) &memFlags);
    clSetKernelArg(kernel_propagate, 3, sizeof(cl_int), (void *) &i);
    clSetKernelArg(kernel_propagate, 4, sizeof(cl_int), (int *) &iw);
    clSetKernelArg(kernel_propagate, 5, sizeof(cl_int), (int *) &ih);

    clEnqueueNDRangeKernel(queue, kernel_propagate, 2, NULL, work_size, NULL, 0, NULL, &events[i]);
  }

  clEnqueueReadBuffer(queue, memLabel, CL_TRUE, 0, iw * ih * sizeof(cl_int), bufLabel, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, memFlags, CL_TRUE, 0, (MAXPASS+1) * sizeof(cl_int), bufFlags, 0, NULL, NULL);

  clFinish(queue);

  long long int total = 0;
  for(i=0;i<=MAXPASS;i++) {
    cl_ulong tstart, tend;
    clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tstart, NULL);
    clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END  , sizeof(cl_ulong), &tend  , NULL);
    clReleaseEvent(events[i]);

    printf("pass %2d : %10lld nano sec\n", i, (long long int)(tend - tstart));
    total += tend - tstart;
  }

  printf("total   : %10lld nano sec\n", total);

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
	int rgb = bufLabel[y * iw + x] == -1 ? 0 : (bufLabel[y * iw + x]  * 1103515245 + 12345);
	//int rgb = bufLabel[y * iw + x] == -1 ? 0 : (bufLabel[y * iw + x]);
	data[y * img->widthStep + x * 3 + 0] = rgb & 0xff; rgb >>= 8;
	data[y * img->widthStep + x * 3 + 1] = rgb & 0xff; rgb >>= 8;
	data[y * img->widthStep + x * 3 + 2] = rgb & 0xff; rgb >>= 8;
      }
    }
  }

  int params[3] = { CV_IMWRITE_PNG_COMPRESSION, 9, 0 };

  cvSaveImage("output.png", img, params);

  free(bufFlags);
  free(bufLabel);
  free(bufPix);

  exit(0);
}
