// Written by Naoki Shibata shibatch.sf.net@gmail.com 
// http://ito-lab.naist.jp/~n-sibata/cclarticle/index.xhtml

// This program is in public domain. You can use and modify this code for any purpose without any obligation.

// This is an example implementation of a connected component labeling algorithm proposed in the following paper.
// Naoki Shibata, Shinya Yamamoto: GPGPU-Assisted Subpixel Tracking Method for Fiducial Markers,
// Journal of Information Processing, Vol.22(2014), No.1, pp.19-28, 2014-01. DOI:10.2197/ipsjjip.22.19

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <strings.h>

#include <malloc.h>
#include <immintrin.h>

#include <cv.h>
#include <highgui.h>

void abortf(const char *mes, ...) {
  va_list ap;
  va_start(ap, mes);
  vfprintf(stderr, mes, ap);
  va_end(ap);
  exit(-1);
}

#if defined(__x86_64__)
static __inline__ uint64_t rdtsc(void)
{
  uint32_t l, h;
  __asm__ __volatile__ ("rdtsc" : "=a" (l), "=d" (h));
  return (uint64_t)l | ((uint64_t)h << 32);
}
#elif defined(__i386__)
static __inline__ uint64_t int rdtsc(void)
{
  uint64_t int x;
  __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
  return x;
}
#endif

static inline int isAllOne(__m256i g) {
  return _mm_test_all_ones(_mm256_extractf128_si256(g, 0) & _mm256_extractf128_si256(g, 1));
}

static inline int isAllZero(__m256i g) {
  return _mm_testz_si128(_mm256_extractf128_si256(g, 0), _mm256_extractf128_si256(g, 0)) & _mm_testz_si128(_mm256_extractf128_si256(g, 1), _mm256_extractf128_si256(g, 1));
}

int ccl_pass1(int32_t *fb, int iw, int ih) {
  const int offset[] = {1, 1-iw, -iw, -1-iw, -1, -1+iw, iw, 1+iw};
  int ret = 0;

  for(int y = 0;y < ih;y++) {
    for(int x = 0;x < (iw >> 3);x++) {
      int32_t *p0 = &fb[y * iw + (x << 3)];

      __m256i g = _mm256_load_si256((__m256i *)p0), og = g, s, m;
      if (isAllOne(g)) continue; // continue if all bits in g are one

      for(int i=0;i<8;i++) {
	s = _mm256_loadu_si256((__m256i *)(p0 + offset[i]));
	m = _mm256_cmpeq_epi32(s, _mm256_set1_epi32(-1));
	m = _mm256_andnot_si256(m, _mm256_cmpgt_epi32(g, s));
	g = _mm256_or_si256(_mm256_and_si256(m, s), _mm256_andnot_si256(m, g));
      }

      for(int i=0;i<6;i++) {
	s = _mm256_i32gather_epi32(fb, g, 4);
	m = _mm256_cmpgt_epi32(g, s);
	if (isAllZero(m)) break;  // break if all bits in m are zero
	g = _mm256_or_si256(_mm256_and_si256(m, s), _mm256_andnot_si256(m, g));
      }

      if (isAllZero(_mm256_xor_si256(g, og))) continue; // continue if g == og

      ret = 1;

      s = _mm256_set1_epi32(0);
      for(int i=0;i<2;i++) {
	for(int j=0;j<4;j++) {
	  // You need to turn on -O3 to eliminate compile errors at the following lines
	  int32_t u = _mm_extract_epi32(_mm256_extractf128_si256( g, i), j);
	  int32_t w = _mm_extract_epi32(_mm256_extractf128_si256(og, i), j);
	  int32_t v = u < fb[w] ? u : fb[w];
	  fb[w] = v;
	  s = _mm256_inserti128_si256(s, _mm_insert_epi32(_mm256_extractf128_si256(s, i), v, j), i);
	}
      }
      g = s;

      s = _mm256_load_si256((__m256i *)(p0 - iw));
      m = _mm256_andnot_si256(_mm256_cmpeq_epi32(g, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(s, g));
      s = _mm256_or_si256(_mm256_and_si256(m, g), _mm256_andnot_si256(m, s));
      _mm256_store_si256((__m256i *)(p0 - iw), s);
      m = _mm256_andnot_si256(_mm256_cmpeq_epi32(s, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(g, s));
      g = _mm256_or_si256(_mm256_and_si256(m, s), _mm256_andnot_si256(m, g));

      s = _mm256_load_si256((__m256i *)p0);
      m = _mm256_cmpgt_epi32(g, s);
      g = _mm256_or_si256(_mm256_and_si256(m, s), _mm256_andnot_si256(m, g));
      _mm256_store_si256((__m256i *)p0, g);
    }
  }

  return ret;
}

void ccl_pass0(int32_t *fb, const int iw1, const int ih, const int32_t bgc) {
  int x, y;

  for(x = -iw1;x < 0;x += 8) {
    int *p0 = &fb[x];
    _mm256_store_si256((__m256i *)p0, _mm256_set1_epi32(-1));
  }

  for(y = 0;y < ih;y++) {
    int cnt = iw1 * y;

    for(x = 0;x < iw1;x += 8) {
      int *p0;
      __m256i s, t, u, m;

      p0 = &fb[y * iw1 + x];
      s = _mm256_add_epi32(_mm256_set1_epi32(cnt), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)); cnt += 8;
      t = _mm256_or_si256(_mm256_cmpeq_epi32(_mm256_load_si256((__m256i const *)p0), _mm256_set1_epi32(bgc)), s);

      _mm256_store_si256((__m256i *)p0, t);

      if (isAllOne(t)) continue;

      //

      u = _mm256_loadu_si256((__m256i const *)&fb[iw1 * (y-2) + x - 1]);
      s = _mm256_load_si256 ((__m256i const *)&fb[iw1 * (y-2) + x - 0]);
      m = _mm256_andnot_si256(_mm256_cmpeq_epi32(u, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(s, u));
      t = _mm256_or_si256(_mm256_and_si256(m, u), _mm256_andnot_si256(m, s));

      u = _mm256_loadu_si256((__m256i const *)&fb[iw1 * (y-1) + x - 1]);
      s = _mm256_load_si256 ((__m256i const *)&fb[iw1 * (y-1) + x - 0]);
      m = _mm256_andnot_si256(_mm256_cmpeq_epi32(u, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(s, u));
      u = _mm256_or_si256(_mm256_and_si256(m, u), _mm256_andnot_si256(m, s));
      m = _mm256_andnot_si256(_mm256_cmpeq_epi32(t, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(u, t));
      t = _mm256_or_si256(_mm256_and_si256(m, t), _mm256_andnot_si256(m, u));

      u = _mm256_loadu_si256((__m256i const *)&fb[iw1 * (y-0) + x - 1]);
      s = _mm256_load_si256 ((__m256i const *)&fb[iw1 * (y-0) + x - 0]);
      m = _mm256_andnot_si256(_mm256_cmpeq_epi32(u, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(s, u));
      u = _mm256_or_si256(_mm256_and_si256(m, u), _mm256_andnot_si256(m, s));
      m = _mm256_andnot_si256(_mm256_cmpeq_epi32(t, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(u, t));
      t = _mm256_or_si256(_mm256_and_si256(m, t), _mm256_andnot_si256(m, u));

      for(int i=0;i<7;i++) {
	u = _mm256_slli_si256(t, 4);
	u = _mm256_or_si256(_mm256_set_epi32(0,0,0,0,0,0,0,-1), u);
	u = _mm256_or_si256(u, _mm256_srli_si256(_mm256_inserti128_si256(_mm256_set1_epi32(0), _mm256_extractf128_si256(t, 0), 1), 12));
	m = _mm256_andnot_si256(_mm256_cmpeq_epi32(u, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(t, u));
	t = _mm256_or_si256(_mm256_and_si256(m, u), _mm256_andnot_si256(m, t));
      }

      _mm256_store_si256((__m256i *)p0, t);
    }
  }

  for(x = 0;x < iw1;x += 8) {
    _mm256_store_si256((__m256i *)&fb[ih * iw1 + x], _mm256_set1_epi32(-1));
  }
}

#define NPASSMAX 30

void ccl(int32_t *fb, int iw1, int ih, const int32_t bgc) {
  uint64_t t[NPASSMAX];

  t[0] = rdtsc();

  ccl_pass0(fb, iw1, ih, bgc);

  t[1] = rdtsc();

  int cnt = 0;

  while(ccl_pass1(fb, iw1, ih)) {
    t[cnt+2] = rdtsc();
    cnt++;
  }

  for(int i=0;i<cnt+1;i++) {
    printf("pass %2d : %10llu rdtsc counts\n", i, (long long unsigned int)(t[i+1] - t[i]));
  }

  printf("total   : %10llu rdtsc counts\n", (long long unsigned int)(t[cnt+1] - t[0]));
}

int main(int argc, char **argv) {
#ifdef __GNUC__
  if (!__builtin_cpu_supports("avx2")) abortf("AVX2 not available\n");
#endif

  if (argc < 2) {
    fprintf(stderr, "Usage : %s <image file name>\nThe program will threshold the image, apply CCL,\nand output the result to output.png.\n", argv[0]);
    fprintf(stderr, "You need a CPU that supports AVX2 instructions to execute this program.\n");
    exit(-1);
  }

  //

  IplImage *img = 0;
  img = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);

  if( !img ) abortf("Could not load %s\n", argv[1]);
  if (img->nChannels != 3) abortf("nChannels != 3\n");

  int iw0 = img->width, ih0 = img->height;
  int iw1 = (iw0 + 32) & ~31, ih1 = ih0 + 4;

  uint8_t *data = (uint8_t *)img->imageData;
  int32_t *fb = memalign(64, iw1 * ih1 * sizeof(int32_t));

  for(int y=0;y<ih0;y++) {
    for(int x=0;x<iw0;x++) {
      fb[(y+3) * iw1 + x] = data[y * img->widthStep + x * 3 + 1] > 127 ? 1 : 0;
    }
  }

  ccl(&fb[iw1 * 3], iw1, ih0, 0);

  for(int y=0;y<ih0;y++) {
    for(int x=0;x<iw0;x++) {
      int rgb = fb[(y+3) * iw1 + x] == -1 ? 0 : (fb[(y+3) * iw1 + x]  * 1103515245 + 12345);
      //int rgb = fb[(y+3) * iw1 + x] = fb[(y+3) * iw1 + x];
      data[y * img->widthStep + x * 3 + 0] = rgb & 0xff; rgb >>= 8;
      data[y * img->widthStep + x * 3 + 1] = rgb & 0xff; rgb >>= 8;
      data[y * img->widthStep + x * 3 + 2] = rgb & 0xff; rgb >>= 8;
    }
  }

  int params[3] = { CV_IMWRITE_PNG_COMPRESSION, 9, 0 };

  cvSaveImage("output.png", img, params);

  free(fb);

  exit(0);
}
