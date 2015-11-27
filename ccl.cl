// Written by Naoki Shibata shibatch.sf.net@gmail.com 
// http://ito-lab.naist.jp/~n-sibata/cclarticle/index.xhtml

// This program is in public domain. You can use and modify this code for any purpose without any obligation.

// This is an example implementation of a connected component labeling algorithm proposed in the following paper.
// Naoki Shibata, Shinya Yamamoto: GPGPU-Assisted Subpixel Tracking Method for Fiducial Markers,
// Journal of Information Processing, Vol.22(2014), No.1, pp.19-28, 2014-01. DOI:10.2197/ipsjjip.22.19

__kernel void labelxPreprocess_int_int(global int *label, global int *pix, global int *flags, int maxpass, int bgc, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  const int p0 = y * iw + x;

  if (y == 0 && x < maxpass+1) {
    flags[x] = x == 0 ? 1 : 0;
  }

  if (x >= iw || y >= ih) return;

#if 0
  label[p0] = pix[p0] == bgc ? -1 : p0;
#else
  if (pix[p0] == bgc) { label[p0] = -1; return; }
  if (y > 0 && pix[p0] == pix[p0-iw]) { label[p0] = p0-iw; return; }
  if (x > 0 && pix[p0] == pix[p0- 1]) { label[p0] = p0- 1; return; }
  label[p0] = p0;
#endif
}

__kernel void label8xMain_int_int(global int *label, global int *pix, global int *flags, int pass, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  if (flags[pass-1] == 0) return;

  int g = label[p0], og = g, s;

  if (g == -1) return;

  if (y > 0) {
    int p1 = p0 - iw; s = label[p1];
    if (s < g && pix[p0] == pix[p1]) g = s;
  }

  if (x > 0) {
    int p1 = p0 - 1; s = label[p1];
    if (s < g && pix[p0] == pix[p1]) g = s;
  }

  if (x < iw-1) {
    int p1 = p0 + 1; s = label[p1];
    if (s < g && pix[p0] == pix[p1]) g = s;
  }

  if (y < ih-1) {
    int p1 = p0 + iw; s = label[p1];
    if (s < g && pix[p0] == pix[p1]) g = s;
  }

  if (y > 0 && x > 0) {
    int p1 = p0 - iw - 1; s = label[p1];
    if (s < g && pix[p0] == pix[p1]) g = s;
  }

  if (y > 0 && x < iw-1) {
    int p1 = p0 - iw + 1; s = label[p1];
    if (s < g && pix[p0] == pix[p1]) g = s;
  }

  if (y < ih-1 && x < iw-1) {
    int p1 = p0 + iw + 1; s = label[p1];
    if (s < g && pix[p0] == pix[p1]) g = s;
  }

  if (y < ih-2 && x > 1) {
    int p1 = p0 + iw - 1; s = label[p1];
    if (s < g && pix[p0] == pix[p1]) g = s;
  }

  for(int j=0;j<16;j++) {
    int s0 = label[g];
    if (s0 >= g) break;
    g = s0;
  }

  if (g != og) {
    atomic_min(&label[og], g);
    atomic_min(&label[p0], g);
    flags[pass] = 1;
  }
}
