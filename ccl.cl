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

  if (pix[p0] == bgc) { label[p0] = -1; return; }
  if (y > 0 && pix[p0] == pix[p0-iw]) { label[p0] = p0-iw; return; }
  if (x > 0 && pix[p0] == pix[p0- 1]) { label[p0] = p0- 1; return; }
  label[p0] = p0;
}

__kernel void label8xMain_int_int(global int *label, global int *pix, global int *flags, int pass, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  if (flags[pass-1] == 0) return;

  int g = label[p0], og = g;

  if (g == -1) return;

  for(int yy=-1;yy<=1;yy++) {
    for(int xx=-1;xx<=1;xx++) {
      if (0 <=  x + xx &&  x + xx < iw && 0 <=  y + yy &&  y + yy < ih) {
	const int p1 = (y + yy) * iw + x + xx, s = label[p1];
	if (s != -1 && s < g) g = s;
      }
    }
  }

  for(int j=0;j<6;j++) g = label[g];

  if (g != og) {
    atomic_min(&label[og], g);
    atomic_min(&label[p0], g);
    flags[pass] = 1;
  }
}
