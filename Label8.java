// Written by Naoki Shibata shibatch.sf.net@gmail.com 
// http://ito-lab.naist.jp/~n-sibata/cclarticle/index.xhtml

// This program is in public domain. You can use and modify this code for any purpose without any obligation.

// This is an example implementation of a connected component labeling algorithm proposed in the following paper.
// Naoki Shibata, Shinya Yamamoto: GPGPU-Assisted Subpixel Tracking Method for Fiducial Markers,
// Journal of Information Processing, Vol.22(2014), No.1, pp.19-28, 2014-01. DOI:10.2197/ipsjjip.22.19

import java.io.*;
import java.awt.image.*;
import javax.imageio.*;

public class Label8 {
    static final int NPASS = 11;

    static void preparation(int[][] fb, int iw, int ih) {
        for(int y=0;y < ih;y++) {
            for(int x=0;x < iw;x++) {
		int ptr = y * iw + x;
		fb[0][ptr] = (fb[0][ptr] == 0) ? -1 : ptr;
            }
        }
    }

    static int CCLSub(int[][] fb, int pass, int x0, int y0, int iw, int ih) {
	int g = fb[pass-1][y0 * iw + x0];

	for(int y=-1;y<=1;y++) {
	    if (y + y0 < 0 || y + y0 >= ih) continue;
	    for(int x=-1;x<=1;x++) {
		if (x + x0 < 0 || x + x0 >= iw) continue;
		int q = (y + y0)*iw + x + x0;
		if (fb[pass-1][q] != -1 && fb[pass-1][q] < g) g = fb[pass-1][q];
	    }
	}

	return g;
    }

    static void propagation(int[][] fb, int pass, int iw, int ih) {
	for(int y=0;y < ih;y++) {
	    for(int x=0;x < iw;x++) {
		int ptr = y * iw + x;

		fb[pass][ptr] = fb[pass-1][ptr];

		int h = fb[pass-1][ptr];
		int g = CCLSub(fb, pass, x, y, iw, ih);

		if (g != -1) {
		    for(int i=0;i<16;i++) g = fb[pass-1][g];

		    fb[pass][h  ] = fb[pass][h  ] < g ? fb[pass][h  ] : g; // !! Atomic, referring result of current pass
		    fb[pass][ptr] = fb[pass][ptr] < g ? fb[pass][ptr] : g; // !! Atomic
		}
	    }
	}
    }

    static void label8(int[][] fb, int iw, int ih) {
	preparation(fb, iw, ih);

        for(int pass=1;pass<NPASS;pass++) {
	    propagation(fb, pass, iw, ih);
        }
    }

    public static void main(String[] args) throws Exception {
	System.setProperty("java.awt.headless", "true"); 
	BufferedImage inImage = ImageIO.read(new File(args[0]));
	int iw = inImage.getWidth(), ih = inImage.getHeight();

	int[][] fb = new int[NPASS][iw * ih];

        for(int y = 0;y < ih;y++) {
            for(int x = 0;x < iw;x++) {
		fb[0][y * iw + x] = ((inImage.getRGB(x, y) >> 8) & 255) > 127 ? 1 : 0;
            }
        }

	label8(fb, iw, ih);

	BufferedImage outImage = new BufferedImage(iw, ih, BufferedImage.TYPE_3BYTE_BGR);
        for(int y = 0;y < ih;y++) {
            for(int x = 0;x < iw;x++) {
		outImage.setRGB(x, y, fb[NPASS-1][y * iw + x] == -1 ? 0 : (fb[NPASS-1][y * iw + x]  * 1103515245 + 12345));
            }
        }
	ImageIO.write(outImage, "png", new File("output.png"));
    }
}
