// This program generates the zigzag spiral pattern.

import java.io.*;
import java.awt.*;
import java.awt.image.*;
import javax.imageio.*;

public class ZSpiral {
    public static void main(String[] args) throws Exception {
	System.setProperty("java.awt.headless", "true"); 

	int size = args.length == 0 ? 202 : (Integer.parseInt(args[0]) * 10 + 2);
	System.out.println("Generating zigzag spiral image " + size + " * " + size);
	int iw = size, ih = size;

	BufferedImage image = new BufferedImage(iw, ih, BufferedImage.TYPE_3BYTE_BGR);
	Graphics2D g = image.createGraphics();

	g.setColor(Color.black);
	g.fillRect(0, 0, iw, ih);

	g.setColor(Color.white);

	int x = 1, y = 1, len = size - 2;

	boolean first = true;

	for(;;) {
	    if (len < 5) break;

	    if (first) {
		first = false;
	    } else {
		g.drawLine(x+0-2  , y+2, x+0-2, y  );
		g.drawLine(x+0-2  , y  , x+0+0, y  );
		g.drawLine(x+0+0  , y  , x+0+0, y+2);
	    }

	    for(int i=0;i<len-4;i+=4) {
		g.drawLine(x+i  , y+2, x+i+2, y+2);
		g.drawLine(x+i+2, y+2, x+i+2, y  );
		g.drawLine(x+i+2, y  , x+i+4, y  );
		g.drawLine(x+i+4, y  , x+i+4, y+2);

		g.drawLine(size - 2 - (y+2), x+i  , size - 2 - (y+2), x+i+2);
		g.drawLine(size - 2 - (y+2), x+i+2, size - 2 - (y  ), x+i+2);
		g.drawLine(size - 2 - (y  ), x+i+2, size - 2 - (y  ), x+i+4);
		g.drawLine(size - 2 - (y  ), x+i+4, size - 2 - (y+2), x+i+4);

		g.drawLine(size - 2 - (x+i  ), size - 2 - (y+2), size - 2 - (x+i+2), size - 2 - (y+2));
		g.drawLine(size - 2 - (x+i+2), size - 2 - (y+2), size - 2 - (x+i+2), size - 2 - (y  ));
		g.drawLine(size - 2 - (x+i+2), size - 2 - (y  ), size - 2 - (x+i+4), size - 2 - (y  ));
		g.drawLine(size - 2 - (x+i+4), size - 2 - (y  ), size - 2 - (x+i+4), size - 2 - (y+2));

		if (i >= len-8) break;

		g.drawLine((y+2), size - 2- (x+i  ), (y+2), size - 2- (x+i+2));
		g.drawLine((y+2), size - 2- (x+i+2), (y  ), size - 2- (x+i+2));
		g.drawLine((y  ), size - 2- (x+i+2), (y  ), size - 2- (x+i+4));
		g.drawLine((y  ), size - 2- (x+i+4), (y+2), size - 2- (x+i+4));
	    }

	    len -= 8;
	    x += 4;
	    y += 4;
	}

	ImageIO.write(image, "png", new File("zspiral" + size + ".png"));
    }
}
