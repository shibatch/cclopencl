CFLAGS= `pkg-config --cflags opencv` -I.
LDFLAGS= `pkg-config --libs opencv` -L. -lOpenCL

ccl : ccl.c
	gcc -Wall $(CFLAGS) ccl.c -o ccl $(LDFLAGS)

clean :
	rm -f *~ *.o *.class ccl output.png
