CFLAGS= `pkg-config --cflags opencv` -I.
LDFLAGS= `pkg-config --libs opencv` -L. -lOpenCL

all : ccl cclavx2 rdtsc

ccl : ccl.c
	gcc -Wall $(CFLAGS) ccl.c -o ccl $(LDFLAGS)

cclavx2 : cclavx2.c
	gcc -std=c99 -O3 -Wall $(CFLAGS) -mavx2 cclavx2.c -o cclavx2 $(LDFLAGS)

rdtsc : rdtsc.c
	gcc -Wall rdtsc.c -o rdtsc

clean :
	rm -f *~ *.o *.class ccl cclavx2 output.png
