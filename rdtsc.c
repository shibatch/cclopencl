#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

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

int main(int argc, char **argv) {
  uint64_t start = rdtsc();
  sleep(1);
  uint64_t end = rdtsc();
  printf("1 second = %llu rdtsc counts\n", (long long unsigned int)(end - start));
  exit(0);
}
