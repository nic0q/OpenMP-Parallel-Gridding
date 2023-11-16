#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

float* fill_vis(char *vis) {
  float *array = calloc(6, sizeof(float));
  char *token = strtok(vis, ",");
  char *token2 = strtok(NULL, ",");
  char *token3 = strtok(NULL, ",");
  char *token4 = strtok(NULL, ",");
  char *token5 = strtok(NULL, ",");
  char *token6 = strtok(NULL, ",");
  char *token7 = strtok(NULL, ",");
  array[0] = atof(token);
  array[1] = atof(token2);
  array[2] = atof(token4);
  array[3] = atof(token5);
  array[4] = atof(token6);
  array[5] = atof(token7);
  return array;
}

int main(int argc, char *argv[]) {
  char vis[200] =
      "-2245.512935,-625.275579,-404.967700,0.019848,-0.009888,12065070.000000,"
      "224749993984.000000,0";
  float* hola = fill_vis(vis);
  printf("%f\n", hola[0]);
  printf("%f\n", hola[1]);
  printf("%f\n", hola[2]);
  printf("%f\n", hola[3]);
  printf("%f\n", hola[4]);
  printf("%f\n", hola[5]);
}
