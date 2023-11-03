#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h> // clock
const float SPEED_OF_LIGHT = 299792458;
float arcsec_to_rad(float deg);
void write_file(char* archive_name, float* pixels, int size);
FILE* open_file(char* nombreArchivo);

int main(int argc, char *argv[]) {
  char buffer[256];

  FILE* file = open_file("hltau_completo_uv.csv");

  float uk, vk, vr, vi, wk, fq, fqspeed, deltaU, deltaV, deltaX = 0.003, *fr, *fi, *wt;
  int ik, jk, N = 2048, index;
  int cont = 0;
  clock_t start_s, end_s;
  double t_s;

  fr = calloc(N * N, sizeof(float));
  fi = calloc(N * N, sizeof(float));
  wt = calloc(N * N, sizeof(float));
  deltaU = 1 / (N * arcsec_to_rad(deltaX));  // to radians
  deltaV = deltaU;

  start_s = clock();
  while(feof(file) == 0) {
    if(feof(file) != 0){     // Si se llegó al final del archivo mientras se esta leyendo lineas del chunk, se sale del ciclo
      break;
    }
    fgets(buffer,sizeof(buffer), file);
    sscanf(buffer, "%f,%f,%*f,%f,%f,%f,%f,%*f", &uk, &vk, &vr, &vi, &wk, &fq);
    fqspeed = fq / SPEED_OF_LIGHT;
    uk = uk * fqspeed;
    vk = vk * fqspeed;

    ik = round(uk / deltaU) + (N / 2);
    jk = round(vk / deltaV) + (N / 2);

    index = ik * N + jk;

    fr[index] += wk * vr;
    fi[index] += wk * vi;
    wt[index] += wk;

    if(cont % 500000 == 0){
      printf("Reading line: %d\n", cont);
    }
    cont++;
  }
  for (int w = 0; w < N * N; w++){
    if(wt[w]!=0){
      fr[w] = fr[w] / wt[w];
      fi[w] = fi[w] / wt[w];
    }
  }
  end_s = clock();

  t_s = ((double)(end_s - start_s)) / (double)CLOCKS_PER_SEC;

  printf("Secuential time %f[s]\n", t_s);
  write_file("datosgrideados_secr.raw", fr, N * N);
  write_file("datosgrideados_seci.raw", fi, N * N);
  return 1;
}

float arcsec_to_rad(float deg){  // arcseconds to radians
  return deg * M_PI / (180 * 3600);
}
void write_file(char* archive_name, float* data, int dim) {
  FILE* file;
  file = fopen(archive_name, "wb");
  size_t elements_written = fwrite(data, sizeof(float), dim, file);

  if (elements_written == dim) {
    printf("All elements were written successfully.\n");
  } else {
    printf("There was an error while writing the elements.\n");
  }

  fclose(file);
}
FILE* open_file(char* file_name) {
    FILE* file = fopen(file_name, "r");
    if (file == NULL) {
        perror("There was an error opening the file");
        return NULL;
    }
    return file;
}