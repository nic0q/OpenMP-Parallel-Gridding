#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define _USE_MATH_DEFINES

const float SPEED_OF_LIGHT = 299792458;
float arcsec_to_rad(float deg);
int grid(int row, float* vis, float deltaU, float deltaV, int N);
void write_file(char* archive_name, float* pixels, int size);
FILE* open_file(char* nombreArchivo);

int main(int argc, char *argv[]) {
  int t = 0, c = 0, N = 0, option, j;
  float deltaX = 0.0, deltaU, deltaV, *absfr, *absfi, *abswt, *fr, *fi, *wt;
  double t1_p, t2_p;
  char buffer[256], *input_file_name = NULL, *output_file_name = NULL;
  
  while ((option = getopt(argc, argv, "i:o:d:N:c:t:")) != -1) {
    switch (option) {
      case 'i':
        input_file_name = optarg;
        break;
      case 'o':
        output_file_name = optarg;
        break;
      case 'd':
        deltaX = atof(optarg);
        break;
      case 'N':
        N = atoi(optarg);
        break;
      case 'c':
        c = atoi(optarg);
        break;
      case 't':
        t = atoi(optarg);
        break;
      default:
        abort();
    }
  }

  FILE* file = open_file(input_file_name);
  FILE* file2 = open_file(input_file_name);

  if(file == NULL){return -1;}

  deltaU = 1 / (N * arcsec_to_rad(deltaX));  // to radians
  deltaV = deltaU;

  absfr = calloc(N * N, sizeof(float));
  absfi = calloc(N * N, sizeof(float));
  abswt = calloc(N * N, sizeof(float));

  fr = calloc(N * N, sizeof(float));
  fi = calloc(N * N, sizeof(float));
  wt = calloc(N * N, sizeof(float));

  t1_p = omp_get_wtime();
  #pragma omp parallel shared(absfr, absfi, abswt, j) firstprivate(fr, fi, wt)
    {
    #pragma omp single
    {
    float vr, vi, wk;
    int cont = 0, it = 0, row, index, pos;
    for (int i = 0; i < t; i++){
      #pragma omp task untied shared(file) private(it) shared(cont)
      while(feof(file) == 0) {
        float* vis = calloc(c * 6, sizeof(float));
        #pragma omp critical // SC
        for (int j = 0; j < c; j++) {
          if(feof(file) != 0){
            break;
          }
          pos = j * 6;
          fgets(buffer,sizeof(buffer),file);
          sscanf(buffer, "%f,%f,%*f,%f,%f,%f,%f,%*f", &vis[pos], &vis[pos+1], &vis[pos+2], &vis[pos+3], &vis[pos+4], &vis[pos+5]);
          if(cont % 500000 == 0){
            printf("Reading line: %d\n", cont);
          }
          cont++;
        }  // SC
        if(vis[0] != 0.0)
        for(it = 0; it < c; it++){
          vr = vis[it * 6 + 2]; // parte real
          vi = vis[it * 6 + 3]; // parte im
          wk = vis[it * 6 + 4]; // peso

          index = grid(it, vis, deltaU, deltaV, N);

          fr[index] += wk * vr;  // acumulate in matrix fr, fi, wt
          fi[index] += wk * vi;
          wt[index] += wk;
        }
      } // while archivo
    } // for tasks
  } // single
  #pragma omp taskwait
  #pragma omp for
  for(int i = 0; i < N * N; i++){
    absfr[i] += fr[i];
    absfi[i] += fi[i];
    abswt[i] += wt[i];
  }
  #pragma omp for
  for (j = 0; j < N * N; j++)
    if(abswt[j] != 0){ // /0: -inf
      absfr[j] = absfr[j] / abswt[j];
      absfi[j] = absfi[j] / abswt[j];
    }
  } // Parallel
  t2_p = omp_get_wtime();
  printf("Private matrices time: %f [s]\n", t2_p - t1_p);

  write_file("datosgrideados_privater.raw", absfr, N * N);
  write_file("datosgrideados_privatei.raw", absfi, N * N);

  absfr = calloc(N * N, sizeof(float));
  absfi = calloc(N * N, sizeof(float));
  abswt = calloc(N * N, sizeof(float));

  fr = calloc(N * N, sizeof(float));
  fi = calloc(N * N, sizeof(float));
  wt = calloc(N * N, sizeof(float));

  t1_p = omp_get_wtime();
  #pragma omp parallel shared(absfr, absfi, abswt, j) firstprivate(fr, fi, wt) num_threads(t)
    {
    #pragma omp single
    {
    float uk , vk, vr, vi, wk, fq, fqspeed;
    int ik, jk, index, cont = 0, it = 0, pos;
    for (int i = 0; i < t; i++){
      #pragma omp task untied shared(file2) private(it) shared(cont)
      while(feof(file2) == 0) {
        float* vis = calloc(c * 6, sizeof(float));
        #pragma omp critical
        for (int j = 0; j < c; j++) {
          if(feof(file2) != 0){     // Si se llegó al final del archivo mientras se esta leyendo lineas del chunk, se sale del ciclo
            break;
          }
          fgets(buffer,sizeof(buffer),file2);
          pos = j * 6;
          sscanf(buffer, "%f,%f,%*f,%f,%f,%f,%f,%*f", &vis[pos], &vis[pos+1], &vis[pos+2], &vis[pos+3], &vis[pos+4], &vis[pos+5]);
          if(cont % 500000 == 0){
            printf("Reading line: %d\n", cont);
          }
          cont++;
        }
        if(vis[0] != 0.0)
        for(it = 0; it < c; it++){
          vr = vis[it * 6 + 2]; // parte real
          vi = vis[it * 6 + 3]; // parte im
          wk = vis[it * 6 + 4]; // peso

          index = grid(it, vis, deltaU, deltaV, N);

          #pragma omp critical
          {
            absfr[index] += wk * vr;  // acumulate in matrix fr, fi, wt
            absfi[index] += wk * vi;
            abswt[index] += wk;
          }
        }
      } // while archivo
    } // for tasks
  } // single
  #pragma omp taskwait
  #pragma omp for
  for (j = 0; j < N * N; j++)
    if(abswt[j]!=0){
      absfr[j] = absfr[j] / abswt[j];
      absfi[j] = absfi[j] / abswt[j];
    }
  } // Parallel  
  t2_p = omp_get_wtime();
  printf("Shared matrices time: %f [s]\n", t2_p - t1_p);

  write_file("datosgrideados_sharedr.raw", absfr, N * N);
  write_file("datosgrideados_sharedi.raw", absfi, N * N);
}

int grid(int row, float* vis, float deltaU, float deltaV, int N){
  int pos = row * 6, index, ik, jk;
  float uk, vk, wk, fq, fqspeed;
  uk = vis[pos];
  vk = vis[pos + 1];
  wk = vis[pos + 4]; // peso
  fq = vis[pos + 5];
  fqspeed = fq / SPEED_OF_LIGHT;
  uk = uk * fqspeed; // uk
  vk = vk * fqspeed; // vk

  ik = round(uk / deltaU) + (N / 2);  // i,j coordinate
  jk = round(vk / deltaV) + (N / 2);

  return ik * N + jk;
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