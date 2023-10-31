// gcc -fopenmp tasks.c -o tasks
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>
#define _USE_MATH_DEFINES
#include <math.h>
const float SPEED_OF_LIGHT = 299792458;
float arcsec_to_rad(float deg);
void write_file(char* archive_name, float* pixels, int size);
FILE* open_file(char* nombreArchivo);

int main(int argc, char *argv[]) {
  int t = 0, c = 0, N = 0, option, w;
  float deltaX = 0.0, deltaU, deltaV, *absfr, *absfi, *abswt, *fr, *fi, *wt;
  double t1_p, t2_p;
  char buffer[256], *input_file_name = NULL, *output_file_name = NULL; // string de largo 256 chars
  
  t1_p = omp_get_wtime();
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

  #pragma omp parallel shared(absfr, absfi, abswt) firstprivate(fr, fi, wt) private(w) num_threads(t)
    {
    #pragma omp single
    {
    float uk , vk, vr, vi, wk, fq, fqspeed;
    int ik, jk, index, cont = 0, a = 0;
    for (int i = 0; i < t; i++){
      #pragma omp task untied shared(file) private(a) shared(cont)
      while(feof(file) == 0) {
        float* vis = calloc(c * 6, sizeof(float));
        #pragma omp critical
        for (int j = 0; j < c; j++) {
          if(feof(file) != 0){     // Si se llegó al final del archivo mientras se esta leyendo lineas del chunk, se sale del ciclo
            break;
          }
          fgets(buffer,sizeof(buffer),file);
          int row = j * 6;
          sscanf(buffer, "%f,%f,%*f,%f,%f,%f,%f,%*f", &vis[row], &vis[row+1], &vis[row+2], &vis[row+3], &vis[row+4], &vis[row+5]);
          if(cont % 500000 == 0){
            printf("Reading line: %d\n", cont);
          }
          cont++;
        }
        if(vis[0] != 0.0)
        for(a = 0; a < c; a++){
          int row = a * 6;

          uk = vis[row];
          vk = vis[row + 1];
          vr = vis[row + 2]; // parte real
          vi = vis[row + 3]; // parte im
          wk = vis[row + 4]; // peso
          fq = vis[row + 5];

          fqspeed = fq / SPEED_OF_LIGHT;
          uk = uk * fqspeed; // uk
          vk = vk * fqspeed; // vk

          ik = round(uk / deltaU) + (N / 2);  // i,j coordinate
          jk = round(vk / deltaV) + (N / 2);

          index = ik * N + jk;
          
          fr[index] += wk * vr;  // acumulate in matrix fr, fi, wt
          fi[index] += wk * vi;
          wt[index] += wk;
        }
      } // while archivo
    } // for tasks
    #pragma omp taskwait
    #pragma omp critical
    for(int i = 0;i < N * N; i++){
      absfr[i] += fr[i];
      absfi[i] += fi[i];
      abswt[i] += wt[i];
    }
  } // single
  #pragma omp for
  for (w = 0; w < N * N; w++)
    if(abswt[w]!=0){
      absfr[w] = absfr[w] / abswt[w];
      absfi[w] = absfi[w] / abswt[w];
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
  #pragma omp parallel shared(absfr, absfi, abswt) firstprivate(fr, fi, wt) private(w) num_threads(t)
    {
    #pragma omp single
    {
    float uk , vk, vr, vi, wk, fq, fqspeed;
    int ik, jk, index, cont = 0, a = 0;
    for (int i = 0; i < t; i++){
      #pragma omp task untied shared(file2) private(a) shared(cont)
      while(feof(file2) == 0) {
        float* vis = calloc(c * 6, sizeof(float));
        #pragma omp critical
        for (int j = 0; j < c; j++) {
          if(feof(file2) != 0){     // Si se llegó al final del archivo mientras se esta leyendo lineas del chunk, se sale del ciclo
            break;
          }
          fgets(buffer,sizeof(buffer),file2);
          int row = j * 6;
          sscanf(buffer, "%f,%f,%*f,%f,%f,%f,%f,%*f", &vis[row], &vis[row+1], &vis[row+2], &vis[row+3], &vis[row+4], &vis[row+5]);
          if(cont % 500000 == 0){
            printf("Reading line: %d\n", cont);
          }
          cont++;
        }
        if(vis[0] != 0.0)
        for(a = 0; a < c; a++){
          int row = a * 6;

          uk = vis[row];
          vk = vis[row + 1];
          vr = vis[row + 2]; // parte real
          vi = vis[row + 3]; // parte im
          wk = vis[row + 4]; // peso
          fq = vis[row + 5];

          fqspeed = fq / SPEED_OF_LIGHT;
          uk = uk * fqspeed; // uk
          vk = vk * fqspeed; // vk

          ik = round(uk / deltaU) + (N / 2);  // i,j coordinate
          jk = round(vk / deltaV) + (N / 2);

          index = ik * N + jk;
          
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
  #pragma omp for
  for (w = 0; w < N * N; w++)
    if(abswt[w]!=0){
      absfr[w] = absfr[w] / abswt[w];
      absfi[w] = absfi[w] / abswt[w];
    }
  } // Parallel  
  t2_p = omp_get_wtime();
  printf("Shared matrices time: %f [s]\n", t2_p - t1_p);

  write_file("datosgrideados_sharedr.raw", absfr, N * N);
  write_file("datosgrideados_sharedi.raw", absfi, N * N);
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