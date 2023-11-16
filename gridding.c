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
int grid(float uk, float vk, float wk, float fq, float deltaU, float deltaV, int N);
void write_file(char* archive_name, float* pixels, int size);
FILE* open_file(char* nombreArchivo);

int main(int argc, char *argv[]) {
  float deltaX = 0.0, deltaU, deltaV, *absfr, *absfi, *abswt, *fr, *fi, *wt, vr, vi, wk, uk, vk, fq;
  char buffer[200], *input_file_name = NULL, *output_file_name = NULL;
  int t = 0, c = 0, N = 0, option, j, index, pos, cont = 0, it, dim;
  double t1_p, t2_p;

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

  if (file == NULL) {return -1;}

  deltaU = 1 / (N * arcsec_to_rad(deltaX));  // to radians
  deltaV = deltaU;
  dim = N * N;

  absfr = calloc(N * N, sizeof(float));
  absfi = calloc(N * N, sizeof(float));
  abswt = calloc(N * N, sizeof(float));

  fr = calloc(N * N, sizeof(float));
  fi = calloc(N * N, sizeof(float));
  wt = calloc(N * N, sizeof(float));

  char** vis_str = (char **)malloc(c * sizeof(char *));
  for (int i = 0; i < c; i++) {
    vis_str[i] = (char *)malloc(200 * sizeof(char)); // +1 para el carácter nulo
  }
  t1_p = omp_get_wtime();
  #pragma omp parallel firstprivate(fr, fi, wt)
  {
    #pragma omp single
    {
    for (int i = 0; i < t; i++) {
      #pragma omp task untied
      {
      while (feof(file) == 0) {
        #pragma omp critical // SC
        {
          for (int j = 0; j < c; j++) {
            if (feof(file) != 0) {
              break;
            }
            fgets(buffer, sizeof(buffer), file);
            strcpy(vis_str[j], buffer);
            if (cont % 500000 == 0) {
              printf("Reading line: %d\n", cont);
            }
            cont++;
          }
        } // SC
        printf("%d ", cont);
        
        for (it = 0; it < c; it++) {
          sscanf(vis_str[it], "%f,%f,%*f,%f,%f,%f,%f,%*f", &uk, &vk, &vr, &vi, &wk, &fq);

          index = grid(uk, vk, wk, fq, deltaU, deltaV, N);
          
          fr[index] += wk * vr;  // acumulate in matrix fr, fi, wt
          fi[index] += wk * vi;
          wt[index] += wk;
        }
      } // while archivo
      } // tasks
    } // for
    } // single
  #pragma omp taskwait
  #pragma omp for
  for (int i = 0; i < dim; i++) {
    absfr[i] += fr[i];
    absfi[i] += fi[i];
    abswt[i] += wt[i];
  }
  #pragma omp for
  for (int j = 0; j < dim; j++) {
    if (abswt[j] != 0) { // /0: -inf
      absfr[j] /= abswt[j];
      absfi[j] /= abswt[j];
    }
  }
  } // Parallel
  t2_p = omp_get_wtime();
  printf("Private matrices time: %f [s]\n", t2_p - t1_p);

  write_file("datosgrideados_privater.raw", absfr, N * N);
  write_file("datosgrideados_privatei.raw", absfi, N * N);

  // absfr = calloc(N * N, sizeof(float));
  // absfi = calloc(N * N, sizeof(float));
  // abswt = calloc(N * N, sizeof(float));

  // cont = 0;

  // t1_p = omp_get_wtime();
  // #pragma omp parallel
  //   {
  //   #pragma omp single
  //   {
  //   for (int i = 0; i < t; i++){
  //     #pragma omp task untied
  //     {
  //     while(feof(file2) == 0) {
  //       float* vis = calloc(c * 6, sizeof(float));
  //       #pragma omp critical
  //       for (int j = 0; j < c; j++) {
  //         if(feof(file2) != 0){     // Si se llegó al final del archivo mientras se esta leyendo lineas del chunk, se sale del ciclo
  //           break;
  //         }
  //         fgets(buffer,sizeof(buffer),file2);
  //         pos = j * 6;
  //         // sscanf(buffer, "%f,%f,%*f,%f,%f,%f,%f,%*f", &vis[pos], &vis[pos+1], &vis[pos+2], &vis[pos+3], &vis[pos+4], &vis[pos+5]);
  //         vis[0] = 4;
  //         if(cont % 500000 == 0){
  //           printf("Reading line: %d\n", cont);
  //         }
  //         cont++;
  //       }
  //       if(vis[0] != 0.0)
  //       for(it = 0; it < c; it++){
  //         vr = vis[it * 6 + 2]; // parte real
  //         vi = vis[it * 6 + 3]; // parte im
  //         wk = vis[it * 6 + 4]; // peso

  //         index = grid(it, vis, deltaU, deltaV, N);
  //         #pragma omp critical
  //         {
  //           absfr[index] += wk * vr;  // acumulate in matrices
  //           absfi[index] += wk * vi;
  //           abswt[index] += wk;
  //         }
  //       }
  //     } // while archivo
  //     } // task
  //   } // for tasks
  // } // single
  // #pragma omp taskwait
  // #pragma omp for
  // for (int j = 0; j < dim; j++)
  //   if(abswt[j]!=0){
  //     absfr[j] = absfr[j] / abswt[j];
  //     absfi[j] = absfi[j] / abswt[j];
  //   }
  // } // Parallel
  // t2_p = omp_get_wtime();
  // printf("Shared matrices time: %f [s]\n", t2_p - t1_p);

  // write_file("datosgrideados_sharedr.raw", absfr, N * N);
  // write_file("datosgrideados_sharedi.raw", absfi, N * N);
}

int grid(float uk, float vk, float wk, float fq, float deltaU, float deltaV, int N){
  int index, ik, jk;
  float fqspeed;
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