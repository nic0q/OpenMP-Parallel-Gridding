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
int grid(float uk, float vk, float fq, float deltaU, float deltaV, int N);
void write_file(char* archive_name, float* pixels, int size);
FILE* open_file(char* nombreArchivo);

float* fill_vis(char *vis);

int main(int argc, char *argv[]) {
  float deltaX = 0.0, deltaU, deltaV, *absfr, *absfi, *abswt, *fr, *fi, *wt, uk, vk, fqspeed;
  int t = 0, c = 0, N = 0, option, ik, jk, index, pos, cont = 0, it, dim;
  char buffer[256], *input_file_name = NULL, *output_file_name = NULL;
  FILE *file_priv, *file_pub;
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
  file_priv = open_file(input_file_name);
  file_pub = open_file(input_file_name);

  if (file_priv == NULL) {
    return -1;
  }
  if (file_pub == NULL) {
    return -1;
  }

  deltaU = 1 / (N * arcsec_to_rad(deltaX));  // to radians
  deltaV = deltaU;
  dim = N * N;

  absfr = calloc(N * N, sizeof(float));
  absfi = calloc(N * N, sizeof(float));
  abswt = calloc(N * N, sizeof(float));

  fr = calloc(N * N, sizeof(float));
  fi = calloc(N * N, sizeof(float));
  wt = calloc(N * N, sizeof(float));

  t1_p = omp_get_wtime();
  #pragma omp parallel firstprivate(fr, fi, wt, ik, jk, index, uk, vk, fqspeed)
  {
    #pragma omp single

    for (int i = 0; i < t; i++)

      #pragma omp task untied

        while (feof(file_priv) == 0) {

          char lines[c][256];

          #pragma omp critical

            for (int j = 0; j < c; j++) {

              if (feof(file_priv) != 0)
                break;

              fgets(buffer, sizeof(buffer), file_priv);
              strcpy(lines[j], buffer);

              if (cont % 500000 == 0)
                printf("Reading line %d\n", cont);
              cont++;
            }
        
          if(lines[0] != NULL)
            for (it = 0; it < c; it++) {
            
              float* vis = fill_vis(lines[it]);

              index = grid(vis[0], vis[1], vis[5], deltaU, deltaV, N);

              fr[index] += vis[4] * vis[2];  // acumulate in matrix fr, fi, wt
              fi[index] += vis[4] * vis[3];
              wt[index] += vis[4];
            }
        }
  
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
        absfr[j] = absfr[j] / abswt[j];
        absfi[j] = absfi[j] / abswt[j];
      }
		}
  }

  t2_p = omp_get_wtime();

  printf("Private matrices time: %f [s]\n", t2_p - t1_p);

  write_file("datosgrideados_privater.raw", absfr, N * N);
  write_file("datosgrideados_privatei.raw", absfi, N * N);

  absfr = calloc(N * N, sizeof(float));
  absfi = calloc(N * N, sizeof(float));
  abswt = calloc(N * N, sizeof(float));

  cont = 0;

  t1_p = omp_get_wtime();
  #pragma omp parallel firstprivate(ik, jk, index, uk, vk, fqspeed)
  {
    #pragma omp single

    for (int i = 0; i < t; i++)
  
      #pragma omp task untied

      while(feof(file_pub) == 0) {

        char lines[c][256];        
          
        #pragma omp critical

          for (int j = 0; j < c; j++) {
            if (feof(file_pub) != 0)
              break;

            fgets(buffer, sizeof(buffer), file_pub);
            strcpy(lines[j], buffer);

            if (cont % 500000 == 0)
              printf("Reading line %d\n", cont);
            cont++;
          }

        if(lines[0] != NULL)
          for (it = 0; it < c; it++) {
            float* vis = fill_vis(lines[it]);

            index = grid(vis[0], vis[1], vis[5], deltaU, deltaV, N);

            #pragma omp critical
            {
              absfr[index] += vis[4] * vis[2];  // acumulate in matrices
              absfi[index] += vis[4] * vis[3];
              abswt[index] += vis[4];
            }
          }
      }
  
    #pragma omp taskwait

    #pragma omp for
    for (int j = 0; j < dim; j++) {
      if(abswt[j]!=0) {
        absfr[j] = absfr[j] / abswt[j];
        absfi[j] = absfi[j] / abswt[j];
      }
		}
  }
  t2_p = omp_get_wtime();

  printf("Shared matrices time: %f [s]\n", t2_p - t1_p);

  write_file("datosgrideados_sharedr.raw", absfr, N * N);
  write_file("datosgrideados_sharedi.raw", absfi, N * N);

  free(fr);
  free(fi);
  free(wt);
  free(absfr);
  free(absfi);
  free(abswt);
}

float* fill_vis(char *vis) {
  float* arr = calloc(6, sizeof(float));
  sscanf(vis, "%f,%f,%*f,%f,%f,%f,%f,%*f", &arr[0], &arr[1], &arr[2], &arr[3], &arr[4], &arr[5]);
  return arr;
}

int grid(float uk, float vk, float fq, float deltaU, float deltaV, int N){
  int index, ik, jk;
  float fqspeed = fq / SPEED_OF_LIGHT;
  uk = uk * fqspeed; // uk
  vk = vk * fqspeed; // vk

  ik = round(uk / deltaU) + (N / 2);  // i,j coordinate
  jk = round(vk / deltaV) + (N / 2);

  return ik * N + jk;
}

float arcsec_to_rad(float deg){  // arcseconds to radians
  return deg * M_PI / (180 * 3600);
}

FILE* open_file(char* file_name) {
  FILE* file = fopen(file_name, "r");
  if (file == NULL) {
    perror("There was an error opening the file");
    return NULL;
  }
  return file;
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

