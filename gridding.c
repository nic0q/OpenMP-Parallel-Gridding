#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PI   3.14159f
const float SPEED_OF_LIGHT = 299792458;

void parallel_normalization(float* fr, float* fi, float* wt, int dim);
void acumulation(float* fr, float* fi, float* wt, int i, float* vis);
void write_file(char* archive_name, float* pixels, int size);
int grid(float* vis, float deltaU, float deltaV, int N);
FILE* open_file(char* nombreArchivo);
float* line_to_float(char *vis);
float arcsec_to_rad(float deg);

int main(int argc, char *argv[]) {
  char *input_file_name = NULL, *output_file_name = NULL, str1a[] = "_privater.raw", str1b[] = "_privatei.raw", str2a[] = "_sharedr.raw", str2b[] = "_sharedi.raw";
  float deltaX = 0.0, deltaU, deltaV, *fr_priv, *fi_priv, *wt_priv, *fr, *fi, *wt;
	int t = 0, c = 0, N = 0, option, index, k, dim;
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

  fr_priv = calloc(dim, sizeof(float));
  fi_priv = calloc(dim, sizeof(float));
  wt_priv = calloc(dim, sizeof(float));

  fr = calloc(dim, sizeof(float));
  fi = calloc(dim, sizeof(float));
  wt = calloc(dim, sizeof(float));

  t1_p = omp_get_wtime();
  #pragma omp parallel firstprivate(fr, fi, wt, index)
  {
    #pragma omp single

    for (int i = 0; i < t; i++)

      #pragma omp task untied

        while (feof(file_priv) == 0) {

          char lines[c][256];

          #pragma omp critical // Critical section

            for (int j = 0; j < c; j++) {
              if (feof(file_priv) != 0)
                break;
              
              fgets(lines[j], sizeof(lines[j]), file_priv);
            }
        
          if(lines[0] != NULL) // Parallel section
            for (k = 0; k < c; k++) {
            
              float* vis = line_to_float(lines[k]); // vis = {uk, vk, vr, vu, wt frequency}

              index = grid(vis, deltaU, deltaV, N);

              acumulation(fr_priv, fi_priv, wt_priv, index, vis);
            }
        }
  
    #pragma omp taskwait

    #pragma omp for
    for (int i = 0; i < dim; i++) {
      fr[i] += fr_priv[i];
      fi[i] += fi_priv[i];
      wt[i] += wt_priv[i];
    }

    parallel_normalization(fr, fi, wt, dim);
  }
  t2_p = omp_get_wtime();

  printf("Private matrices time: %f [s]\n", t2_p - t1_p);

  write_file(strcat(output_file_name, str1a), fr, dim);
  output_file_name[strlen(output_file_name) - strlen(str1a)] = '\0';
  write_file(strcat(output_file_name, str1b), fi, dim);
  output_file_name[strlen(output_file_name) - strlen(str1b)] = '\0';

  free(fr_priv);
  free(fi_priv);
  free(wt_priv);

  free(fr);
  free(fi);
  free(wt);

  fr = calloc(dim, sizeof(float));
  fi = calloc(dim, sizeof(float));
  wt = calloc(dim, sizeof(float));

  t1_p = omp_get_wtime();
  #pragma omp parallel firstprivate(index)
  {
    #pragma omp single

    for (int i = 0; i < t; i++)
  
      #pragma omp task untied

      while(feof(file_pub) == 0) {

        char lines[c][256];        
          
        #pragma omp critical // Critical section

          for (int j = 0; j < c; j++) {
            if (feof(file_pub) != 0)
              break;
              fgets(lines[j], sizeof(lines[j]), file_pub);
          }

        if(lines[0] != NULL) // Parallel section
          for (k = 0; k < c; k++) {
            float* vis = line_to_float(lines[k]);

            index = grid(vis, deltaU, deltaV, N);

            #pragma omp critical
            {
              acumulation(fr, fi, wt, index, vis);
            }
          }
      }
  
    #pragma omp taskwait

    parallel_normalization(fr, fi, wt, dim);
  }
  t2_p = omp_get_wtime();

  printf("Shared matrices time: %f [s]\n", t2_p - t1_p);

  write_file(strcat(output_file_name, str2a), fr, dim);
  output_file_name[strlen(output_file_name) - strlen(str2a)] = '\0';
  write_file(strcat(output_file_name, str2b), fi, dim);
  output_file_name[strlen(output_file_name) - strlen(str2b)] = '\0';

  free(fr);
  free(fi);
  free(wt);
}

float arcsec_to_rad(float deg){  // arcseconds to radians
  return deg * PI / (180 * 3600);
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

float* line_to_float(char *vis) {
  float* arr = calloc(6, sizeof(float));
  sscanf(vis, "%f,%f,%*f,%f,%f,%f,%f,%*f", &arr[0], &arr[1], &arr[2], &arr[3], &arr[4], &arr[5]);
  return arr;
}

int grid(float* vis, float deltaU, float deltaV, int N) {
  float fqspeed, uk, vk;
	int index, ik, jk;
  fqspeed = vis[5] / SPEED_OF_LIGHT;
  uk = vis[0] * fqspeed;
  vk = vis[1] * fqspeed;

  ik = round(uk / deltaU) + (N / 2);
  jk = round(vk / deltaV) + (N / 2);

  return ik * N + jk;
}

void parallel_normalization(float* fr, float* fi, float* wt, int dim){
  #pragma omp for
  for (int i = 0; i < dim; i++) {
    if(wt[i]!=0) {
      fr[i] = fr[i] / wt[i];
      fi[i] = fi[i] / wt[i];
    }
	}
}
void acumulation(float* fr, float* fi, float* wt, int i, float* vis){
  fr[i] += vis[4] * vis[2];  // acumulate in matrices
  fi[i] += vis[4] * vis[3];
  wt[i] += vis[4];  
}