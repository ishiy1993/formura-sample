#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include "diffusion.h"

int mpi_my_rank;

double cfl = 0.05;
double xl = 0.0;
double xr = 1.0;
double x0 = (xl + xr)/2.0;
double h = (xr-xl)/NX;
double dt = cfl*h*h;
int NT = 0.05/dt;

double gauss(double x, double y, double z) {
  return exp(-pow((x-x0)/(5*h),2.0) - pow((y-x0)/(5*h),2.0) - pow((z-x0)/(5*h),2.0));
}

void init(Formura_Navigator &navi) {
  for(int ix = navi.lower_x; ix <= navi.upper_x; ++ix) {
    double x = (ix + navi.offset_x)*h;
    for(int iy = navi.lower_y; iy <= navi.upper_y; ++iy) {
      double y = (iy + navi.offset_y)*h;
      for(int iz = navi.lower_z; iz <= navi.upper_z; ++iz) {
        double z = (iz + navi.offset_z)*h;
        q[ix][iy][iz] = gauss(x,y,z);
      }
    }
  }
}

int main(int argc, char **argv) {
  Formura_Navigator navi;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_my_rank);
  Formura_Init(&navi, MPI_COMM_WORLD);

  init(navi);

  printf("NX = %d\n", NX);
  printf("NT = %d\n", NT);

  double t;
  char fn[256];
  char fn_xy[256];
  char fn_xz[256];

  while(navi.time_step < NT) {
    t = navi.time_step * dt;
    printf("it = %d: t = %f\n", navi.time_step, t);

    sprintf(fn, "data/%f_%d.dat", t, mpi_my_rank);
    sprintf(fn_xy, "data/%f_%d-xy.dat", t, mpi_my_rank);
    sprintf(fn_xz, "data/%f_%d-xz.dat", t, mpi_my_rank);
    FILE *fp = fopen(fn, "w");
    FILE *fp_xy = fopen(fn_xy, "w");
    FILE *fp_xz = fopen(fn_xz, "w");
    for(int ix = navi.lower_x; ix <= navi.upper_x; ++ix) {
      double x = (ix + navi.offset_x)*h;
      for(int iy = navi.lower_y; iy <= navi.upper_y; ++iy) {
        double y = (iy + navi.offset_y)*h;
        for(int iz = navi.lower_z; iz <= navi.upper_z; ++iz) {
          double z = (iz + navi.offset_z)*h;
          fprintf(fp, "%f %f %f %f\n", x, y, z, q[ix][iy][iz]);

          if (z == x0) {
            fprintf(fp_xy, "%f %f %f %f\n", x, y, z, q[ix][iy][iz]);
          }

          if (y == x0) {
            fprintf(fp_xz, "%f %f %f %f\n", x, y, z, q[ix][iy][iz]);
          }
        }
        fprintf(fp, "\n");
      }
      fprintf(fp, "\n");
      fprintf(fp_xy, "\n");
      fprintf(fp_xz, "\n");
    }
    fclose(fp);
    fclose(fp_xy);
    fclose(fp_xz);
    printf("write: data/%f_%d.dat\n", t, mpi_my_rank);

    Formura_Forward(&navi);
  }

  MPI_Finalize();
}
