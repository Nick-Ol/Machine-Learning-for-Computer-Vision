#include <math.h>
#include <sys/types.h>
#include "mex.h"
#ifdef WIN
typedef signed long int int32_t;
#endif
#define INF 1E8
/*
 * Generalized distance transforms.
 * We use a simple nlog(n) divide and conquer algorithm instead of the
 * theoretically faster linear method, for no particular reason except
 * that this is a bit simpler and I wanted to test it out.
 *
 * The code is a bit convoluted because dt1d can operate either along
 * a row or column of an array.
 */

/* Iasonas' note: 
* slightly modified + back-inserted (linear-time) lower envelope algorithm
* nlog(n) algorithm is still possible by calling the function as 
*  [M, Ix, Iy] = dt(vals, ax, bx, ay, by,0); 
*/

static inline int square(int x) { return x*x; }


void dt_linear(double *src, double *dst, int32_t *ptr, int step, int n, double a, double b) {
    int    *v  = new int[n];
    float *z = new float[n+1];
    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;
    for (int q = 1; q <= n-1; q++) {
        float s  = ((src[q*step] - a*square(q) + b*q )-(src[v[k]*step] - a*square(v[k]) + b*v[k]))/(-a*2*(q-v[k]));
        while (s <= z[k]) {
            k--;
            s  = ((src[q*step] - a*square(q) + b*q )-(src[v[k]*step] - a*square(v[k]) + b*v[k]))/(-a*2*(q-v[k]));
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k+1] = +INF;
    }
    k = 0;
    for (int q = 0; q <= n-1; q++) {
        while (z[k+1] < q) {
            k++;
        }
        dst[q*step] = src[v[k]*step] - a*square(q-v[k]) - b*(q-v[k]);
        ptr[q*step] = v[k];
    }
    delete [] v;
    delete [] z;
}



// dt helper function
void dt_helper(double *src, double *dst, int32_t *ptr, int step, 
					int s1, int s2, int d1, int d2, double a, double b) {
 if (d2 >= d1) {
   int d = (d1+d2) >> 1;
   int s = s1;
   for (int p = s1+1; p <= s2; p++)
     if (src[s*step] - a*square(d-s) - b*(d-s) < 
		 src[p*step] - a*square(d-p) - b*(d-p))
		s = p;
   dst[d*step] = src[s*step] - a*square(d-s) - b*(d-s);
   ptr[d*step] = s;
   dt_helper(src, dst, ptr, step, s1, s, d1, d-1, a, b);
   dt_helper(src, dst, ptr, step, s, s2, d+1, d2, a, b);
 }
}

// dt of 1d array - arguments: function, distance, max index, step, range, a,b
void dt1d(double *src, double *dst, int32_t *ptr, int step, int n, 
	  double a, double b,bool linear) {
	if (linear)
		dt_linear(src, dst, ptr, step, n, a, b);
	else
		dt_helper(src, dst, ptr, step, 0, n-1, 0, n-1, a, b);
}

// matlab entry point
// [M, Ix, Iy] = dt(vals, ax, bx, ay, by)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  if ((nrhs != 5)&&(nrhs != 6))
    mexErrMsgTxt("Wrong number of inputs"); 
  if (nlhs != 3)
    mexErrMsgTxt("Wrong number of outputs");
  if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS)
    mexErrMsgTxt("Invalid input");

  const int *dims = mxGetDimensions(prhs[0]);
  double *vals = (double *)mxGetPr(prhs[0]);
  double ax = mxGetScalar(prhs[1]);
  double bx = mxGetScalar(prhs[2]);
  double ay = mxGetScalar(prhs[3]);
  double by = mxGetScalar(prhs[4]);
  bool linear = 1;
  if (nrhs==6)
	linear  = (int) mxGetScalar(prhs[5]);
	
  mxArray *mxM = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  mxArray *mxIx = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
  mxArray *mxIy = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
  double   *M = (double *)mxGetPr(mxM);
  int32_t *Ix = (int32_t *)mxGetPr(mxIx);
  int32_t *Iy = (int32_t *)mxGetPr(mxIy);

  double *tmpM = (double *)mxCalloc(dims[0]*dims[1], sizeof(double));
  int32_t *tmpIx = (int32_t *)mxCalloc(dims[0]*dims[1], sizeof(int32_t));
  int32_t *tmpIy = (int32_t *)mxCalloc(dims[0]*dims[1], sizeof(int32_t));

  for (int x = 0; x < dims[1]; x++)
    dt1d(vals+x*dims[0], 	tmpM+x*dims[0], tmpIy+x*dims[0], 1, 	 dims[0], ay, by,linear);
  
  for (int y = 0; y < dims[0]; y++)
    dt1d(tmpM+y, 			M+y, 			tmpIx+y, 		dims[0], dims[1], ax, bx,linear);

  // get argmins and adjust for matlab indexing from 1
  for (int x = 0; x < dims[1]; x++) {
    for (int y = 0; y < dims[0]; y++) {
      int p = x*dims[0]+y;
      Ix[p] = tmpIx[p]+1;
      Iy[p] = tmpIy[tmpIx[p]*dims[0]+y]+1;
    }
  }

  mxFree(tmpM);
  mxFree(tmpIx);
  mxFree(tmpIy);
  plhs[0] = mxM;
  plhs[1] = mxIx;
  plhs[2] = mxIy;
}
