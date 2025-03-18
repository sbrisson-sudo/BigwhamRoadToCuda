
#include <gtest/gtest.h>

#ifdef IL_BLAS
#include <il/Array2D.h>
#include <il/Array2DView.h>
#include <il/linearAlgebra/dense/factorization/svdDecomposition.h>
#include <il/blas.h>
#define min(a,b) ((a)>(b)?(b):(a))

/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, MKL_INT m, MKL_INT n,const  double* a, MKL_INT lda );
extern void print_AS_array( char* desc, MKL_INT m, MKL_INT n,const  double* a, MKL_INT lda );
extern double get_L2_norm( const  double* a, il::Array<double> sol, il::int_t size  );

TEST(Svd, test1) {
/*

   LAPACKE_dgesvd Example.
   =======================
   test taken from:
   https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_dgesvd_row.c.htm

   Program computes the singular value decomposition of a general
   rectangular matrix A:

     8.79   9.93   9.83   5.45   3.16
     6.11   6.91   5.04  -0.27   7.98
    -9.15  -7.93   4.86   4.85   3.01
     9.57   1.64   8.83   0.74   5.80
    -3.49   4.02   9.80  10.00   4.27
     9.84   0.15  -8.99  -6.02  -5.31

   Description.
   ============

   The routine computes the singular value decomposition (SVD) of a real
   m-by-n matrix A, optionally computing the left and/or right singular
   vectors. The SVD is written as

   A = U*SIGMA*VT

   where SIGMA is an m-by-n matrix which is zero except for its min(m,n)
   diagonal elements, U is an m-by-m orthogonal matrix and VT (V transposed)
   is an n-by-n orthogonal matrix. The diagonal elements of SIGMA
   are the singular values of A; they are real and non-negative, and are
   returned in descending order. The first min(m, n) columns of U and V are
   the left and right singular vectors of A.

   Note that the routine returns VT, not V.

   Example Program Results.
   ========================

 LAPACKE_dgesvd (row-major, high-level) Example Program Results

 Singular values
  27.47  22.64   8.56   5.99   2.01

 Left singular vectors (stored columnwise)
  -0.59   0.26   0.36   0.31   0.23   0.55
  -0.40   0.24  -0.22  -0.75  -0.36   0.18
  -0.03  -0.60  -0.45   0.23  -0.31   0.54
  -0.43   0.24  -0.69   0.33   0.16  -0.39
  -0.47  -0.35   0.39   0.16  -0.52  -0.46
   0.29   0.58  -0.02   0.38  -0.65   0.11

 Right singular vectors (stored rowwise)
  -0.25  -0.40  -0.69  -0.37  -0.41
   0.81   0.36  -0.25  -0.37  -0.10
  -0.26   0.70  -0.22   0.39  -0.49
   0.40  -0.45   0.25   0.43  -0.62
  -0.22   0.14   0.59  -0.63  -0.44
*/

  /* Locals */
  MKL_INT m = 6, n = 5, lda = n, ldu = m, ldvt = n, info;
  double superb[min(m,n)-1];

  il::Array2D<double> A_temp{ il::value,{{8.79,6.11,-9.15, 9.57,-3.49, 9.84 },
                                         { 9.93, 6.91, -7.93, 1.64,4.02,0.15},
                                         { 9.83, 5.04, 4.86, 8.83, 9.80,-8.99},
                                         { 5.45, -0.27, 4.85, 0.74, 10.00, -6.02},
                                         { 3.16, 7.98, 3.01,  5.80, 4.27, -5.31}}};

  il::Array2D<double> A_null{ il::value,{{0.,0.,0.,0.,0.,0. },
                                         {0.,0.,0.,0.,0.,0. },
                                         {0.,0.,0.,0.,0.,0. },
                                         {0.,0.,0.,0.,0.,0. },
                                         {0.,0.,0.,0.,0.,0. }}};

  il::Array2DEdit<double> A = A_temp.Edit();

  /* Compute SVD */
  il::SVD svd = svdDecomposition(il::io, A );


  /* THE FOLLOWING CHECKS ARE SPECIFIC FOR A MATRIX with m > n */
  /* Print singular values */
  //print_matrix( "Singular values", 1, n, reinterpret_cast<const double*>(svd.s_edit.data()), 1 );

  /* Print left singular vectors */
  //print_matrix( "Left singular vectors (stored columnwise)", m, m, reinterpret_cast<const double*>(svd.u_edit.data()), ldu );

  /* Print right singular vectors */
  //print_matrix( "Right singular vectors (stored rowwise)", n, n, reinterpret_cast<const double*>(svd.vt_edit.data()), ldvt );

  il::Array<double> left_sing_vec_sol { il::value,
                                      {-0.59,  -0.40,  -0.03,  -0.43,  -0.47,   0.29,
                                            0.26,   0.24,  -0.60,   0.24,  -0.35,   0.58,
                                            0.36,  -0.22,  -0.45,  -0.69,   0.39,  -0.02,
                                            0.31,  -0.75,   0.23,   0.33,   0.16,   0.38,
                                            0.23,  -0.36,  -0.31,   0.16,  -0.52,  -0.65,
                                            0.55,   0.18,   0.54,  -0.39,  -0.46,   0.11}};
  double err_u = get_L2_norm( svd.u_edit.data(), left_sing_vec_sol, left_sing_vec_sol.size());

  il::Array<double> right_sing_vec_sol { il::value,
                                        {  -0.25,   0.81,  -0.26,   0.40,  -0.22,
                                               -0.40,   0.36,   0.70,  -0.45,   0.14,
                                               -0.69,  -0.25,  -0.22,   0.25,   0.59,
                                               -0.37,  -0.37,   0.39,   0.43,  -0.63,
                                               -0.41,  -0.10,  -0.49,  -0.62,  -0.44}};

  double err_vt = get_L2_norm( svd.vt_edit.data(), right_sing_vec_sol, right_sing_vec_sol.size());

  /* Check decomposition
 * A = u . s. vt
 * A  is m x n (column major)
 * u  is m x m (column major)
 * s  is m x n (column major)
 * vt is n x n (column major)
 *
 *                u(i,j)*s(jj) for each i if j < n+1 with n < m
 *              /
 * [u.s](i,j) =
 *              \
 *                0 for each i, if j > n with n < m since s(jj) = 0
 * */

  for(MKL_INT i = 0; i < n; i++ ) {
    for(MKL_INT j = 0; j < m; j++ ) {
      // looping colum major, assuming m > n
      svd.u_edit.Data()[i*m+j] = svd.u_edit.Data()[i*m+j] * svd.s_edit.Data()[i] ;
    }
  }

  /*
   * [[u.s].vt](i,j) = sum over k ([u.s](i,k) * vt(k,j))
   * */
  //print_matrix( "u.s ", m, m, reinterpret_cast<const double*>(svd.u_edit.Data()), ldu );

  il::Array2DView<double> us_view = svd.u.view(il::Range{0,m},il::Range{0,n});

  //print_matrix( "u.s view ", m, n, reinterpret_cast<const double*>(us_view.data()), m );

  il::Array2DView<double> vt_view = svd.vt.view();

  //print_matrix( "vt view ", n, n, reinterpret_cast<const double*>(vt_view.data()), n );

  il::Array2DEdit<double> Aback = A_null.Edit(il::Range{0,m},il::Range{0,n});

  il::blas(1.0, us_view, vt_view,0.0, il::io, Aback);

  /* Print right singular vectors */
  //print_matrix( "A", m, n, reinterpret_cast<const double*>(Aback.Data()), n );

  //print_AS_array( "A", m, n, reinterpret_cast<const double*>(Aback.Data()), m );

  il::Array< double> const original_mat { il::value,
                                         { 8.79,   6.11,  -9.15,   9.57,  -3.49,
                                           9.84,   9.93,   6.91,  -7.93,   1.64,
                                           4.02,   0.15,   9.83,   5.04,   4.86,
                                           8.83,   9.80,  -8.99,   5.45,  -0.27,
                                           4.85,   0.74,  10.00,  -6.02,   3.16,
                                           7.98,   3.01,   5.80,   4.27,  -5.31}
                                  };

  double err_mat = get_L2_norm( Aback.data(), original_mat, original_mat.size());

/*
  Right singular vectors (stored rowwise)
  */
  const double epsilon = 0.01;
  ASSERT_TRUE(il::abs(svd.s_edit.data()[0] - 27.47)/27.47 <= epsilon &&
              il::abs(svd.s_edit.data()[1] - 22.64)/22.64 <= epsilon &&
              il::abs(svd.s_edit.data()[2] - 8.56)/8.56 <= epsilon &&
              il::abs(svd.s_edit.data()[3] - 5.99)/5.99 <= epsilon &&
              il::abs(svd.s_edit.data()[4] - 2.01)/2.01 <= epsilon);
  ASSERT_TRUE(err_u <= epsilon);
  ASSERT_TRUE(err_vt <= epsilon);
  ASSERT_TRUE(err_mat <= epsilon);
}



TEST(Svd, test2) {
/*
   Program computes the singular value decomposition of a general
   rectangular matrix A:

     8.79   9.93   9.83
     6.11   6.91   5.04

   Example Program Results.
   ========================

   Singular values
    27.47  22.64   8.56   5.99   2.01

   Left singular vectors (stored columnwise)
    -0.59   0.26   0.36   0.31   0.23   0.55
    -0.40   0.24  -0.22  -0.75  -0.36   0.18
    -0.03  -0.60  -0.45   0.23  -0.31   0.54
    -0.43   0.24  -0.69   0.33   0.16  -0.39
    -0.47  -0.35   0.39   0.16  -0.52  -0.46
     0.29   0.58  -0.02   0.38  -0.65   0.11

   Right singular vectors (stored rowwise)
    -0.25  -0.40  -0.69  -0.37  -0.41
     0.81   0.36  -0.25  -0.37  -0.10
    -0.26   0.70  -0.22   0.39  -0.49
     0.40  -0.45   0.25   0.43  -0.62
    -0.22   0.14   0.59  -0.63  -0.44
*/

  /* Locals */
  MKL_INT m = 6, n = 5;

  il::Array2D<double> A_temp{ il::value,{{8.79,6.11 },
                                         { 9.93, 6.91},
                                         { 9.83, 5.04}}};

  il::Array2D<double> A_null{ il::value,{{0.,0. },
                                         {0.,0. },
                                         {0.,0. }}};

  il::Array2DEdit<double> A = A_temp.Edit();

  /* Compute SVD */
  il::SVD svd = svdDecomposition(il::io, A );

  il::Array<double> left_sing_vec_sol { il::value,
                                        {-0.844499,  -0.535558,
                                              0.535558,  -0.844499}};

  double err_u = get_L2_norm( svd.u_edit.data(), left_sing_vec_sol, left_sing_vec_sol.size());

  il::Array<double> right_sing_vec_sol { il::value,
                                         {  -0.547589, -0.370705, -0.750149,
                                                -0.618816, -0.424027,  0.661263,
                                                -0.563217,  0.826304,  0.00279448}};

  double err_vt = get_L2_norm( svd.vt_edit.data(), right_sing_vec_sol, right_sing_vec_sol.size());


/*
  Right singular vectors (stored rowwise)
  */
  const double epsilon = 0.00001;
  ASSERT_TRUE(il::abs(svd.s_edit.data()[0] - 19.5318)/19.5318 <= epsilon &&
              il::abs(svd.s_edit.data()[1] - 1.2202)/1.2202 <= epsilon );
  ASSERT_TRUE(err_u <= epsilon);
  ASSERT_TRUE(err_vt <= epsilon);
}

#endif  // IL_BLAS



/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, MKL_INT m, MKL_INT n, const double* a, MKL_INT lda ) {
  MKL_INT i, j;
  printf( "\n %s\n", desc );
  for( i = 0; i < m; i++ ) {
    for( j = 0; j < n; j++ ) printf( " %6.2f", a[i*lda+j] );
    printf( "\n" );
  }
}

/* Auxiliary routine: printing a matrix */
void print_AS_array( char* desc, MKL_INT m, MKL_INT n, const double* a, MKL_INT lda ) {
  MKL_INT i;
  printf( "\n %s\n", desc );
  for( i = 0; i < m*n; i++ ) {
    printf( " %6.2f", a[i] );
  }
}

double get_L2_norm( const  double* a, il::Array<double> sol, il::int_t size){
  MKL_INT i;
  double err=0., num=0., den=0.;
  for( i = 0; i < size; i++ ) {
    num = num + (a[i] - sol[i]) * (a[i] - sol[i]);
    den = den + sol[i]*sol[i];
  }
  err = sqrt(num)/sqrt(den);
  return err;
}