//
// Created by peruzzo on 03.05.21.
//

#ifndef IL_SVD_H
#define IL_SVD_H
#include <il/linearAlgebra/dense/factorization/SVD.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#elif IL_OPENBLAS
#include <lapacke.h>
#endif

namespace il {
inline il::SVD svdDecomposition(il::io_t, il::Array2DEdit<double>& A ) {

    // NOTE: ASSUMING COLUM MAJOR STORAGE!
    //
    // Documentation:
    //https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-least-squares-and-eigenvalue-problem-routines/lapack-least-squares-and-eigenvalue-problem-driver-routines/singular-value-decomposition-lapack-driver-routines/gesvd.html

  const int layout = LAPACK_COL_MAJOR;
    const lapack_int m = static_cast<lapack_int>(A.size(0));
    const lapack_int n = static_cast<lapack_int>(A.size(1));
    const lapack_int lda = m;
  /* the leading dimension of a 2D matrix is an increment that is used to find
    *  the starting point for the matrix elements in each successive column
    *  of the array */

    const char jobu = 'A';
    const char jobvt = 'A';

    const il::int_t min_mn = m < n ? m : n;
    double superb[min_mn - 1];

    /* Local arrays */
    const lapack_int ldu = m;
    const lapack_int ldvt = n;

    il::SVD svd(A.size(0),A.size(1));

    /* Compute SVD */
    const lapack_int lapack_error = LAPACKE_dgesvd( layout, jobu, jobvt, m, n,
                         reinterpret_cast<double*>(A.Data()),
                         lda,
                         reinterpret_cast<double*>(svd.s_edit.Data()),
                         reinterpret_cast<double*>(svd.u_edit.Data()), ldu,
                         reinterpret_cast<double*>(svd.vt_edit.Data()), ldvt, superb );
    IL_EXPECT_FAST(lapack_error == 0);
    return svd;
}

}  // namespace il
#endif //IL_SVD_H
