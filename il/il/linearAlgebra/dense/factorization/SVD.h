//
// Created by peruzzo on 03.05.21.
//

#ifndef IL_PARTIALSVD_H
#define IL_PARTIALSVD_H
#include <il/Array2D.h>
#include <il/Array2DView.h>
#include <il/Array.h>
#include <il/ArrayView.h>
namespace il {

  class SVD {
    // it stores the SVD decomposition
    // A = u . s . v^T
   public:
    il::Array2D<double> u;
    il::Array2D<double> vt;
    il::Array<double> s;
    il::Array2DEdit<double> u_edit;
    il::Array2DEdit<double> vt_edit;
    il::ArrayEdit<double> s_edit;

    SVD(il::int_t m, il::int_t n){
      il::Array2D<double> u_loc{m,m};
      il::Array2D<double> vt_loc{n,n};
      il::Array<double> s_loc{n};

      u=u_loc;
      vt=vt_loc;
      s=s_loc;

      u_edit = u.Edit();
      vt_edit = vt.Edit();
      s_edit = s.Edit();

    }
  };
}  // namespace il
#endif //IL_PARTIALSVD_H
