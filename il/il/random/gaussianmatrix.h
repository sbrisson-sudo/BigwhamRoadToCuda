//
// This file is part of IL.
//
// Created by Brice Lecampion on 31.12.21.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
// See the LICENSE file for more details.
//

#ifndef IL_GAUSSIANMATRIX_H
#define IL_GAUSSIANMATRIX_H

#include <il/Array2D.h>

#ifdef IL_BLAS
#include <mkl.h>
#endif

namespace il {
// generate an gaussian matrix G with zero mean and unit variance
// from wich we can always do shear_modulus_=mu + sigma^2 G
    inline il::Array2D<double> gaussianmatrix(il::int_t nr, il::int_t nc) {
        il::Array2D<double> G{nr, nc};

        VSLStreamStatePtr stream;
        const MKL_INT brng = VSL_BRNG_SOBOL;
        int error_code;
        error_code = vslNewStream(&stream, brng, static_cast<MKL_INT>(nc));
        error_code = vdRngGaussian
                (VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream,nr * nc,
                                  G.data(), 0., 1.0);
        error_code = vslDeleteStream(&stream);

        return G;
    }

}  // namespace il


#endif
