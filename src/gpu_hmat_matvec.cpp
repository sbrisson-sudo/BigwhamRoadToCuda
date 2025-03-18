#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <sys/time.h>
#include <cmath>

#include <il/Array2D.h>

#include "fakeHMatCUDA.hpp"


int main(int argc, char* argv[]){

    // Read command line arguments
    if (argc < 3){
        std::cout << "Usage : " << argv[0] << " <block_matrix_size> <block_size>\n";
        return 1;
    }
    const int block_matrix_size = std::stoi(argv[1]);
    const int block_size = std::stoi(argv[2]);
    std::cout << "n = " << block_matrix_size << ", m = " << block_size << std::endl;

    // If argc >= 4 we also get the number of matvec opeearions to perform
    int n_matvec = 1;
    if (argc >= 4){
        n_matvec = std::stoi(argv[3]);
    }   

    // Create the fake hmat blocks
    FakeHmatCUDA fake_hmat;
    fake_hmat.initDiagBlocksMatrix(block_matrix_size, block_size);

    std::cout << "Fake hmat initialized\n";

    // ----------------------------
    // Matvec tests

    // Create lhs vector
    il::Array<double> x(block_matrix_size*block_size);

    double x_max = 1;
    double delta_x = 1/(double)(block_matrix_size*block_size-1);
    auto x_edit = x.Edit();
    x_edit[0] = 0;
    for (int i(1); i<block_matrix_size*block_size; i++) x_edit[i] = x_edit[i-1] + delta_x;

    il::Array<double> y(block_matrix_size*block_size);

    // For profiling purposes
    struct timeval t1, t2;
    double time;

    // To ensure result correctness
    double l2_norm_res;
    il::ArrayView<double> y_view;

    // ----------------------------
    // CPU matvec = OpenMP parallel
    gettimeofday(&t1, 0);

    for (int i(0); i<n_matvec; i++)
        y = fake_hmat.matvec_cpu(x.view());

    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    std::cout << "Matvec time CPU (CPU) = " << time / n_matvec << "ms\n";

    l2_norm_res = 0;
    y_view = y.view();
    for (int i(0); i<block_matrix_size*block_size; i++) l2_norm_res += y_view[i]*y_view[i];
    l2_norm_res = std::sqrt(l2_norm_res);
    std::cout << "L2 norm res CPU = " << l2_norm_res << "\n";

    // ----------------------------
    // GPU matvec 1 = multiple gemv on multiple CUDA streams
    gettimeofday(&t1, 0);

    for (int i(0); i<n_matvec; i++)
        y = fake_hmat.matvec_gpu(x.view());

    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    std::cout << "Matvec time GPU (cuda streams) = " << time / n_matvec << "ms\n";

    l2_norm_res = 0;
    y_view = y.view();
    for (int i(0); i<block_matrix_size*block_size; i++) l2_norm_res += y_view[i]*y_view[i];
    l2_norm_res = std::sqrt(l2_norm_res);
    std::cout << "L2 norm res GPU (cuda streams) = " << l2_norm_res << "\n";

    // ----------------------------
    // GPU matvec 2 = block sparse row gemv
    gettimeofday(&t1, 0);

    for (int i(0); i<n_matvec; i++)
        y = fake_hmat.matvec_gpu_block_sparse(x.view());

    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    std::cout << "Matvec time GPU (BSR) = " << time / n_matvec << "ms\n";

    l2_norm_res = 0;
    y_view = y.view();
    for (int i(0); i<block_matrix_size*block_size; i++) l2_norm_res += y_view[i]*y_view[i];
    l2_norm_res = std::sqrt(l2_norm_res);
    std::cout << "L2 norm res GPU (BSR) = " << l2_norm_res << "\n";

    // ----------------------------
    // GPU matvec 3 = batched gemv
    gettimeofday(&t1, 0);

    y = fake_hmat.matvec_gpu_batched(x.view());

    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    std::cout << "Matvec time GPU (batched) = " << time << "ms\n";

    l2_norm_res = 0;
    y_view = y.view();
    for (int i(0); i<block_matrix_size*block_size; i++) l2_norm_res += y_view[i]*y_view[i];
    l2_norm_res = std::sqrt(l2_norm_res);
    std::cout << "L2 norm res GPU (batched) = " << l2_norm_res << "\n";


    return 0;

}

