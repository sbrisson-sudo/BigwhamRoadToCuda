#ifndef FAKE_HMAT
#define FAKE_HMAT

#include <vector>
#include <memory>
#include <omp.h>

#include <il/Array2D.h>
#include <il/Array.h>
#include <il/blas.h>

#include "contiguous_array2d_vector.hpp"

struct fullBlockMetadata {
    int i_start, j_start, i_end, j_end;
} ;

/**
 * @class FakeHmat
 * @brief Simulate a collection of hiearchical matrix full ranks blocks
 *
 * This class provides functions to initialize a block sparse matrix, similar to 
 * the collection of full ranks blocks of a hierarchical matix
 * It then provides a function to compute matrix-vector operations, parallelized
 * using OpenMP
 */
template <typename T>
class FakeHmat{
protected:
    size_t hmat_block_size_;    // Size of the hmat 
    size_t block_size_;         // Size of blocks, total size of hmat = hmat_block_size * block_size
    size_t size_;

    // Blocks data
    ContiguousArray2DVector<T> full_blocks_data_;

    // Blocks metadata
    std::vector<fullBlockMetadata> full_blocks_metadata;


public:
    FakeHmat(){};

    // Filler methods
    virtual void initDiagBlocksMatrix(const int hmat_size, const int block_size);
    virtual void initDiagBlocksMatrix(const int hmat_size, const int block_size, const T fill_val);

    // Helper method
    const T* data() { return full_blocks_data_.data();}
    const T* fullBlockData(const int block_id){
        return full_blocks_data_[block_id].data();
    }

    // CPU matvec operation
    il::Array<T> matvec_cpu(il::ArrayView<T> x);
};

/**
 * @brief Initialize the block sparse matrix as a block diagonal matrix
 *
 * The blocks are all set to the same value.
 * 
 * @param hmat_size The size of the matrix in number of blocks.
 * @param block_size The size of the individual blocks.
 * @param fill_val The value to fill the blocks with.
 */
template <typename T>
void FakeHmat<T>::initDiagBlocksMatrix(const int hmat_size, const int block_size, const T fill_val){

    block_size_ = block_size;
    hmat_block_size_ = hmat_size;
    size_ = block_size_*hmat_block_size_;

    const int num_blocks = hmat_size;

    std::vector<std::unique_ptr<il::Array2D<T>>> blocks_data_tmp;
    blocks_data_tmp.resize(num_blocks);
    full_blocks_metadata.reserve(num_blocks);

    for (int i(0); i<hmat_size; i++){
        full_blocks_metadata.push_back(fullBlockMetadata{
            i*block_size,
            i*block_size,
            (i+1)*block_size-1,
            (i+1)*block_size-1
        });
        std::unique_ptr<il::Array2D<T>> block_data = std::make_unique<il::Array2D<T>>(block_size,block_size,fill_val);
        blocks_data_tmp[i] = std::move(block_data);;
    }

    full_blocks_data_.copyArray2DVectorContent(blocks_data_tmp, false);
}

/**
 * @brief Initialize the block sparse matrix as a block diagonal matrix
 *
 * The blocks are filled with integer values corresponding to the order in
 * which they are created
 * 
 * @param hmat_size The size of the matrix in number of blocks.
 * @param block_size The size of the individual blocks.
 */
template <typename T>
void FakeHmat<T>::initDiagBlocksMatrix(const int hmat_size, const int block_size){

    block_size_ = block_size;
    hmat_block_size_ = hmat_size;
    size_ = block_size_*hmat_block_size_;

    const int num_blocks = hmat_size;

    std::vector<std::unique_ptr<il::Array2D<T>>> blocks_data_tmp;
    blocks_data_tmp.resize(num_blocks);
    full_blocks_metadata.reserve(num_blocks);

    double fill_val;

    for (int i(0); i<hmat_size; i++){
        full_blocks_metadata.push_back(fullBlockMetadata{
            i*block_size,
            i*block_size,
            (i+1)*block_size-1,
            (i+1)*block_size-1
        });
        fill_val = static_cast<double>(i);
        std::unique_ptr<il::Array2D<T>> block_data = std::make_unique<il::Array2D<T>>(block_size,block_size,fill_val);
        blocks_data_tmp[i] = std::move(block_data);;
    }

    full_blocks_data_.copyArray2DVectorContent(blocks_data_tmp, false);
}

/**
 * @brief Compute the matrix-vector operation on the CPU
 *
 * This function is parrallelized using OpenMP, the individual matvec 
 * operations are performed via blas dgemv (via a wrapper library)
 *
 * @param x The vector to be multiplied with.
 * @return y The result vector
 */
template <typename T>
il::Array<T> FakeHmat<T>::matvec_cpu(il::ArrayView<T> x){
    
    il::Array<T> y(size_, 0.0, il::align_t(), 64);

#pragma omp parallel
    {
        int i0, j0, iend, jend;
        il::Array<T> yprivate(size_, 0.0, il::align_t(), 64);
#pragma omp for schedule(guided) nowait
        for (il::int_t i = 0; i < full_blocks_metadata.size(); i++) {

            i0 = full_blocks_metadata[i].i_start;
            j0 = full_blocks_metadata[i].j_start;
            iend = full_blocks_metadata[i].i_end;
            jend = full_blocks_metadata[i].j_end;

            auto a = (full_blocks_data_[i]).view();

            auto xs = x.view(il::Range{j0, jend+1});
            auto ys = yprivate.Edit(il::Range{i0, iend+1});

            il::blas(1.0, a, xs, 1.0, il::io, ys);
        }
#pragma omp critical
        il::blas(1., yprivate.view(), il::io_t{}, y.Edit());
    }

    return y;
}



#endif