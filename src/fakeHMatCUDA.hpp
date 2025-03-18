#ifndef FAKE_HMAT_CUDA
#define FAKE_HMAT_CUDA

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime_api.h>

#include "fakeHMat.hpp"

// Error checking helper functions
#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t err, const char* const func, const char* const file,
    const int line) {
    if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line
            << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
    }
}

#define CHECK_CUBLAS_ERROR(val) check_cublas((val), #val, __FILE__, __LINE__)
static const char* cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:
            return "Unknown cuBLAS error";
    }
}
void check_cublas(cublasStatus_t err, const char* const func, const char* const file,
    const int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS Error at: " << file << ":" << line
            << std::endl;
    std::cerr << cublasGetErrorString(err) << " " << func << std::endl;
    exit(1);
    }
}


/**
 * @class FakeHmatCUDA
 * @brief Extends FakeHmat to allow for GPU matvec operation using Cuda (via Cublas and Cusparse)
 *
 * This class provide initializers that initialize the block sparse matrix 
 * and set up the different objects needed to then call for Cuda blas operations
 * Three ways of performing the matvec operation are implemented :
 *   1. One relying on individual cublasDgemv calls (one per block)
 *   2. One relying on cusparseDbsrmv from Cusparse
 *   3. One relying on cublasDgemvStridedBatched (not yet functionnal)
 */
class FakeHmatCUDA : public FakeHmat<double> {
private:
    double* d_blocks_data_;     // Stores the block data on device
    double* d_x_;               // Store the lhs vector on device
    double* d_y_;               // Store the rhs vector on device

    size_t vector_size_;        // Size of the vectors (in bytes)
    bool cuda_initialized_;     
    cublasHandle_t cublas_handle_;

    // For cuda stream + cublasDgemv
    std::vector<cudaStream_t> streams_;
    int num_streams_;

    // For cublasDgemvBatched : arrays of pointers
    double** d_data_pointers_;
    double** d_x_pointers_;
    double** d_y_pointers_;
    // We keep a copy in host memory for dumping
    double** h_data_pointers_;
    double** h_x_pointers_;
    double** h_y_pointers_;

    // For BSR sparse operation
    cusparseHandle_t cusparse_handle_;
    cusparseMatDescr_t bsr_descr_;
    int* d_bsrRowPtr_;  // Device array of row pointers
    int* d_bsrColInd_;  // Device array of column indices

public :
    FakeHmatCUDA() : FakeHmat<double>(), d_blocks_data_(nullptr), cuda_initialized_(false) {
        CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle_));
        cusparseCreate(&cusparse_handle_);
    }

    void initDiagBlocksMatrix(const int hmat_size, const int block_size) override;

    // matvec operations
    il::Array<double> matvec_gpu(il::ArrayView<double> x);
    il::Array<double> matvec_gpu_batched(il::ArrayView<double> x);
    il::Array<double> matvec_gpu_block_sparse(il::ArrayView<double> x);

    // dump of pointers for checks
    double* dumpDataPtr() { return d_blocks_data_; }
    double* dumpXPtr() { return d_x_; }
    double* dumpYPtr() { return d_y_; }
    double** dumpBatchedDataPtr() { return h_data_pointers_; }
    double** dumpBatchedXPtr() { return h_x_pointers_; }
    double** dumpBatchedYPtr() { return h_y_pointers_; }

    ~FakeHmatCUDA() {
        CHECK_CUBLAS_ERROR(cublasDestroy(cublas_handle_));
        if (cuda_initialized_) {
            CHECK_CUDA_ERROR(cudaFree(d_blocks_data_
    ));
            CHECK_CUDA_ERROR(cudaFree(d_x_));
            CHECK_CUDA_ERROR(cudaFree(d_y_));
            for (int i = 0; i < num_streams_; i++) {
                CHECK_CUDA_ERROR(cudaStreamDestroy(streams_[i]));
            }
            
            // Batched
            CHECK_CUDA_ERROR(cudaFree(d_data_pointers_));
            CHECK_CUDA_ERROR(cudaFree(d_x_pointers_));
            CHECK_CUDA_ERROR(cudaFree(d_y_pointers_));

            // BSR
            cusparseDestroy(cusparse_handle_);
            CHECK_CUDA_ERROR(cudaFree(d_bsrRowPtr_));
            CHECK_CUDA_ERROR(cudaFree(d_bsrColInd_));
        }
    }
};

/**
 * @brief Initialize a disgonal block sparse matrix on the CPU and the GPU
 *
 * Initializes the matrix on the CPU and the copy its data on the GPU memory
 * On top of that various objects are initialized :
 *   1. CUDA streams that are to be used when relying on the first matvec 
 * implementation (multiple cublasDgemv calls)
 *   2. Data structures decsribing the matrix as Block sparse row matrix 
 * (see https://docs.nvidia.com/cuda/cusparse/index.html#block-sparse-row-bsr)
 *   3. For the use of batched operation : the array of pointers 
 *
 * @param hmat_size The size of the matrix in number of blocks.
 * @param block_size The size of the individual blocks.
 */
void FakeHmatCUDA::initDiagBlocksMatrix(const int hmat_size, const int block_size){

    // Call for base hmat initializer 
    FakeHmat<double>::initDiagBlocksMatrix(hmat_size, block_size);

    // Allocate and copy blocks data on device 
    size_t total_size = this->full_blocks_data_.bufferSize();
    CHECK_CUDA_ERROR(cudaMalloc(&d_blocks_data_, total_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_blocks_data_, this->full_blocks_data_.data(), total_size, cudaMemcpyHostToDevice));       
                    
    // Allocate memory for the lhs and rhs vectors
    vector_size_ = hmat_block_size_ * block_size_ * sizeof(double);
    CHECK_CUDA_ERROR(cudaMalloc(&d_x_, vector_size_));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y_, vector_size_));

    // Create CUDA streams (for matvec_gpu)
    int num_blocks =  this->full_blocks_metadata.size();
    num_streams_ = num_blocks;
    streams_.resize(num_streams_);
    for (int i = 0; i < num_streams_; i++) {
        cudaStreamCreate(&streams_[i]);
    }

    // Set up the pointers to work with batched operations (for matvec_gpu_batched)
    h_data_pointers_ = new double*[num_blocks];
    h_x_pointers_ = new double*[num_blocks];
    h_y_pointers_ = new double*[num_blocks];

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data_pointers_, num_blocks * sizeof(double*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x_pointers_, num_blocks * sizeof(double*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_y_pointers_, num_blocks * sizeof(double*)));

    int pos_data = 0;
    int i0, j0, iend, jend;
    for (int i = 0; i < num_blocks; i++) {

        i0 = full_blocks_metadata[i].i_start;
        j0 = full_blocks_metadata[i].j_start;
        iend = full_blocks_metadata[i].i_end;
        jend = full_blocks_metadata[i].j_end;

        // Set the pointers for this block
        h_data_pointers_[i] = d_blocks_data_ + pos_data * sizeof(double);
        h_x_pointers_[i] = d_x_ + j0 * sizeof(double);
        h_y_pointers_[i] = d_y_ + i0 * sizeof(double);

        // Increment the position for the data pointer
        pos_data += (iend +1 - i0) * (jend + 1 - j0);
    }

    // Then copy it to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_data_pointers_, h_data_pointers_, num_blocks * sizeof(double*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_x_pointers_, h_x_pointers_, num_blocks * sizeof(double*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_y_pointers_, h_y_pointers_, num_blocks * sizeof(double*), cudaMemcpyHostToDevice));

    // Set up the column indices and row ptr arrays for block spare row description
    int* h_bsrRowPtr_ = new int[hmat_block_size_ + 1]; // Size = num block rows + 1
    int* h_bsrColInd_ = new int[num_blocks]; // size = num blocks

    CHECK_CUDA_ERROR(cudaMalloc(&d_bsrRowPtr_, (hmat_block_size_ + 1)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bsrColInd_, num_blocks * sizeof(int)));

    // Ok for now as we have only one block per block row 
    // But for real bsr matrix : need to sort the blocks by increasing row and column
    h_bsrRowPtr_[0] = 0;
    for (int i = 0; i < hmat_block_size_; i++) h_bsrRowPtr_[i+1] = h_bsrRowPtr_[i] + 1;

    int j_start;
    for (int i = 0; i < num_blocks; i++) {
        j_start = full_blocks_metadata[i].j_start;
        h_bsrColInd_[i] = j_start / block_size_;
    }

    // Copy to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_bsrRowPtr_, h_bsrRowPtr_, (hmat_block_size_ + 1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bsrColInd_, h_bsrColInd_, num_blocks * sizeof(int), cudaMemcpyHostToDevice));

    // Matrix descriptor neede for cusparse 
    cusparseCreateMatDescr(&bsr_descr_);
    cusparseSetMatType(bsr_descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(bsr_descr_, CUSPARSE_INDEX_BASE_ZERO);
}

/**
 * @brief Compute the matrix-vector operation on the GPU via cublasDgemv
 *
 * For each blocks of our sparse matrix we set a Cuda stream as the cublas 
 * context and call cublasDgemv
 *
 * @param x The vector to be multiplied with.
 * @return y The result vector
 */
il::Array<double> FakeHmatCUDA::matvec_gpu(il::ArrayView<double> x){

    CHECK_CUDA_ERROR(cudaMemcpy(d_x_, x.data(), vector_size_, cudaMemcpyHostToDevice));      

    // We loop on the blocks
    int i0, j0, iend, jend;
    for (il::int_t i = 0; i < full_blocks_metadata.size(); i++) {

        i0 = full_blocks_metadata[i].i_start;
        j0 = full_blocks_metadata[i].j_start;
        iend = full_blocks_metadata[i].i_end;
        jend = full_blocks_metadata[i].j_end;

        // Set the CUDA stream
        CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle_, streams_[i]));

        // Call to gemv
        double alpha = 1.0;
        double beta = 0.0;
        CHECK_CUBLAS_ERROR(cublasDgemv(cublas_handle_, CUBLAS_OP_N, 
            block_size_, block_size_ ,                 // dimensions
            &alpha,                
            d_blocks_data_ + i * block_size_*block_size_, 
            block_size_, 
            d_x_ + i0, 1,         // vector x and increment
            &beta,                  // beta
            d_y_ + j0, 1));        // vector y and increment
    }

    cudaDeviceSynchronize();

    // We copy y to the host
    il::Array<double> y(hmat_block_size_*block_size_);
    double* y_data = y.Data();
    CHECK_CUDA_ERROR(cudaMemcpy(y_data, d_y_, vector_size_, cudaMemcpyDeviceToHost)); 
        
    // We reset the rhs to zero
    CHECK_CUDA_ERROR(cudaMemset(d_y_, 0, vector_size_));

    return y;
}

/**
 * @brief Compute the matrix-vector operation on the GPU via cublasDgemvStridedBatched
 *
 * Here rather than multiple gemv calls we rely on batched operations
 * This requires to pass arrays of pointers to our data and our lhs and rhs vectors
 * Data races are to be expected if multiple blocks were to write to the same 
 * part of the output vector (not the case for block diagonal matrices)
 *
 * @param x The vector to be multiplied with.
 * @return y The result vector
 */
il::Array<double> FakeHmatCUDA::matvec_gpu_batched(il::ArrayView<double> x){

    CHECK_CUDA_ERROR(cudaMemcpy(d_x_, x.data(), vector_size_, cudaMemcpyHostToDevice));   
    
    double alpha = 1.0;
    double beta = 0.0;

    // // If constant stride
    // CHECK_CUBLAS_ERROR(cublasDgemvStridedBatched(
    //     cublas_handle_, 
    //     CUBLAS_OP_N,
    //     block_size_,
    //     block_size_, 
    //     &alpha,
    //     d_blocks_data_, 
    //     block_size_, 
    //     block_size_*block_size_*sizeof(double),
    //     d_x_, 
    //     1,
    //     block_size_*sizeof(double),
    //     &beta,
    //     d_y_, 
    //     1,
    //     block_size_*sizeof(double),
    //     num_streams_
    // ));

   CHECK_CUBLAS_ERROR(cublasDgemvBatched(
        cublas_handle_, CUBLAS_OP_N,
        block_size_, block_size_, 
        &alpha,
        (const double**)d_data_pointers_, block_size_,
        (const double**)d_x_pointers_, 1,
        &beta,
        d_y_pointers_, 1,
        num_streams_
    ));

    cudaDeviceSynchronize();

    // We copy y to the host
    il::Array<double> y(hmat_block_size_*block_size_);

    double* y_data = y.Data();
    CHECK_CUDA_ERROR(cudaMemcpy(y_data, d_y_, vector_size_, cudaMemcpyDeviceToHost)); 
        
    // We reset the rhs to zero
    CHECK_CUDA_ERROR(cudaMemset(d_y_, 0, vector_size_));

    return y;
}

/**
 * @brief Compute the matrix-vector operation on the GPU via cusparseDbsrmv
 *
 * Using a description of the matrix as a block sparse row matrix, we use
 * the associated cusparse method. Not that this method is only appliquable
 * for square blocks (hence won't be usable for low rank approximations based
 * on rectangular matrices)
 *
 * @param x The vector to be multiplied with.
 * @return y The result vector
 */
il::Array<double> FakeHmatCUDA::matvec_gpu_block_sparse(il::ArrayView<double> x){

    CHECK_CUDA_ERROR(cudaMemcpy(d_x_, x.data(), vector_size_, cudaMemcpyHostToDevice)); 
    
    double alpha = 1.0;
    double beta = 0.0;

    cusparseDbsrmv(
        cusparse_handle_,
        CUSPARSE_DIRECTION_COLUMN,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        hmat_block_size_, hmat_block_size_, num_streams_,
        &alpha,
        bsr_descr_,
        d_blocks_data_,
        d_bsrRowPtr_,
        d_bsrColInd_,
        block_size_,
        d_x_,
        &beta,
        d_y_
    );

    cudaDeviceSynchronize();

    // We copy y to the host
    il::Array<double> y(hmat_block_size_*block_size_, 0.0);
    double* y_data = y.Data();
    CHECK_CUDA_ERROR(cudaMemcpy(y_data, d_y_, vector_size_, cudaMemcpyDeviceToHost)); 
        
    // We reset the rhs to zero
    CHECK_CUDA_ERROR(cudaMemset(d_y_, 0, vector_size_));

    return y;
}

#endif