#ifndef CONTIGUOUS_ARRAY2D_VEC
#define CONTIGUOUS_ARRAY2D_VEC


#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstddef>

#include <il/Array2D.h>

// The idea is to define a class that will hold two objects :
// 1. a vector of pointers to Array2D objects 
// 2. a buffer containing all the data of these Array2D objects
// Important : we drop the associated metadata

template <typename T>
class ContiguousArray2DVector {
private :
    std::vector<std::shared_ptr<il::Array2D<T>>> array2d_objects_;    
    // std::unique_ptr<T[]> all_data_;
    std::unique_ptr<T[], void(*)(T*)> all_data_{nullptr, [](T* ptr){ /* empty deleter */ }};
    size_t allocated_memory;

public :

    // Cache alignment considerations
    constexpr static size_t CACHE_LINE_SIZE = 64;
    // Round up to the nearest multiple of CACHE_LINE_SIZE
    static size_t align_to_cache_line(size_t size) {
        return (size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    }

    ContiguousArray2DVector(){}

    ContiguousArray2DVector(const std::vector<std::unique_ptr<il::Array2D<T>>>& source_blocks){
        copyArray2DVectorContent(source_blocks);
    }

    /* Custom delete to avoid double freeing
    */
    ~ContiguousArray2DVector() {
        // Detach all data pointers before destruction to prevent double-free
        for (auto& block : array2d_objects_) {
            block->nullifyData();  // Set data pointer to nullptr
        }
        
        // Clear the vector of Array2D objects first
        array2d_objects_.clear();
        
        // Finally free the contiguous data buffer
        all_data_.reset();
    }

    /* Where the actual work is done */
    void copyArray2DVectorContent(const std::vector<std::unique_ptr<il::Array2D<T>>>& source_blocks, const bool aligned = true){
        // We reserve the pointer vector
        array2d_objects_.reserve(source_blocks.size());

        // 1. compute total size
        size_t total_data_size = 0;
        std::vector<size_t> aligned_sizes;
        aligned_sizes.reserve(source_blocks.size());

        size_t aligned_block_size;
        for (const auto& block : source_blocks) {
            // total_data_size += block->size(0) * block->size(1);
            size_t block_size = block->size(0) * block->size(1);
            if (aligned){
                aligned_block_size = align_to_cache_line(block_size * sizeof(T)) / sizeof(T);
            } else {
                aligned_block_size = block_size; // Do not align
            }
            aligned_sizes.push_back(aligned_block_size);
            total_data_size += aligned_block_size;
        }

        // 2. allocate the buffer
        // all_data_ = std::make_unique<T[]>(total_data_size);

        // 2. Allocate with guaranteed alignment using aligned_alloc (C++17)
        // This ensures the start of the buffer is cache-line aligned
        allocated_memory = total_data_size * sizeof(T);
        T* aligned_memory = static_cast<T*>(std::aligned_alloc(
            CACHE_LINE_SIZE, total_data_size * sizeof(T)));

        // std::cout << "Data buffer total size  = " << total_data_size * sizeof(T) << "("<< total_data_size <<" elements )" << std::endl;
        
        // Use unique_ptr with custom deleter for proper cleanup
        all_data_.reset(aligned_memory);
        
        // 3. copy the Array2D elements
        size_t current_offset = 0;
        int i = 0;
        for (const auto& src_block : source_blocks) {

            // std::cout << "Source data pointer = " << src_block->data() << std::endl;
            
            // Create a new block of same size
            auto new_block = std::make_unique<il::Array2D<T>>(
                src_block->size(0),
                src_block->size(1)
            ); 

            // std::cout << "Block data pointer = " << new_block->data() << std::endl;

            // Deallocate the data we just allocated when creating the object
            new_block->deallocateData();
            
            // Redirect data pointer to our contiguous buffer
            new_block->setData(&all_data_[current_offset]);
            // std::cout << "Getting the buffer for a block at offset " << current_offset << std::endl;

            // std::cout << "Block data pointer = " << new_block->data()   << std::endl;

            // Copy the data
            auto src_block_view = src_block->view();
            auto new_block_edit = new_block->Edit();

            for (size_t i = 0; i < new_block->size(0); i++) {
                for (size_t j = 0; j < new_block->size(1); j++) {
                    // std::cout << "Copying value  " << i << "," << j << std::endl;
                    new_block_edit(i,j) = src_block_view(i ,j);
                }
            }
            // std::cout << "Copying value = done " << std::endl;

            // Update offset
            // current_offset += new_block->size(0) * new_block->size(1);
            current_offset += aligned_sizes[i];
            i++;

            // Add to our vector
            array2d_objects_.push_back(std::move(new_block));
        }
    }

    // Access function for the whole vector
    const std::vector<std::shared_ptr<il::Array2D<T>>>& blocks() const {
        return array2d_objects_;
    }

    // For the memory buffer
    const T* data() const {
        return all_data_.get();
    }

    // For the total allocated memory
    const size_t bufferSize() const {
        return allocated_memory;
    }

    // // Element access
    // il::Array2D<T>& at(size_t block_idx) {
    //     return array2d_objects_[block_idx];
    // }
    
    // Size helper
    size_t size() const {
        return array2d_objects_.size();
    }


    // Access operator 
    const il::Array2D<T>& operator[](int i) {
        return *array2d_objects_[i];
    };

};


#endif