#===============================================================================
#
# Copyright 2018 The InsideLoop Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#===============================================================================
#
# To compile with icpc with full optimizations
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpc
#
# One can use -march=native with gcc and clang

cmake_minimum_required(VERSION 3.11)

set(PACKAGE_VERSION_MAJOR "0")
set(PACKAGE_VERSION_MINOR "1")

project(InsideLoop CXX)

set(IL_MATH_VENDOR "MKL_sequential" CACHE STRING "Version of cblas to use")
set_property(CACHE IL_MATH_VENDOR PROPERTY STRINGS
    MKL_sequential
    MKL_parallel
    OpenBLAS
    )
option(IL_OPENMP "OPENMP" ON)
option(IL_BENCHMARK "Activate InsideLoop Benchmarks" OFF)
option(IL_TESTING "Activate InsideLoop tests" OFF)
if(IL_BENCHMARK)
    option(IL_TBB "TBB" ON)      # --> this must be 1 at the moment
    option(IL_CILK "CILK" ON)      # --> this must be 1 at the moment
else()
    set(IL_TBB OFF CACHE BOOL "TBB" FORCE)
    set(IL_CILK OFF CACHE BOOL "CILK" FORCE)
endif()
mark_as_advanced(IL_TBB IL_CILK IL_BENCHMARK IL_TESTING)

# CBLAS handling
if(IL_MATH_VENDOR MATCHES "MKL_sequential")
    set(BLA_VENDOR Intel10_64lp_seq)
elseif(IL_MATH_VENDOR MATCHES "MKL_parallel")
    set(BLA_VENDOR Intel10_64lp)
elseif(IL_MATH_VENDOR MATCHES "OpenBLAS")
    set(BLA_VENDOR OpenBLAS)
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
#set_target_properties(BLAS::BLAS PROPERTIES IMPORTED_GLOBAL TRUE)

set(CBLAS_LIBRARY_DIRS)
foreach(_lib ${BLAS_LIBRARIES})
    if(EXISTS "${_lib}")
        get_filename_component(_blas_library_dir "${_lib}" DIRECTORY)
        get_filename_component(_parent_dir "${_blas_library_dir}" DIRECTORY)
        list(APPEND _cblas_potential_include_dir ${_parent_dir})
        if (_parent_dir MATCHES "/usr/lib")
            list(APPEND _cblas_potential_include_dir "/usr")
        endif()
    endif()
endforeach()
list(REMOVE_DUPLICATES CBLAS_LIBRARY_DIRS)
set(CBLAS_LIBRARY_DIRS ${CBLAS_LIBRARY_DIRS} PARENT_SCOPE)

find_path(CBLAS_INCLUDE_DIR NAMES cblas.h mkl_cblas.h
    HINTS ${_cblas_potential_include_dir}
          MKLROOT ENV MKLROOT
    PATH_SUFFIXES include include/intel64 include/x86_64-linux-gnu
    NO_DEFAULT_PATH
    REQUIRED
    )
mark_as_advanced(CBLAS_INCLUDE_DIR)

# OpenMP handling
if(IL_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

set(_il_compile_definitions)

# Define the library
add_library(InsideLoop INTERFACE)
target_compile_features(InsideLoop INTERFACE cxx_std_11)
target_include_directories(InsideLoop
  INTERFACE $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  )
# small trick for build includes in public
set_property(TARGET InsideLoop APPEND PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

# target_link_libraries(InsideLoop INTERFACE BLAS::BLAS LAPACK::LAPACK)
target_link_libraries(InsideLoop INTERFACE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
target_include_directories(InsideLoop SYSTEM INTERFACE ${CBLAS_INCLUDE_DIR})
list(APPEND _il_compile_definitions IL_BLAS)
if(IL_MATH_VENDOR MATCHES "MKL")
  list(APPEND _il_compile_definitions IL_MKL)
elseif(IL_MATH_VENDOR MATCHES "OpenBLAS")
  list(APPEND _il_compile_definitions IL_OPENBLAS)
endif()

if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(InsideLoop INTERFACE OpenMP::OpenMP_CXX)
    list(APPEND _il_compile_definitions IL_OPENMP)
endif()

target_compile_definitions(InsideLoop INTERFACE ${_il_compile_definitions})
# Setting variables to help with external interfaces
set(IL_COMPILE_DEFINITIONS "${_il_compile_definitions}" CACHE INTERNAL "Internal InsideLoop compiler flags")
if (IL_OPENMP)
  set(IL_COMPILE_OPTIONS "${OpenMP_CXX_FLAGS}" CACHE INTERNAL "Internal InsideLoop compiler flags")
else()
  set(IL_COMPILE_OPTIONS "" CACHE INTERNAL "Internal InsideLoop compiler flags")
endif()
if (IL_OPENMP)
  set(il_openmp_link_options ${OpenMP_CXX_FLAGS})
else()
  set(il_openmp_link_options)
endif()
set(IL_LINK_OPTIONS "${BLAS_LIBRARIES};${il_openmp_link_options}" CACHE INTERNAL "Internal InsideLoop linker flags")
set(IL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR};${CBLAS_INCLUDE_DIR}" CACHE INTERNAL "Internal InsideLoop include directories")

if(NOT INSIDELOOP_TARGETS_EXPORT)
  set(INSIDELOOP_TARGETS_EXPORT InsideLoopTargets)
endif()

include(GNUInstallDirs)
install(TARGETS InsideLoop
  EXPORT ${INSIDELOOP_TARGETS_EXPORT}
  )

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

if("${INSIDELOOP_TARGETS_EXPORT}" STREQUAL "InsideLoopTargets")
  install(EXPORT InsideLoopTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  #Export for build tree
  export(EXPORT InsideLoopTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/InsideLoopTargets.cmake")
  export(PACKAGE InsideLoop)
endif()


################################################################################
# Choose graphic framework
################################################################################
set(SOURCE_FILES
    il/algorithmArray.h
    il/algorithmArray2D.h
    il/Array.h
    il/Array2D.h
    il/Array2C.h
    il/Array2DView.h
    il/Array3C.h
    il/Array3D.h
    il/Array4D.h
    il/Array4C.h
    il/ArrayView.h
    il/BandArray2C.h
    il/core.h
    il/CudaArray2D.h
    il/data.h
    il/Dynamic.h
    il/print.h
    il/Map.h
    il/Info.h
    il/linearAlgebra.h
    il/LowerArray2D.h
    il/math.h
    il/norm.h
    il/numpy.h
    il/png.h
    il/Deque.h
    il/Set.h
    il/SmallArray.h
    il/SparseMatrixCSR.h
    il/StaticArray.h
    il/StaticArray2D.h
    il/StaticArray2C.h
    il/StaticArray3D.h
    il/StaticArray4D.h
    il/Status.h
    il/String.h
    il/string_util.h
    il/Timer.h
    il/Type.h
    il/toml.h
    il/TriDiagonal.h
    il/UTF16String.h
    il/unicode.h
    il/UpperArray2D.h
    il/algorithm/algorithmArray.h
    il/container/1d/Array.h
    il/container/1d/ArrayView.h
    il/container/1d/SmallArray.h
    il/container/1d/StaticArray.h
    il/container/2d/Array2D.h
    il/container/2d/Array2C.h
    il/container/2d/BandArray2C.h
    il/container/2d/Array2DView.h
    il/container/2d/LowerArray2D.h
    il/container/2d/SparseMatrixCSR.h
    il/container/2d/StaticArray2D.h
    il/container/2d/StaticArray2C.h
    il/container/2d/TriDiagonal.h
    il/container/2d/UpperArray2D.h
    il/container/3d/Array3C.h
    il/container/3d/Array3D.h
    il/container/3d/Array3DView.h
    il/container/3d/StaticArray3D.h
    il/container/4d/Array4D.h
    il/container/4d/Array4C.h
    il/container/4d/StaticArray4D.h
    il/container/cuda/1d/CudaArray.h
    il/container/cuda/2d/CudaArray2D.h
    il/container/cuda/2d/CudaSparseMatrixCSR.h
    il/container/cuda/cudaCopy.h
    il/container/hash/HashFunction.h
    il/container/hash/Map.h
    il/container/hash/MapArray.h
    il/container/hash/Set.h
    il/container/string/String.h
    il/container/string/UTF16String.h
    il/container/string/unicode.h
    il/container/string/algorithmString.h
    il/container/dynamic/Type.h
    il/container/dynamic/Dynamic.h
    il/container/info/Info.h
    il/container/info/Status.h
    il/container/deque/Deque.h
    il/core/core.h
    il/core/math/safe_arithmetic.h
    il/core/memory/allocate.h
    il/io/io_base.h
    il/io/ppm/ppm.h
    il/io/numpy/numpy.h
    il/io/numpy/numpy.cpp
    il/io/toml/toml.h
    il/io/toml/toml.cpp
    il/io/png/png.h
#    il/io/png/png.cpp
    il/io/filepack/filepack.h
    il/io/format/print.cc
    il/linearAlgebra/dense/blas/blas.h
    il/linearAlgebra/dense/blas/dot.h
    il/linearAlgebra/dense/norm.h
    il/linearAlgebra/dense/factorization/linearSolve.h
    il/linearAlgebra/dense/factorization/LU.h
    il/linearAlgebra/dense/factorization/Cholesky.h
    il/linearAlgebra/dense/factorization/Householder.h
    il/linearAlgebra/dense/factorization/Eigen.h
    il/linearAlgebra/dense/factorization/Singular.h
    il/linearAlgebra/dense/factorization/svdDecomposition.h
    il/linearAlgebra/sparse/blas/_code/conjugate_gradient_blaze.h
    il/linearAlgebra/sparse/blas/sparseBlas.h
    il/linearAlgebra/sparse/blas/sparseDot.h
    il/linearAlgebra/sparse/blas/sparseLinearAlgebra.h
    il/linearAlgebra/sparse/blas/SparseMatrixBlas.h
    il/linearAlgebra/cuda/dense/blas/cudaBlas.h
#    il/linearAlgebra/matrixFree/solver/Gmres.cpp
    il/unit/time.h
    il/random/sobol.h)

################################################################################
# For Unit Tests
################################################################################
if(IL_TESTING)
    ################################################################################
    # A test program
    ################################################################################
    add_executable(InsideLoopExe ${SOURCE_FILES} main.cpp)
    target_link_libraries(InsideLoopExe PRIVATE InsideLoop)
    #target_include_directories(InsideLoopExe PRIVATE ${CMAKE_SOURCE_DIR} /opt/eigen-3.3.2 /opt/blaze-3.0 /opt/cuda-8.0/include)

#    set(CMAKE_SKIP_BUILD_RPATH FALSE)
#    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

    set(IL_FOLDER "${CMAKE_SOURCE_DIR}/il")

    set(UNIT_TEST_FILES
        il/container/1d/_test/Array_test.cpp
        il/container/1d/_test/Dummy_test.h
        il/container/1d/_test/Dummy_test.cpp
        il/container/2d/_test/Array2D_test.cpp
        il/container/2d/_test/Array2C_test.cpp
        il/container/3d/_test/Array3D_test.cpp
        il/container/string/_test/String_test.cpp
        il/container/dynamic/_test/Dynamic_test.cpp
        il/container/info/_test/Info_test.cpp
        il/core/math/_test/safe_arithmetic_test.cpp
        il/linearAlgebra/dense/_test/norm_test.cpp
        il/linearAlgebra/dense/blas/_test/blas_test.cpp
        il/linearAlgebra/dense/blas/_test/linear_solve_test.cpp
        il/linearAlgebra/dense/blas/_test/dot_test.cpp
        il/linearAlgebra/dense/blas/_test/cross_test.cpp
        il/linearAlgebra/dense/factorization/_test/Eigen_test.cpp
        il/linearAlgebra/dense/factorization/_test/Singular_test.cpp
        il/linearAlgebra/dense/factorization/_test/Svd_test.cpp
        il/linearAlgebra/sparse/factorization/_test/Pardiso_test.cpp
        il/linearAlgebra/sparse/factorization/_test/GmresIlu0_test.cpp
        il/io/_test/numpy_test.cpp
        il/io/toml/_test/toml_valid_test.cpp
        gtest/src/gtest-all.cc
        il/core/_test/Status_test.cpp
        il/MapArray.h)
    #il/container/string/_test/StringView_test.cpp il/container/tree/_test/Tree_test.cpp)

    add_executable(InsideLoopUnitTest ${SOURCE_FILES} ${UNIT_TEST_FILES} test.cpp)

    target_compile_definitions(InsideLoopUnitTest PRIVATE IL_FOLDER='\"${IL_FOLDER}\"')
    target_include_directories(InsideLoopUnitTest PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/gtest)
    target_link_libraries(InsideLoopUnitTest PRIVATE InsideLoop)

    # For unit tests: The precondition of our fonctions are checked with assert
    # macros that terminate the program in debug mode. In order to test those macros
    # in our unit tests, the flag IL_UNIT_TEST turns them into exceptions that can
    # be caught and checked.
    target_compile_definitions(InsideLoopUnitTest PRIVATE IL_UNIT_TEST)
endif()

################################################################################
# For Benchmarks
################################################################################
if(IL_BENCHMARK)
    foreach(p
            CMP0054 # CMake 3.1
            CMP0056 # export EXE_LINKER_FLAGS to try_run
            )
        if(POLICY ${p})
            cmake_policy(SET ${p} NEW)
        endif()
    endforeach()

    option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." ON)
    option(BENCHMARK_ENABLE_EXCEPTIONS "Enable the use of exceptions in the benchmark library." ON)
    option(BENCHMARK_ENABLE_LTO "Enable link time optimisation of the benchmark library." OFF)
    option(BENCHMARK_USE_LIBCXX "Build and test using libc++ as the standard library." OFF)
    option(BENCHMARK_BUILD_32_BITS "Build a 32 bit version of the library" OFF)

    # Make sure we can import out CMake functions
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

    # Read the git tags to determine the project version
    include(GetGitVersion)
    get_git_version(GIT_VERSION)

    # Tell the user what versions we are using
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" VERSION ${GIT_VERSION})
    message("-- Version: ${VERSION}")

    # The version of the libraries
    set(GENERIC_LIB_VERSION ${VERSION})
    string(SUBSTRING ${VERSION} 0 1 GENERIC_LIB_SOVERSION)

    # Import our CMake modules
    include(CheckCXXCompilerFlag)
    include(AddCXXCompilerFlag)
    include(CXXFeatureCheck)

   add_cxx_compiler_flag(-wd654)
    add_cxx_compiler_flag(-Wthread-safety)
    if (HAVE_CXX_FLAG_WTHREAD_SAFETY)
        cxx_feature_check(THREAD_SAFETY_ATTRIBUTES)
    endif()

    # On most UNIX like platforms g++ and clang++ define _GNU_SOURCE as a
    # predefined macro, which turns on all of the wonderful libc extensions.
    # However g++ doesn't do this in Cygwin so we have to define it ourselfs
    # since we depend on GNU/POSIX/BSD extensions.
    if (CYGWIN)
        add_definitions(-D_GNU_SOURCE=1)
    endif()

    # Coverage build type
    set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING
        "Flags used by the C++ compiler during coverage builds."
        FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE
        "${CMAKE_EXE_LINKER_FLAGS_DEBUG}" CACHE STRING
        "Flags used for linking binaries during coverage builds."
        FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE
        "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}" CACHE STRING
        "Flags used by the shared libraries linker during coverage builds."
        FORCE)
    mark_as_advanced(
        CMAKE_CXX_FLAGS_COVERAGE
        CMAKE_EXE_LINKER_FLAGS_COVERAGE
        CMAKE_SHARED_LINKER_FLAGS_COVERAGE)
    set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Coverage."
        FORCE)
    add_cxx_compiler_flag(--coverage COVERAGE)

    if (BENCHMARK_USE_LIBCXX)
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            add_cxx_compiler_flag(-stdlib=libc++)
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            add_cxx_compiler_flag(-nostdinc++)
            message("libc++ header path must be manually specified using CMAKE_CXX_FLAGS")
            # Adding -nodefaultlibs directly to CMAKE_<TYPE>_LINKER_FLAGS will break
            # configuration checks such as 'find_package(Threads)'
            list(APPEND BENCHMARK_CXX_LINKER_FLAGS -nodefaultlibs)
            # -lc++ cannot be added directly to CMAKE_<TYPE>_LINKER_FLAGS because
            # linker flags appear before all linker inputs and -lc++ must appear after.
            list(APPEND BENCHMARK_CXX_LIBRARIES c++)
        else()
            message(FATAL "-DBENCHMARK_USE_LIBCXX:BOOL=ON is not supported for compiler")
        endif()
    endif(BENCHMARK_USE_LIBCXX)

    # C++ feature checks
    # Determine the correct regular expression engine to use
    cxx_feature_check(STD_REGEX)
    cxx_feature_check(GNU_POSIX_REGEX)
    cxx_feature_check(POSIX_REGEX)
    if(NOT HAVE_STD_REGEX AND NOT HAVE_GNU_POSIX_REGEX AND NOT HAVE_POSIX_REGEX)
        message(FATAL_ERROR "Failed to determine the source files for the regular expression backend")
    endif()
    if (NOT BENCHMARK_ENABLE_EXCEPTIONS AND HAVE_STD_REGEX AND NOT HAVE_GNU_POSIX_REGEX AND NOT HAVE_POSIX_REGEX)
        message(WARNING "Using std::regex with exceptions disabled is not fully supported")
    endif()
    cxx_feature_check(STEADY_CLOCK)
    # Ensure we have pthreads
    find_package(Threads REQUIRED)

    set(BENCHMARK_FILES
        gbenchmark/src/benchmark.cc
        gbenchmark/src/benchmark_register.cc
        gbenchmark/src/colorprint.cc
        gbenchmark/src/commandlineflags.cc
        gbenchmark/src/complexity.cc
        gbenchmark/src/console_reporter.cc
        gbenchmark/src/counter.cc
        gbenchmark/src/csv_reporter.cc
        gbenchmark/src/json_reporter.cc
        gbenchmark/src/reporter.cc
        gbenchmark/src/sleep.cc
        gbenchmark/src/string_util.cc
        gbenchmark/src/sysinfo.cc
        gbenchmark/src/timers.cc
        il/container/hash/_benchmark/Map_il_vs_std_benchmark.h
        il/container/string/_benchmark/String_benchmark.h
        il/container/string/_benchmark/String_il_vs_std_benchmark.h
        il/linearAlgebra/matrixFree/solver/Gmres.h
        il/File.h il/container/tree/Tree.h il/container/hash/_benchmark/town_benchmark.h il/linearAlgebra/matrixFree/solver/Cg.h il/Cg.h)

    # C++ feature checks
    # Determine the correct regular expression engine to use
    cxx_feature_check(STD_REGEX)
    cxx_feature_check(GNU_POSIX_REGEX)
    cxx_feature_check(POSIX_REGEX)
    if(NOT HAVE_STD_REGEX AND NOT HAVE_GNU_POSIX_REGEX AND NOT HAVE_POSIX_REGEX)
        message(FATAL_ERROR "Failed to determine the source files for the regular expression backend")
    endif()
    if (NOT BENCHMARK_ENABLE_EXCEPTIONS AND HAVE_STD_REGEX
            AND NOT HAVE_GNU_POSIX_REGEX AND NOT HAVE_POSIX_REGEX)
        message(WARNING "Using std::regex with exceptions disabled is not fully supported")
    endif()
    cxx_feature_check(STEADY_CLOCK)
    # Ensure we have pthreads
    find_package(Threads REQUIRED)

    # For TBB
    if (IL_TBB)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIL_TBB")
        #    target_link_libraries(InsideLoop tbb)
    endif()

    # For Cilk
    if (IL_CILK)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIL_CILK")
    endif()


    add_executable(InsideLoopBenchmark ${SOURCE_FILES} ${BENCHMARK_FILES} benchmark.cpp)
    target_include_directories(InsideLoopBenchmark PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/gbenchmark/include)
    target_link_libraries(InsideLoopBenchmark PRIVATE InsideLoop)
endif()
