#include <iostream>
#include <stdexcept>
#include <sycl/sycl.hpp>

#include "xetla.h"
//#include "xetla_arch.h"
#include "cnpy.h"

#include "moe_gemm_kernel_impl.hpp"
#include "moe_gemm_policy.hpp"
#include "launch_kernels.hpp"

void init_rows_for_expert(int *total_rows_for_each_expert_h, int M, int expert_num) {
    total_rows_for_each_expert_h[0] = M;
    for (int i = 1; i < expert_num; ++i) {
        total_rows_for_each_expert_h[i] = 0;
    }
}

void test_group_gemm() {
    using T = sycl::ext::oneapi::bfloat16;
    using Policy = MoEGEMMPolicy;
    queue q;
    const T * activation = nullptr;
    const uint32_t * weights = nullptr;
    const T *scale = nullptr;
    T * outputs = nullptr;
    const int total_m = 1;
    const int gemm_n = 1;
    const int gemm_k = 1;
    const int expert_num = 32;
    std::string data_file = "int4.npz";
    cnpy::NpyArray x_npy = cnpy::npz_load(data_file, "x");
    cnpy::NpyArray w_npy = cnpy::npz_load(data_file, "w");
    cnpy::NpyArray y_npy = cnpy::npz_load(data_file, "y");
    cnpy::NpyArray s_npy = cnpy::npz_load(data_file, "s");
    cnpy::NpyArray shape_npy = cnpy::npz_load(data_file, "shape");
    int M = shape_npy.data<int64_t>()[0];
    int N = shape_npy.data<int64_t>()[1];
    int K = shape_npy.data<int64_t>()[2];
    int group_size = shape_npy.data<int64_t>()[3];

    T * x_dev = malloc_device<T>(x_npy.num_vals, q);
    uint32_t * w_dev = malloc_device<uint32_t>(w_npy.num_vals, q); // int32 for int4
    T * y_dev = malloc_device<T>(y_npy.num_vals, q);
    T * y_test = malloc_host<T>(y_npy.num_vals, q);
    T * s_dev = malloc_device<T>(s_npy.num_vals, q);
    int *total_rows_for_each_expert_h = malloc_host<int>(expert_num, q);
    int *total_rows_for_each_expert = malloc_device<int>(expert_num, q);

    init_rows_for_expert(total_rows_for_each_expert_h, M, expert_num);
    q.copy(x_npy.data<uint16_t>(), (uint16_t *)x_dev, x_npy.num_vals);
    q.copy(w_npy.data<uint32_t>(), (uint32_t *)w_dev, w_npy.num_vals);
    // q.copy(y_npy.data<uint16_t>(), (uint16_t *)y_dev, y_npy.num_vals);
    q.copy(s_npy.data<uint16_t>(), (uint16_t *)s_dev, s_npy.num_vals);
    q.copy(total_rows_for_each_expert_h, total_rows_for_each_expert, expert_num);
    q.wait();
    if (group_size == 128) {
        printf("Launch groupsize=128\n");
        constexpr int GS = 128;
        auto funcs = LaunchMoEGEMMINT4<T, Policy, GS>(
            q, x_dev, w_dev, s_dev, y_dev,
            M, N, K, total_rows_for_each_expert,
            total_rows_for_each_expert_h, expert_num);
        DPCPP_Q_SUBMIT_CGFS(q, funcs);
        q.wait();
        q.copy(y_dev, y_test, y_npy.num_vals);
        q.wait();
    }

    for (int i = 0; i < 16; ++i) {
        printf("%f ", (float)y_test[i]);
    }
    printf("\n");
    for (int i = 0; i < 16; ++i) {
        printf("%f ", (float)y_npy.data<T>()[i]);
    }
    printf("\n");
}

int main() {
    test_group_gemm();
}
