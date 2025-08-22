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
    // using T = sycl::half;
    using T = sycl::ext::oneapi::bfloat16;
    using Policy = MoEGEMMPolicy;
    queue q;
    const T * activation = nullptr;
    const uint32_t * weights = nullptr;
    const T *scale = nullptr;
    T * outputs = nullptr;
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

    T * x_dev = malloc_shared<T>(x_npy.num_vals, q);
    uint32_t * w_dev = malloc_shared<uint32_t>(w_npy.num_vals, q); // int32 for int4
    T * y_dev = malloc_shared<T>(y_npy.num_vals, q);
    T * y_test = malloc_host<T>(y_npy.num_vals, q);
    T * s_dev = malloc_shared<T>(s_npy.num_vals, q);
    int *total_rows_for_each_expert_h = malloc_host<int>(expert_num, q);
    int *total_rows_for_each_expert = malloc_shared<int>(expert_num, q);
    int *offset_rows_for_each_expert = malloc_shared<int>(expert_num, q);
    int *offset_rows_for_each_expert_h = malloc_host<int>(expert_num, q);

    init_rows_for_expert(total_rows_for_each_expert_h, M, expert_num);
    q.copy(x_npy.data<uint16_t>(), (uint16_t *)x_dev, x_npy.num_vals);
    q.copy(w_npy.data<uint32_t>(), (uint32_t *)w_dev, w_npy.num_vals);
    // q.copy(y_npy.data<uint16_t>(), (uint16_t *)y_dev, y_npy.num_vals);
    q.copy(s_npy.data<uint16_t>(), (uint16_t *)s_dev, s_npy.num_vals);
    q.copy(total_rows_for_each_expert_h, total_rows_for_each_expert, expert_num);
    q.wait();
    if (group_size == 128) {
        constexpr int GS = 128;
        printf("Launch groupsize=%d\n", GS);
        auto funcs = LaunchMoEGEMMINT4<T, Policy, GS>(
            q, x_dev, w_dev, s_dev, y_dev,
            M, N, K, total_rows_for_each_expert,
            total_rows_for_each_expert_h,
            offset_rows_for_each_expert,
            expert_num);
        DPCPP_Q_SUBMIT_CGFS(q, funcs);
        q.wait();
        q.copy(y_dev, y_test, y_npy.num_vals);
        q.copy(offset_rows_for_each_expert, offset_rows_for_each_expert_h, expert_num);
        q.wait();
    } else if (group_size == 256) {
        constexpr int GS = 256;
        printf("Launch groupsize=%d\n", GS);
        auto funcs = LaunchMoEGEMMINT4<T, Policy, GS>(
            q, x_dev, w_dev, s_dev, y_dev,
            M, N, K, total_rows_for_each_expert,
            total_rows_for_each_expert_h,
            offset_rows_for_each_expert,
            expert_num);
        DPCPP_Q_SUBMIT_CGFS(q, funcs);
        q.wait();
        q.copy(y_dev, y_test, y_npy.num_vals);
        q.copy(offset_rows_for_each_expert, offset_rows_for_each_expert_h, expert_num);
        q.wait();
    } else if (group_size == 64) {
        constexpr int GS = 64;
        printf("Launch groupsize=%d\n", GS);
        auto funcs = LaunchMoEGEMMINT4<T, Policy, GS>(
            q, x_dev, w_dev, s_dev, y_dev,
            M, N, K, total_rows_for_each_expert,
            total_rows_for_each_expert_h,
            offset_rows_for_each_expert,
            expert_num);
        DPCPP_Q_SUBMIT_CGFS(q, funcs);
        q.wait();
        q.copy(y_dev, y_test, y_npy.num_vals);
        q.copy(offset_rows_for_each_expert, offset_rows_for_each_expert_h, expert_num);
        q.wait();
    }
    printf("test:\n");
    sycl::half * y_half = reinterpret_cast<sycl::half *>(y_test);
    sycl::ext::oneapi::bfloat16 * y_bf16 = reinterpret_cast<sycl::ext::oneapi::bfloat16 *>(y_test);
    for (int i = 0; i < 16; ++i) {
        printf("%7.4f, ", (float)y_half[i]);
    }
    printf("\n");
    for (int i = 0; i < 16; ++i) {
        printf("%7.4f, ", (float)y_bf16[i]);
    }
    printf("\n");
    for (int m = 0; m < std::min(64, M); ++m) {
        for (int n = 0; n < 128; ++n) {
            if (n % 16 == 0)
                printf("(%03d,%03d): ", m, n);
            printf("%7.4f, ", (float)y_test[m * N + n]);
            if (n % 16 == 15)
                printf("\n");
        }
        printf("\n");
    }
    printf("\nrefe:\n");
    for (int m = 0; m < std::min(64, M); ++m) {
        for (int n = 0; n < 128; ++n) {
            if (n % 16 == 0)
                printf("(%03d,%03d): ", m, n);
            printf("%7.4f, ", (float)y_npy.data<T>()[m * N + n]);
            if (n % 16 == 15)
                printf("\n");
        }
        printf("\n");
    }
    printf("\n");
    // T * bf16_ptr=reinterpret_cast<T *>(offset_rows_for_each_expert_h);
    // for (int i = 0; i < expert_num; ++i) {
    //     printf("%f ", (float)(bf16_ptr[i]));
    // }
    // printf("\n");
}

int main() {
    test_group_gemm();
}
