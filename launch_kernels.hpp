#pragma once
#define DPCPP_Q_SUBMIT(q, cgf, ...)                                     \
    {                                                                   \
        auto e = (q).submit((cgf), ##__VA_ARGS__);                      \
        (q).throw_asynchronous();                                       \
    }

// utility to submit a list of CGFs
#define DPCPP_Q_SUBMIT_CGFS(q, cgfs, ...)           \
    for (auto& cgf : (cgfs)) {                      \
        DPCPP_Q_SUBMIT((q), cgf, ##__VA_ARGS__);    \
    }

inline float get_exe_time(const sycl::event &e) {
  return (e.template get_profiling_info<
              sycl::info::event_profiling::command_end>() -
          e.template get_profiling_info<
              sycl::info::event_profiling::command_start>()) /
         1000.0f; // us
}
