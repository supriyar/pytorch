#pragma once

#include "qnnpack.h"
#include <ATen/ATen.h>
#include <thread>

static pthreadpool_t qnnpack_threadpool_ = nullptr;

static pthreadpool_t qnnpack_threadpool() {
  unsigned int threads;
  #ifdef INTRA_OP_PARALLEL
      threads = at::get_num_threads();
  #else
      threads = std::thread::hardware_concurrency();
  #endif
  qnnpack_threadpool_ = pthreadpool_create(threads);
  if (qnnpack_threadpool_ == nullptr) {
    throw std::runtime_error("could not initialize QNNPack's pthreadpool");
  }
  std::cout << "Num threads " << threads <<std::endl;
  return qnnpack_threadpool_;
}

enum class Activation : uint8_t { NONE = 0, RELU = 1 };

#if defined(__ANDROID__) && !defined(__NDK_MAJOR__)
template <class T>
inline float Round(const float x) {
  return ::nearbyintf(x);
}
inline double Round(const double x) {
  return ::nearbyint(x);
}
#else
template <class T>
inline T Round(const T x) {
  return std::nearbyint(x);
}
#endif


inline uint8_t QuantizeUint8(float scale, int32_t zero_point, float value) {
  const int32_t qmin = std::numeric_limits<uint8_t>::min();
  const int32_t qmax = std::numeric_limits<uint8_t>::max();
  auto r = zero_point + static_cast<int32_t>(Round(value / scale));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return static_cast<uint8_t>(r);
}


inline std::pair<uint8_t, uint8_t>
activationLimits(float scale, int32_t zero_point, Activation Ac) {
  switch (Ac) {
    case Activation::NONE:
      return {std::numeric_limits<uint8_t>::min(),
              std::numeric_limits<uint8_t>::max()};
    case Activation::RELU:
      return {QuantizeUint8(scale, zero_point, 0.0),
              std::numeric_limits<uint8_t>::max()};
    default:
#ifdef _MSC_VER
      __assume(0);
#else
      __builtin_unreachable();
#endif
  }
}
