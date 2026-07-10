#pragma once

// GCC/Clang define this as 1 under -ffinite-math-only (included by
// -ffast-math).  The ping-pong runtime's safety contract explicitly depends on
// NaN/Inf remaining observable.  Fail compilation if a target-local flag later
// re-enables the unsafe optimization despite the top-level CMake setting.
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0
#error "A3 deploy safety runtime must not be compiled with -ffast-math/-ffinite-math-only"
#endif
