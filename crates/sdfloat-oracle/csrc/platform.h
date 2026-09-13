/* Platform header for the softfloat-oracle build (little-endian, 64-bit,
   GCC/Clang). Mirrors SoftFloat's build/Linux-x86_64-GCC/platform.h with a
   real thread-local for the rounding-mode global so parallel test threads do
   not race on it. */
#define LITTLEENDIAN 1
#define INLINE inline
#define THREAD_LOCAL _Thread_local
#define SOFTFLOAT_BUILTIN_CLZ 1
#define SOFTFLOAT_INTRINSIC_INT128 1
#include "opts-GCC.h"
