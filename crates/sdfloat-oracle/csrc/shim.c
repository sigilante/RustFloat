/* Flat C entry points for the Rust oracle. Every function takes the rounding
   mode explicitly and stores it into SoftFloat's (thread-local) global before
   the call, so the Rust side never touches C globals. Float values cross the
   boundary as plain unsigned integers; f128 as a little-word-order pair. */
#include <stdint.h>
#include "platform.h"   /* THREAD_LOCAL must match softfloat_state.c */
#include "softfloat.h"

#define SET(m) (softfloat_roundingMode = (uint_fast8_t)(m))

#define BIN(W, T, OP) \
  T orc_f##W##_##OP(uint8_t m, T a, T b) { \
    SET(m); return f##W##_##OP((float##W##_t){a}, (float##W##_t){b}).v; }
#define UN(W, T, OP) \
  T orc_f##W##_##OP(uint8_t m, T a) { \
    SET(m); return f##W##_##OP((float##W##_t){a}).v; }
#define CMP(W, T, OP) \
  int orc_f##W##_##OP(T a, T b) { \
    return f##W##_##OP((float##W##_t){a}, (float##W##_t){b}); }
#define TOINT(W, T, IT, IW) \
  IT orc_f##W##_to_i##IW(uint8_t m, T a) { \
    return f##W##_to_i##IW((float##W##_t){a}, m, 0); }
#define FROMINT(W, T, IT, IN) \
  T orc_##IN##_to_f##W(uint8_t m, IT a) { \
    SET(m); return IN##_to_f##W(a).v; }
#define CVT(WA, TA, WB, TB) \
  TB orc_f##WA##_to_f##WB(uint8_t m, TA a) { \
    SET(m); return f##WA##_to_f##WB((float##WA##_t){a}).v; }

#define SCALAR(W, T) \
  BIN(W, T, add) BIN(W, T, sub) BIN(W, T, mul) BIN(W, T, div) \
  UN(W, T, sqrt) \
  T orc_f##W##_round_to_int(uint8_t m, T a) { \
    return f##W##_roundToInt((float##W##_t){a}, m, 0).v; } \
  CMP(W, T, eq) CMP(W, T, lt) CMP(W, T, le) \
  TOINT(W, T, int32_t, 32) TOINT(W, T, int64_t, 64) \
  FROMINT(W, T, int32_t, i32) FROMINT(W, T, int64_t, i64) \
  FROMINT(W, T, uint32_t, ui32) FROMINT(W, T, uint64_t, ui64)

SCALAR(16, uint16_t)
SCALAR(32, uint32_t)
SCALAR(64, uint64_t)

CVT(16, uint16_t, 32, uint32_t) CVT(16, uint16_t, 64, uint64_t)
CVT(32, uint32_t, 16, uint16_t) CVT(32, uint32_t, 64, uint64_t)
CVT(64, uint64_t, 16, uint16_t) CVT(64, uint64_t, 32, uint32_t)

/* f128: in/out as {lo, hi} pairs (LITTLEENDIAN: v[0] is the low word). */
static inline float128_t Q(const uint64_t *p) { float128_t q; q.v[0] = p[0]; q.v[1] = p[1]; return q; }
static inline void PUT(uint64_t *o, float128_t q) { o[0] = q.v[0]; o[1] = q.v[1]; }

#define QBIN(OP) \
  void orc_f128_##OP(uint8_t m, const uint64_t *a, const uint64_t *b, uint64_t *o) { \
    SET(m); PUT(o, f128_##OP(Q(a), Q(b))); }
#define QCMP(OP) \
  int orc_f128_##OP(const uint64_t *a, const uint64_t *b) { return f128_##OP(Q(a), Q(b)); }

QBIN(add) QBIN(sub) QBIN(mul) QBIN(div)
void orc_f128_sqrt(uint8_t m, const uint64_t *a, uint64_t *o) { SET(m); PUT(o, f128_sqrt(Q(a))); }
void orc_f128_round_to_int(uint8_t m, const uint64_t *a, uint64_t *o) { PUT(o, f128_roundToInt(Q(a), m, 0)); }
QCMP(eq) QCMP(lt) QCMP(le)
int32_t orc_f128_to_i32(uint8_t m, const uint64_t *a) { return f128_to_i32(Q(a), m, 0); }
int64_t orc_f128_to_i64(uint8_t m, const uint64_t *a) { return f128_to_i64(Q(a), m, 0); }
void orc_i32_to_f128(uint8_t m, int32_t a, uint64_t *o)  { SET(m); PUT(o, i32_to_f128(a)); }
void orc_i64_to_f128(uint8_t m, int64_t a, uint64_t *o)  { SET(m); PUT(o, i64_to_f128(a)); }
void orc_ui32_to_f128(uint8_t m, uint32_t a, uint64_t *o) { SET(m); PUT(o, ui32_to_f128(a)); }
void orc_ui64_to_f128(uint8_t m, uint64_t a, uint64_t *o) { SET(m); PUT(o, ui64_to_f128(a)); }
void orc_f16_to_f128(uint8_t m, uint16_t a, uint64_t *o) { SET(m); PUT(o, f16_to_f128((float16_t){a})); }
void orc_f32_to_f128(uint8_t m, uint32_t a, uint64_t *o) { SET(m); PUT(o, f32_to_f128((float32_t){a})); }
void orc_f64_to_f128(uint8_t m, uint64_t a, uint64_t *o) { SET(m); PUT(o, f64_to_f128((float64_t){a})); }
uint16_t orc_f128_to_f16(uint8_t m, const uint64_t *a) { SET(m); return f128_to_f16(Q(a)).v; }
uint32_t orc_f128_to_f32(uint8_t m, const uint64_t *a) { SET(m); return f128_to_f32(Q(a)).v; }
uint64_t orc_f128_to_f64(uint8_t m, const uint64_t *a) { SET(m); return f128_to_f64(Q(a)).v; }
