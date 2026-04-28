#pragma once
/*
 * C ABI for ezfalcon force objects.
 * Capsule name: "ezfalcon_force_v1"
 */

#ifdef _MSC_VER
#  define EZFALCON_RESTRICT __restrict
#else
#  define EZFALCON_RESTRICT __restrict__
#endif

typedef void (*ezfalcon_force_fn)(
    int      n,
    const double * EZFALCON_RESTRICT pos,   /* (n, 3) row-major */
    const double * EZFALCON_RESTRICT vel,   /* (n, 3); may be NULL for conservative-only */
    const double * EZFALCON_RESTRICT mass,  /* (n,) */
    double   t,
    void    *params,
    double  * EZFALCON_RESTRICT acc_out,    /* (n, 3); accumulate (+=) — driver zeroes */
    double  * EZFALCON_RESTRICT pot_out     /* (n,) or NULL; accumulate */
);

typedef struct {
    ezfalcon_force_fn fn;
    void *params;
    int   conservative;           /* 1 -> can fill pot_out, 0 -> velocity-dependent */
    void (*free_params)(void *);  /* optional destructor; may be NULL */
} ezfalcon_force_t;

#define EZFALCON_FORCE_CAPSULE_NAME "ezfalcon_force_v1"
