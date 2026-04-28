#pragma once
/*
 * C ABI for ezfalcon integrator step functions.
 * Capsule name: "ezfalcon_integrator_v1"
 */

#include "../forces/_c_force_abi.h"

typedef void (*ezfalcon_step_fn)(
    int n,
    double * EZFALCON_RESTRICT pos,
    double * EZFALCON_RESTRICT vel,
    const double * EZFALCON_RESTRICT mass,
    double t,
    double dt,
    ezfalcon_force_t *self_gravity,          /* may be NULL */
    ezfalcon_force_t *external,              /* may be NULL */
    void *integrator_state,                  /* persistent scratch (e.g. cached prev_acc) */
    double * EZFALCON_RESTRICT acc_total_scratch,   /* (n, 3) scratch */
    double * EZFALCON_RESTRICT acc_self_scratch,    /* (n, 3) scratch — written if self_gravity */
    double * EZFALCON_RESTRICT pot_self_scratch     /* (n,) scratch — written if self_gravity */
);

typedef struct {
    ezfalcon_step_fn  step;
    void             *state;
    void (*free_state)(void *);  /* may be NULL */
} ezfalcon_integrator_t;

#define EZFALCON_INTEGRATOR_CAPSULE_NAME "ezfalcon_integrator_v1"
