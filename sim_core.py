"""
Core simulation module for the Digital Incubator Temperature Control project.

Plant (continuous):
    dT/dt = -(T - T_ambient)/tau + (K/tau)*u ,   u in [0,1]

Controller: digital PI (optionally PID) with anti-windup by conditional
integration (clamping). Runs at Ts = 0.1 s (zero-order hold on u).
Plant integrated with RK4 at a finer step (dt_sim) for numerical accuracy.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Plant parameters (fixed, chosen inside the allowed ranges)
# ---------------------------------------------------------------------------
K_PLANT = 50.0      # deg C   (range: 20-80)
TAU_PLANT = 100.0   # s       (range: 30-180)

TS = 0.1            # controller sampling period [s]
DT_SIM = 0.01        # integration sub-step [s]  (10 substeps per Ts)
T_END = 300.0        # total simulation time [s]


def plant_derivative(T, T_ambient, u, tau, K, extra_loss_tau=None):
    """dT/dt for the first order thermal model.

    extra_loss_tau: optional second, parallel heat-loss path (e.g. an open
    incubator door). Physically this models an additional conduction path
    to the ambient that does NOT change the heater's effective gain
    (unlike simply shrinking `tau`, which would also -- incorrectly --
    boost K/tau).
    """
    base = -(T - T_ambient) / tau + (K / tau) * u
    if extra_loss_tau is not None:
        base += -(T - T_ambient) / extra_loss_tau
    return base


def rk4_step(T, T_ambient, u, tau, K, dt, extra_loss_tau=None):
    k1 = plant_derivative(T, T_ambient, u, tau, K, extra_loss_tau)
    k2 = plant_derivative(T + 0.5 * dt * k1, T_ambient, u, tau, K, extra_loss_tau)
    k3 = plant_derivative(T + 0.5 * dt * k2, T_ambient, u, tau, K, extra_loss_tau)
    k4 = plant_derivative(T + dt * k3, T_ambient, u, tau, K, extra_loss_tau)
    return T + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class PIDController:
    """Digital PID (PI when Kd=0) with clamping anti-windup.

    Position form:
        u = Kp*e + Ki*Ts*sum(e)  + Kd*(e - e_prev)/Ts
    Anti-windup: integral term is only updated when the controller is not
    saturated, OR when the update would move the output back into range
    (conditional integration / clamping method).
    """

    def __init__(self, Kp, Ki, Kd=0.0, Ts=TS, u_min=0.0, u_max=1.0,
                 anti_windup=True):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ts = Ts
        self.u_min = u_min
        self.u_max = u_max
        self.anti_windup = anti_windup
        self.integral = 0.0
        self.e_prev = 0.0
        self.first = True

    def reset(self):
        self.integral = 0.0
        self.e_prev = 0.0
        self.first = True

    def step(self, error):
        # proposed integral update
        integral_candidate = self.integral + error * self.Ts

        derivative = 0.0 if self.first else (error - self.e_prev) / self.Ts

        # unsaturated output using the candidate integral
        u_unsat = (self.Kp * error + self.Ki * integral_candidate
                   + self.Kd * derivative)

        if self.anti_windup:
            if self.u_min <= u_unsat <= self.u_max:
                self.integral = integral_candidate
            else:
                # saturating: only let the integral move if it would pull
                # the output back toward the allowed range (conditional
                # integration), otherwise freeze it.
                if (u_unsat > self.u_max and error < 0) or \
                   (u_unsat < self.u_min and error > 0):
                    self.integral = integral_candidate
                # else: freeze integral (do not accumulate further windup)
        else:
            self.integral = integral_candidate

        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        u_sat = min(max(u, self.u_min), self.u_max)

        self.e_prev = error
        self.first = False
        return u_sat, u  # return saturated and raw (for logging)


def moving_average_filter(window):
    buf = []

    def filt(x):
        buf.append(x)
        if len(buf) > window:
            buf.pop(0)
        return sum(buf) / len(buf)
    return filt


def low_pass_filter(alpha):
    state = {"y": None}

    def filt(x):
        if state["y"] is None:
            state["y"] = x
        else:
            state["y"] = alpha * x + (1 - alpha) * state["y"]
        return state["y"]
    return filt


def run_scenario(setpoint_fn, ambient_fn, tau_fn, controller, T0=25.0,
                  t_end=T_END, Ts=TS, dt_sim=DT_SIM, K=K_PLANT,
                  noise_std=0.0, filt=None, seed=0, extra_loss_tau_fn=None):
    """Run one closed-loop simulation.

    setpoint_fn(t), ambient_fn(t), tau_fn(t) -> scalars at time t.
    controller: PIDController instance (already configured / reset).
    noise_std: std-dev of Gaussian measurement noise added to T before it
               reaches the controller (0 = no noise).
    filt: optional filter callable applied to the noisy measurement before
          it is fed to the controller (e.g. moving_average_filter(5)).
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(t_end / Ts))
    substeps = int(round(Ts / dt_sim))

    t_log = np.zeros(n_steps + 1)
    T_log = np.zeros(n_steps + 1)
    Tmeas_log = np.zeros(n_steps + 1)
    sp_log = np.zeros(n_steps + 1)
    u_log = np.zeros(n_steps + 1)
    e_log = np.zeros(n_steps + 1)

    T = T0
    t = 0.0
    u = 0.0

    for k in range(n_steps + 1):
        sp = setpoint_fn(t)
        T_noisy = T + (rng.normal(0, noise_std) if noise_std > 0 else 0.0)
        T_meas = filt(T_noisy) if filt is not None else T_noisy
        error = sp - T_meas

        u_sat, _ = controller.step(error)
        u = u_sat

        t_log[k] = t
        T_log[k] = T
        Tmeas_log[k] = T_meas
        sp_log[k] = sp
        u_log[k] = u
        e_log[k] = sp - T  # true error (based on true T) for analysis

        if k == n_steps:
            break

        # integrate plant over this control period with ZOH on u
        for _ in range(substeps):
            T_amb = ambient_fn(t)
            tau = tau_fn(t)
            extra_tau = extra_loss_tau_fn(t) if extra_loss_tau_fn else None
            T = rk4_step(T, T_amb, u, tau, K, dt_sim, extra_tau)
            t += dt_sim

    return {
        "t": t_log, "T": T_log, "Tmeas": Tmeas_log, "sp": sp_log,
        "u": u_log, "e": e_log,
    }


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(t, T, sp_final, sp_initial, step_start_time=0.0,
                     band=0.02):
    """Compute rise time, settling time, overshoot, steady-state error
    for a step change from sp_initial to sp_final occurring at
    step_start_time."""
    delta = sp_final - sp_initial
    mask = t >= step_start_time
    tt = t[mask]
    TT = T[mask]
    t0 = step_start_time

    if abs(delta) < 1e-9:
        return dict(rise_time=None, settling_time=None, overshoot=None,
                    steady_state_error=(sp_final - T[-1]))

    # Rise time: 10% -> 90% of the step (works for increasing or decreasing)
    lo = sp_initial + 0.1 * delta
    hi = sp_initial + 0.9 * delta

    def first_crossing(level):
        if delta > 0:
            idx = np.where(TT >= level)[0]
        else:
            idx = np.where(TT <= level)[0]
        return tt[idx[0]] - t0 if len(idx) else None

    t_lo = first_crossing(lo)
    t_hi = first_crossing(hi)
    rise_time = (t_hi - t_lo) if (t_lo is not None and t_hi is not None) else None

    # Overshoot (%): relative to the size of the step
    if delta > 0:
        peak = TT.max()
        overshoot = max(0.0, (peak - sp_final) / abs(delta) * 100.0)
    else:
        peak = TT.min()
        overshoot = max(0.0, (sp_final - peak) / abs(delta) * 100.0)

    # Settling time: last time the response leaves the +-band*|delta| tube
    # around the final value (using band of the step size, common convention)
    tube = band * abs(delta)
    outside = np.where(np.abs(TT - sp_final) > tube)[0]
    settling_time = (tt[outside[-1]] - t0) if len(outside) else 0.0

    steady_state_error = sp_final - T[-1]

    return dict(rise_time=rise_time, settling_time=settling_time,
                overshoot=overshoot, steady_state_error=steady_state_error)
