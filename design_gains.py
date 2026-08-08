import numpy as np
from sim_core import K_PLANT, TAU_PLANT, PIDController, run_scenario, compute_metrics

K = K_PLANT
tau = TAU_PLANT

# --- Pole placement design -------------------------------------------------
# Plant: G(s) = K/(tau s + 1)
# PI:    C(s) = Kp + Ki/s
# Closed loop char. eq: tau*s^2 + (1+K*Kp)*s + K*Ki = 0
# => s^2 + [(1+K*Kp)/tau] s + (K*Ki/tau) = 0
# match to s^2 + 2*zeta*wn*s + wn^2
zeta = 0.9
Ts_settle_target = 60.0  # desired 2% settling time [s]
wn = 4.0 / (zeta * Ts_settle_target)

Ki = wn**2 * tau / K
Kp = (2 * zeta * wn * tau - 1) / K

print(f"wn={wn:.5f} rad/s")
print(f"Kp={Kp:.5f}")
print(f"Ki={Ki:.6f}")

# --- Quick verification: step from 25 -> 40, constant ambient=25 ----------
ctrl = PIDController(Kp=Kp, Ki=Ki, Kd=0.0, u_min=0.0, u_max=1.0, anti_windup=True)

def sp_fn(t):
    return 40.0

def amb_fn(t):
    return 25.0

def tau_fn(t):
    return tau

res = run_scenario(sp_fn, amb_fn, tau_fn, ctrl, T0=25.0, t_end=300.0, K=K)
m = compute_metrics(res["t"], res["T"], sp_final=40.0, sp_initial=25.0, step_start_time=0.0)
print(m)
print("final T:", res["T"][-1])
print("max u:", res["u"].max())
