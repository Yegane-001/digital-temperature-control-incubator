import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_core import (K_PLANT, TAU_PLANT, PIDController, run_scenario,
                       compute_metrics, T_END)

K = K_PLANT
TAU = TAU_PLANT

Kp, Ki = 0.24667, 0.010974


def make_controller():
    return PIDController(Kp=Kp, Ki=Ki, Kd=0.0, u_min=0.0, u_max=1.0,
                          anti_windup=True)


def plot_scenario(res, title, fname, event_lines=None):
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axs[0].plot(res["t"], res["T"], label="دمای محفظه T(t)", color="tab:red")
    axs[0].plot(res["t"], res["sp"], label="دمای مطلوب (Setpoint)",
                color="black", linestyle="--")
    axs[0].set_ylabel("دما (°C)")
    axs[0].legend(loc="lower right")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(res["t"], res["u"] * 100.0, color="tab:orange")
    axs[1].set_ylabel("سیگنال کنترلی u (%)")
    axs[1].set_ylim(-5, 105)
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(res["t"], res["e"], color="tab:blue")
    axs[2].set_ylabel("خطا e(t) = SP - T")
    axs[2].set_xlabel("زمان (s)")
    axs[2].grid(True, alpha=0.3)

    if event_lines:
        for ax in axs:
            for tv, lbl in event_lines:
                ax.axvline(tv, color="gray", linestyle=":", linewidth=1)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


results = {}
metrics = {}

# ---------------------------------------------------------------------------
# Scenario 1: T0=25 -> setpoint 40, constant ambient, no disturbance
# ---------------------------------------------------------------------------
ctrl1 = make_controller()
res1 = run_scenario(
    setpoint_fn=lambda t: 40.0,
    ambient_fn=lambda t: 25.0,
    tau_fn=lambda t: TAU,
    controller=ctrl1, T0=25.0, K=K,
)
m1 = compute_metrics(res1["t"], res1["T"], sp_final=40.0, sp_initial=25.0,
                      step_start_time=0.0)
results["s1"] = res1
metrics["s1"] = m1
plot_scenario(res1, "سناریو ۱: پاسخ پله (T0=25°C \u2192 SP=40°C)",
              "scenario1.png")

# ---------------------------------------------------------------------------
# Scenario 2: baseline + door opens for 10s at t=120s (extra heat loss)
# ---------------------------------------------------------------------------
def extra_loss_fn_door(t):
    if 120.0 <= t <= 130.0:
        return TAU / 3.0   # extra parallel heat-loss path while door is open
    return None

ctrl2 = make_controller()
res2 = run_scenario(
    setpoint_fn=lambda t: 40.0,
    ambient_fn=lambda t: 25.0,
    tau_fn=lambda t: TAU,
    controller=ctrl2, T0=25.0, K=K,
    extra_loss_tau_fn=extra_loss_fn_door,
)
results["s2"] = res2
# metric of interest: max temperature dip and recovery (settling) after the
# disturbance ends at t=130
dip_mask = (res2["t"] >= 120) & (res2["t"] <= 160)
min_T_during = res2["T"][dip_mask].min()
recover_mask = res2["t"] >= 130
after = res2["T"][recover_mask]
t_after = res2["t"][recover_mask]
band = 0.02 * 40.0  # 2% of setpoint magnitude as tolerance band (deg C)
outside = np.where(np.abs(after - 40.0) > band)[0]
recovery_time = (t_after[outside[-1]] - 130.0) if len(outside) else 0.0
metrics["s2"] = dict(min_temperature_dip=min_T_during,
                      max_deviation=40.0 - min_T_during,
                      recovery_time_after_disturbance=recovery_time,
                      steady_state_error=40.0 - res2["T"][-1])
plot_scenario(res2, "سناریو ۲: باز شدن در محفظه در t=120s به مدت ۱۰ ثانیه",
              "scenario2.png", event_lines=[(120, "باز شدن در"), (130, "بسته شدن در")])

# ---------------------------------------------------------------------------
# Scenario 3: ambient temperature drop 25 -> 15 at t=150s
# ---------------------------------------------------------------------------
def amb_fn_step(t):
    return 25.0 if t < 150.0 else 15.0

ctrl3 = make_controller()
res3 = run_scenario(
    setpoint_fn=lambda t: 40.0,
    ambient_fn=amb_fn_step,
    tau_fn=lambda t: TAU,
    controller=ctrl3, T0=25.0, K=K,
)
results["s3"] = res3
m3 = compute_metrics(res3["t"], res3["T"], sp_final=40.0, sp_initial=40.0,
                      step_start_time=150.0)
# for a disturbance rejection, "step" in metric sense doesn't apply the same
# way; report max deviation & recovery/settling similarly to scenario 2
dist_mask = res3["t"] >= 150
after3 = res3["T"][dist_mask]
t_after3 = res3["t"][dist_mask]
max_dev3 = (40.0 - after3).max()
band3 = 0.02 * 40.0
outside3 = np.where(np.abs(after3 - 40.0) > band3)[0]
recovery3 = (t_after3[outside3[-1]] - 150.0) if len(outside3) else 0.0
metrics["s3"] = dict(max_deviation=max_dev3, recovery_time=recovery3,
                      steady_state_error=40.0 - res3["T"][-1])
plot_scenario(res3, "سناریو ۳: کاهش دمای محیط از ۲۵ به ۱۵ درجه در t=150s",
              "scenario3.png", event_lines=[(150, "افت دمای محیط")])

# ---------------------------------------------------------------------------
# Scenario 4: setpoint change 40 -> 60 at t=150s
# ---------------------------------------------------------------------------
def sp_fn_step(t):
    return 40.0 if t < 150.0 else 60.0

ctrl4 = make_controller()
res4 = run_scenario(
    setpoint_fn=sp_fn_step,
    ambient_fn=lambda t: 25.0,
    tau_fn=lambda t: TAU,
    controller=ctrl4, T0=25.0, K=K,
)
results["s4"] = res4
m4 = compute_metrics(res4["t"], res4["T"], sp_final=60.0, sp_initial=40.0,
                      step_start_time=150.0)
metrics["s4"] = m4
plot_scenario(res4, "سناریو ۴: تغییر ست‌پوینت از ۴۰ به ۶۰ درجه در t=150s",
              "scenario4.png", event_lines=[(150, "تغییر ست‌پوینت")])

print("=== Scenario 1 metrics ===")
print(metrics["s1"])
print("=== Scenario 2 metrics ===")
print(metrics["s2"])
print("=== Scenario 3 metrics ===")
print(metrics["s3"])
print("=== Scenario 4 metrics ===")
print(metrics["s4"])

np.save("results_summary.npy", metrics, allow_pickle=True)
