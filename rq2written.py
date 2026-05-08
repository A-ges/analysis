"""
# RQ2 - Configuration effects on system-level metrics

**Research question:** 

How do different configurations of heterogeneous households
influence aggregate load, flexibility patterns, costs, and overall grid stability
at the system level?

**Data source:** `results/rq2_run_summaries.csv` 

With one row per (population, network_code, seed). With 3 network variants (a, b, c) and
10 seeds per setup

For every actual run there is a corresponding no-shift run with the same
population, same N and same seed (always run on variant "a" since the
baseline is identical across variants). This isolates the effect of
behavioral shifting from random agent-composition effects.

EMD (Earth Mover's Distance) is computed against a matched baseline 
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
import statsmodels.stats.anova as sm_anova

project_root = Path(".")
sys.path.insert(0, str(project_root))

#paths to cached data and figure folder
results_dir = Path("results")
repr_dir = Path("results/representative")
figure_dir = Path("figures/rq2")
agg_curves_dir = Path("RQ2 aggregate curves data")
figure_dir.mkdir(parents=True, exist_ok=True)

#consistent plot style
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5})

#setting default colors and markers for each population
pop_colors = {
    "Habitual": "red",
    "Progressive": "orange",
    "Tipping": "magenta",
    "Balanced": "black",
    "Price-Maximalist": "green",
    "Social-Maximalist": "blue"}

pop_styles = {
    "Habitual": ("solid", "o"),
    "Progressive": ("solid", "s"),
    "Tipping": ("solid", "^"),
    "Balanced": ("solid", "D"),
    "Price-Maximalist": ("dashed", "P"),
    "Social-Maximalist": ("dashed", "X")}

#list of all six populations
pop_order_all = ["Habitual", "Progressive", "Tipping", "Balanced", "Price-Maximalist", "Social-Maximalist"]


metrics_rq2 = [
    "mean_cost_norm",
    "mean_price_advantage",
    "emd_vs_baseline",
    "par_mean",
    "mean_flexibility",
    "mean_adjustment"]
 
metric_titles = {
    "mean_cost_norm": "Mean norm. cost (units/kWh)",
    "mean_price_advantage": "Mean price advantage",
    "emd_vs_baseline": "EMD vs baseline",
    "par_mean": "Peak-to-average ratio",
    "mean_flexibility": "Mean flexibility (hrs/day/agent)",
    "mean_adjustment": "Adjustment day-29 (hrs)"}


#network sizes used in RQ2 (and generated with datagenerator)
n_values = [100, 250, 500, 750, 1000]

#total simulated days
days = 30

#significance threshold used in all tests
alpha = 0.05

print("notebook loaded and ready")

## 2. Load and inspect RQ2 data

#load the main rq2 summary csv
df = pd.read_csv(results_dir / "rq2_run_summaries.csv")
df["mean_flexibility"] = df["total_flexibility"] / df["N"]
 
print("rows:", len(df))
print("columns:", list(df.columns))
print()
print(df.groupby(["pop_label", "N"]).size().rename("n_runs").to_string())

"""
rows: 900
columns: ['pop_label', 'network_code', 'N', 'seed', 'par_mean', 'peak_hour_mean', 'peak_hour_shift', 'mean_cost_norm', 'mean_price_advantage', 'total_flexibility', 'mean_adjustment', 'emd_vs_baseline', 'mean_flexibility']

pop_label          N   
Balanced           100     30
                   250     30
                   500     30
                   750     30
                   1000    30
Habitual           100     30
                   250     30
                   500     30
                   750     30
                   1000    30
Price-Maximalist   100     30
                   250     30
                   500     30
                   750     30
                   1000    30
Progressive        100     30
                   250     30
                   500     30
                   750     30
                   1000    30
Social-Maximalist  100     30
                   250     30
                   500     30
                   750     30
                   1000    30
Tipping            100     30
                   250     30
                   500     30
                   750     30
                   1000    30
"""
## 3. Functions

#same ones as in RQ1
def bootstrap_ci(values, n_boot=2000, ci_level=0.95, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan

    #generate bootstrap means
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boot_means[b] = sample.mean()

    mean_val = float(vals.mean())
    lo = float(np.percentile(boot_means, (1 - ci_level) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci_level) / 2 * 100))
    return mean_val, lo, hi


#cohen's d effect size between two groups
def cohens_d(vals_a, vals_b):
    a = np.array(vals_a, dtype=float)
    b = np.array(vals_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_sd == 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def rank_biserial_r(x, y, u_stat):
    """
    Compute rank-biserial correlation as effect size for Mann-Whitney U.
 
    Parameters:
    -> x: array for group 1
    -> y: array for group 2
    -> u_stat: U statistic from scipy.stats.mannwhitneyu(x, y), which is U1
 
    Returns rank-biserial r in [-1, 1]
    -> r > 0 means x tends to be larger than y
    -> r < 0 means y tends to be larger than x
    """
    n1 = len(x)
    n2 = len(y)
    #r = (U1 - U2) / (n1 * n2) where U2 = n1*n2 - U1
    r = (2 * u_stat - n1 * n2) / (n1 * n2)
    return float(r)
 
 
def epsilon_sq_kw(h_stat, k, n):
    """
    Compute epsilon-squared as effect size for the Kruskal-Wallis test.
 
    Parameters:
    -> h_stat: Kruskal-Wallis H statistic
    -> k: number of groups
    -> n: total number of observations across all groups
 
    Returns epsilon-squared in [0, 1]
    -> >= 0.01 small, >= 0.06 medium, >= 0.14 large (Cohen, 1988 adapted)
    """
    if n <= k:
        return np.nan
    return (h_stat - k + 1) / (n - k)

"""
## 4. Assumption checks

Same logic as RQ1

Test normality and equal variance for each metric across
the six populations at N=500

If assumptions hold for a metric use one-way ANOVA. If they fail use
Kruskal-Wallis instead.
"""

use_parametric = {}
df_500 = df[df["N"] == 500].copy()

print("=== assumption checks at N=500 ===")
print()
 
for metric in metrics_rq2:
    print("--- metric:", metric_titles[metric], "---")
 
    group_data = {}
    for pop in pop_order_all:
        vals = df_500[df_500["pop_label"] == pop][metric].dropna().values
        group_data[pop] = vals
 
    all_normal = True
    print("  shapiro-wilk (H0: normal):")
    for pop in pop_order_all:
        vals = group_data[pop]
        if len(vals) < 3:
            print("   ", pop, ": too few values")
            all_normal = False
            continue
        w_stat, p_sw = stats.shapiro(vals)
        if p_sw < alpha:
            result = "FAIL"
            all_normal = False
        else:
            result = "pass"
        print("   ", pop, ": W=", round(w_stat, 3), ", p=", round(p_sw, 4), "->", result)
 
    all_groups = [group_data[p] for p in pop_order_all if len(group_data[p]) > 0]
    lev_stat, p_lev = stats.levene(*all_groups)
    if p_lev < alpha:
        equal_var = False
        lev_result = "FAIL"
    else:
        equal_var = True
        lev_result = "pass"
    print("  levene (H0: equal variances): W=", round(lev_stat, 3), ", p=", round(p_lev, 4), "->", lev_result)
 
    go_parametric = all_normal and equal_var
    use_parametric[metric] = go_parametric
    if go_parametric:
        print("  -> will use: one-way ANOVA")
    else:
        print("  -> will use: Kruskal-Wallis")
    print()
 
print("assumption check complete")

"""
=== assumption checks at N=500 ===

--- metric: Mean norm. cost (units/kWh) ---
  shapiro-wilk (H0: normal):
    Habitual : W= 0.863 , p= 0.0012 -> FAIL
    Progressive : W= 0.723 , p= 0.0 -> FAIL
    Tipping : W= 0.879 , p= 0.0027 -> FAIL
    Balanced : W= 0.934 , p= 0.0641 -> pass
    Price-Maximalist : W= 0.861 , p= 0.0011 -> FAIL
    Social-Maximalist : W= 0.745 , p= 0.0 -> FAIL
  levene (H0: equal variances): W= 1.058 , p= 0.3857 -> pass
  -> will use: Kruskal-Wallis

--- metric: Mean price advantage ---
  shapiro-wilk (H0: normal):
    Habitual : W= 0.943 , p= 0.1112 -> pass
    Progressive : W= 0.898 , p= 0.0074 -> FAIL
    Tipping : W= 0.964 , p= 0.3955 -> pass
    Balanced : W= 0.943 , p= 0.1067 -> pass
    Price-Maximalist : W= 0.924 , p= 0.0343 -> FAIL
    Social-Maximalist : W= 0.975 , p= 0.6785 -> pass
  levene (H0: equal variances): W= 3.644 , p= 0.0037 -> FAIL
  -> will use: Kruskal-Wallis

--- metric: EMD vs baseline ---
  shapiro-wilk (H0: normal):
    Habitual : W= 0.985 , p= 0.9367 -> pass
    Progressive : W= 0.987 , p= 0.9688 -> pass
    Tipping : W= 0.94 , p= 0.0912 -> pass
    Balanced : W= 0.984 , p= 0.916 -> pass
    Price-Maximalist : W= 0.953 , p= 0.2001 -> pass
    Social-Maximalist : W= 0.934 , p= 0.0626 -> pass
  levene (H0: equal variances): W= 0.693 , p= 0.6292 -> pass
  -> will use: one-way ANOVA

--- metric: Peak-to-average ratio ---
  shapiro-wilk (H0: normal):
    Habitual : W= 0.943 , p= 0.1079 -> pass
    Progressive : W= 0.974 , p= 0.6613 -> pass
    Tipping : W= 0.958 , p= 0.2736 -> pass
    Balanced : W= 0.965 , p= 0.4036 -> pass
    Price-Maximalist : W= 0.945 , p= 0.1256 -> pass
    Social-Maximalist : W= 0.98 , p= 0.8334 -> pass
  levene (H0: equal variances): W= 2.598 , p= 0.0271 -> FAIL
  -> will use: Kruskal-Wallis

--- metric: Mean flexibility (hrs/day/agent) ---
  shapiro-wilk (H0: normal):
    Habitual : W= 0.941 , p= 0.0947 -> pass
    Progressive : W= 0.95 , p= 0.1724 -> pass
    Tipping : W= 0.877 , p= 0.0023 -> FAIL
    Balanced : W= 0.954 , p= 0.2226 -> pass
    Price-Maximalist : W= 0.968 , p= 0.4865 -> pass
    Social-Maximalist : W= 0.982 , p= 0.8692 -> pass
  levene (H0: equal variances): W= 2.124 , p= 0.0648 -> pass
  -> will use: Kruskal-Wallis

--- metric: Adjustment day-29 (hrs) ---
  shapiro-wilk (H0: normal):
    Habitual : W= 0.91 , p= 0.015 -> FAIL
    Progressive : W= 0.911 , p= 0.0157 -> FAIL
    Tipping : W= 0.933 , p= 0.0596 -> pass
    Balanced : W= 0.954 , p= 0.2103 -> pass
    Price-Maximalist : W= 0.944 , p= 0.1193 -> pass
    Social-Maximalist : W= 0.946 , p= 0.1285 -> pass
  levene (H0: equal variances): W= 15.507 , p= 0.0 -> FAIL
  -> will use: Kruskal-Wallis

assumption check complete

## 5. Figure 1: day-29 aggregate load overlay (N=500, all populations)

For each of the six populations, what does the aggregate load curve look like
at the end of the simulation (day 29)?

Data is loaded from `RQ2 aggregate curves data/rq2_agg_curves.npz`, which was
generated by `rq2_aggregate_curves_generator.py`.

Each population contributes 30 day-29 curves (10 seeds x 3 network variants).
The median and IQR are computed across those 30 curves.

This figure answers: do different population compositions produce visibly
different shifted load shapes by the end of the simulation?
"""

#load the pre-generated aggregate curves file
agg_curves_path = agg_curves_dir / "rq2_agg_curves.npz"
data_agg = np.load(agg_curves_path, allow_pickle=False)

time_axis = np.linspace(0, 24, 96, endpoint=False)

fig, ax = plt.subplots(figsize=(11, 5))

for pop in pop_order_all:
    #load the array: rows = runs, columns = 15-min slots
    profiles = data_agg[pop]

    #compute median and IQR across all 30 runs
    lp_median = np.median(profiles, axis=0)
    lp_q25 = np.percentile(profiles, 25, axis=0)
    lp_q75 = np.percentile(profiles, 75, axis=0)

    color = pop_colors[pop]
    lstyle, marker = pop_styles[pop]

    #shade the iqr band first so lines sit on top
    ax.fill_between(time_axis, lp_q25, lp_q75, color=color, alpha=0.10)

    #plot the median line
    ax.plot(time_axis, lp_median,
            color=color, linestyle=lstyle, linewidth=2.0, label=pop)

ax.set_xlabel("Hour of day")
ax.set_ylabel("Aggregate load (kW)")
ax.set_title(
    "RQ2 - Day 29 aggregate load by population | N=500\n"
    "Median across 30 runs (10 seeds \u00d7 3 variants) | shaded band = IQR")
ax.set_xticks(range(0, 25, 2))
ax.grid(axis="y", alpha=0.2)
ax.legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig(figure_dir / "fig1_day29_load_by_population.png", bbox_inches="tight")
plt.show()

"""
## 6. Figure 2: combined 2x3 population x N comparison

Each panel shows one of the six system-level metrics plotted as lines across
the five network sizes (N = 100 to 1000).

Each line is one population. Points are bootstrapped means; bands are 95% CI.

This figure shows how both population composition and network size jointly
shape system-level outcomes.
"""

metric_list = list(metric_titles.keys())

#build the 2 row x 3 column grid
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes_flat = axes.flatten()

#one panel per metric
for ax_i, metric in enumerate(metric_list):
    ax = axes_flat[ax_i]

    #one line per population
    for pop in pop_order_all:
        pop_df = df[df["pop_label"] == pop]

        #collect mean and CI at each N value with explicit lists
        x_vals = []
        y_means = []
        y_lo = []
        y_hi = []
        for n_val in n_values:
            cell = pop_df[pop_df["N"] == n_val][metric].dropna().values
            if len(cell) == 0:
                continue
            m, lo, hi = bootstrap_ci(cell)
            x_vals.append(n_val)
            y_means.append(m)
            y_lo.append(lo)
            y_hi.append(hi)
        if len(x_vals) == 0:
            continue

        #pull the color and style for this population
        color = pop_colors[pop]
        lstyle, marker = pop_styles[pop]

        #plot the mean line
        ax.plot(x_vals, y_means,
                color=color, linestyle=lstyle, marker=marker,
                markersize=5, linewidth=1.6, label=pop, zorder=3)
        #fill the CI band
        ax.fill_between(x_vals, y_lo, y_hi,
                        color=color, alpha=0.12, zorder=2)

    #x-axis label only on the bottom row
    if ax_i >= 3:
        ax.set_xlabel("Network size (N)")
    ax.set_ylabel(metric_titles[metric], fontsize=9)
    ax.set_xticks(n_values)
    x_tick_labels = []
    for n in n_values:
        x_tick_labels.append(str(n))
    ax.set_xticklabels(x_tick_labels, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_title(metric_titles[metric].split("(")[0].strip(), fontsize=10)

#shared legend below the figure
legend_handles = []
for pop in pop_order_all:
    legend_handles.append(
        plt.Line2D([0], [0],
                   color=pop_colors[pop],
                   linestyle=pop_styles[pop][0],
                   marker=pop_styles[pop][1],
                   linewidth=1.5, markersize=5,
                   label=pop))
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.04))

fig.suptitle(
    "RQ2 - System metrics across populations and network sizes\n"
    "lines=mean | bands=95% CI", fontsize=11)
plt.tight_layout()
plt.savefig(figure_dir / "fig2_population_N_comparison.png", bbox_inches="tight")
plt.show()

"""
## 7. Figure 3: peak-hour shift vs baseline (N=500)

How many hours the daily load peak moved compared to the matched baseline.
Negative values mean the peak moved earlier; positive values mean later.

**Why this matters for interpreting PAR:** PAR (peak-to-average ratio)
goes up when any peak becomes sharper relative to the mean load. But a
sharper peak at 14:00 when solar generation is high and grid prices are low
is not a problem - it is actually desirable. A sharper peak at 18:00 when
everyone cooks dinner and solar is gone is the problem. Reading peak-hour
shift alongside PAR tells the complete story.
"""

#filter to n=500 for this supplementary figure
df_ph = df[df["N"] == 500].copy()

fig, ax = plt.subplots(figsize=(10, 5))

#x positions for the populations
x_positions = {}
for i, pop in enumerate(pop_order_all):
    x_positions[pop] = i

for pop in pop_order_all:
    pop_vals = df_ph[df_ph["pop_label"] == pop]["peak_hour_shift"].dropna().values
    x_pos = x_positions[pop]
    color = pop_colors[pop]

    #jitter x slightly so overlapping dots are visible
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.15, 0.15, size=len(pop_vals))

    ax.scatter(x_pos + jitter, pop_vals,
               color=color, s=22, alpha=0.65, zorder=3)

    #overlay the bootstrapped mean as error bars on diamond
    if len(pop_vals) > 0:
        m, lo, hi = bootstrap_ci(pop_vals)
        ax.errorbar(x_pos, m,
                    yerr=[[m - lo], [hi - m]],
                    fmt="D", color=color, markersize=9,
                    markeredgecolor="white", markeredgewidth=0.7,
                    elinewidth=1.8, capsize=4, zorder=5)

#reference line at zero = no shift vs baseline
ax.axhline(0, color="grey", linestyle="--", linewidth=1.0, label="No shift (baseline)")

ax.set_xticks(list(x_positions.values()))
ax.set_xticklabels(list(x_positions.keys()), rotation=20, ha="right", fontsize=8.5)
ax.set_ylabel("Peak-hour shift vs baseline (hours)\nnegative = moved earlier")
ax.set_title("RQ2 - peak-hour shift per population (N=500) \n bar=95% CI")
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(figure_dir / "fig3_peak_hour_shift.png", bbox_inches="tight")
plt.show()

"""
## 8. Figure 4: EMD vs baseline strip plot (N=500)

For each population, how different its **day-29** load curve is from the
matched no-shift baseline, measured by Earth Mover's Distance.

The baseline curve used for comparison is the **median** load profile across
all 30 baseline days (the typical no-shift day). The actual run curve is always
taken from **day 29** (the final shifted state). This captures the full effect
of 30 days of learning rather than diluting it by averaging over early days.

EMD = the minimum total amount of load that would need to be moved
(in kWh x time-slots) to transform one distribution into the other.
"""

#filter to n=500
df_emd = df[df["N"] == 500].copy()

fig, ax = plt.subplots(figsize=(10, 5))

x_positions = {}
for i, pop in enumerate(pop_order_all):
    x_positions[pop] = i

for pop in pop_order_all:
    pop_vals = df_emd[df_emd["pop_label"] == pop]["emd_vs_baseline"].dropna().values
    x_pos = x_positions[pop]
    color = pop_colors[pop]

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.15, 0.15, size=len(pop_vals))

    ax.scatter(x_pos + jitter, pop_vals,
               color=color, s=22, alpha=0.65, zorder=3)

    #bootstrapped mean and CI as a diamond
    if len(pop_vals) > 0:
        m, lo, hi = bootstrap_ci(pop_vals)
        ax.errorbar(x_pos, m,
                    yerr=[[m - lo], [hi - m]],
                    fmt="D", color=color, markersize=9,
                    markeredgecolor="black", markeredgewidth=0.8,
                    elinewidth=1.8, capsize=4, zorder=5)

ax.set_xticks(list(x_positions.values()))
ax.set_xticklabels(list(x_positions.keys()), rotation=20, ha="right", fontsize=8.5)
ax.set_ylabel("EMD vs baseline (slot units)")
ax.set_title(
    "RQ2 - EMD vs baseline (N=500)\n"
    "day-29 load vs median baseline | diamond=mean | bar=95% CI")
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(figure_dir / "fig4_emd_strip.png", bbox_inches="tight")
plt.show()

"""
## 9. Kruskal-Wallis: N effect per metric

Tests whether network size (N) significantly affects each system-level metric,
pooling all six populations together.

**Why Kruskal-Wallis and not two-way ANOVA:**
The assumption checks above show that all metrics  except EMD fail normality or equal
variance for at least some populations. The line plots in Figure 2 provide the visual companion to this test.

**Bonferroni correction:** six metrics are tested, so each p-value is
multiplied by 6 before comparison to alpha = 0.05.
"""

n_metrics_rq2 = len(metrics_rq2)
kw_n_rows = []
 
print("=== kruskal-wallis: N effect (all populations pooled) ===")
print()
 
for metric in metrics_rq2:
    print("---", metric_titles[metric], "---")
 
    #collect values per N level, pooling all populations
    n_groups = {}
    for n_val in n_values:
        vals = df[df["N"] == n_val][metric].dropna().values
        n_groups[n_val] = vals
 
    groups_list = [n_groups[n_val] for n_val in n_values if len(n_groups[n_val]) > 0]
    h_stat, p_kw = stats.kruskal(*groups_list)
    p_kw_bon = min(p_kw * n_metrics_rq2, 1.0)
    is_sig = p_kw_bon < alpha
    n_total = sum(len(n_groups[n_val]) for n_val in n_values if len(n_groups[n_val]) > 0)
    k_n = len([n_val for n_val in n_values if len(n_groups[n_val]) > 0])
    eps_sq = epsilon_sq_kw(h_stat, k_n, n_total)
 
    if is_sig:
        sig_label = "SIGNIFICANT"
    else:
        sig_label = "not significant"
 
    print("  H =", round(h_stat, 3),
          "| p =", round(p_kw, 4),
          "| p_bonferroni =", round(p_kw_bon, 4),
          "| epsilon_sq =", round(eps_sq, 3),
          "|", sig_label)
 
    kw_n_rows.append({
        "metric": metric,
        "H": round(h_stat, 3),
        "p_raw": round(p_kw, 4),
        "p_bonferroni": round(p_kw_bon, 4),
        "effect_size": round(eps_sq, 3),
        "effect_label": "epsilon_sq",
        "significant": is_sig})
 
print()
df_kw_n = pd.DataFrame(kw_n_rows)
df_kw_n.to_csv(results_dir / "rq2_kw_N_effect.csv", index=False)
print("kruskal-wallis N-effect results saved to results/rq2_kw_N_effect.csv")

"""
=== kruskal-wallis: N effect (all populations pooled) ===

--- Mean norm. cost (units/kWh) ---
  H = 2.62 | p = 0.6232 | p_bonferroni = 1.0 | epsilon_sq = -0.002 | not significant
--- Mean price advantage ---
  H = 0.475 | p = 0.9759 | p_bonferroni = 1.0 | epsilon_sq = -0.004 | not significant
--- EMD vs baseline ---
  H = 0.176 | p = 0.9963 | p_bonferroni = 1.0 | epsilon_sq = -0.004 | not significant
--- Peak-to-average ratio ---
  H = 524.606 | p = 0.0 | p_bonferroni = 0.0 | epsilon_sq = 0.582 | SIGNIFICANT
--- Mean flexibility (hrs/day/agent) ---
  H = 4.413 | p = 0.353 | p_bonferroni = 1.0 | epsilon_sq = 0.0 | not significant
--- Adjustment day-29 (hrs) ---
  H = 6.803 | p = 0.1467 | p_bonferroni = 0.88 | epsilon_sq = 0.003 | not significant

kruskal-wallis N-effect results saved to results/rq2_kw_N_effect.csv


10. Kruskal-Wallis: population effect at N=500
At N=500 specifically run the same test structure as in RQ1, this time comparing six populations against each other on each system-level metric.

Post-hoc pairwise comparisons use Mann-Whitney U on all 15 population pairs.

Bonferroni correction: We test six metrics, so each p-value is multiplied by 6 before being compared to alpha = 0.05.
"""

#collect values per population at N=500
oneway_rows = []

print("=== kruskal-wallis at N=500: population effect ===")
print()

for metric in metrics_rq2:
    print("---", metric_titles[metric], "---")

    #collect data per population
    pop_data = {}
    for pop in pop_order_all:
        vals = df_500[df_500["pop_label"] == pop][metric].dropna().values
        pop_data[pop] = vals

    #check if parametric is allowed for this metric
    if use_parametric.get(metric, False):
        #parametric path: one-way ANOVA across populations (only EMD passes assumption checks with my setups)
        groups_list = []
        for pop in pop_order_all:
            if len(pop_data[pop]) > 0:
                groups_list.append(pop_data[pop])
        f_stat, p_anova = stats.f_oneway(*groups_list)
        p_bon = min(p_anova * n_metrics_rq2, 1.0)
        is_sig = p_bon < alpha

        #compute eta-squared from SS components
        all_vals = np.concatenate([pop_data[pop] for pop in pop_order_all
                                   if len(pop_data[pop]) > 0])
        grand_mean = all_vals.mean()
        ss_between = 0.0
        ss_total = 0.0
        for pop in pop_order_all:
            vals = pop_data[pop]
            if len(vals) > 0:
                ss_between = ss_between + len(vals) * (vals.mean() - grand_mean) ** 2
                ss_total = ss_total + ((vals - grand_mean) ** 2).sum()
        if ss_total > 0:
            eta_sq = ss_between / ss_total
        else:
            eta_sq = np.nan

        print("  F=", round(f_stat, 3),
              ", p=", round(p_anova, 4),
              ", p_bonferroni=", round(p_bon, 4),
              ", eta_sq=", round(eta_sq, 3),
              ", significant=", is_sig)

        oneway_rows.append({
            "metric": metric,
            "test": "ANOVA",
            "statistic": round(f_stat, 3),
            "p_raw": round(p_anova, 4),
            "p_bonferroni": round(p_bon, 4),
            "effect_size": round(eta_sq, 3),
            "effect_label": "eta_sq",
            "significant": is_sig,
            "pair": "all populations"})

        #if significant -> run Tukey HSD post-hoc across all 15 population pairs
        if is_sig:
            print("  -> running Tukey HSD post-hoc comparisons")

            #flatten values and labels for pairwise_tukeyhsd
            all_vals_flat = np.concatenate([pop_data[pop] for pop in pop_order_all
                                            if len(pop_data[pop]) > 0])
            all_labels_flat = []
            for pop in pop_order_all:
                for _ in range(len(pop_data[pop])):
                    all_labels_flat.append(pop)
            all_labels_flat = np.array(all_labels_flat)

            tukey_result = pairwise_tukeyhsd(all_vals_flat, all_labels_flat, alpha=alpha)
            tukey_table = tukey_result._results_table.data

            for row in tukey_table[1:]:
                g1, g2, meandiff, p_adj, lower, upper, reject = row
                p_tukey_bon = min(float(p_adj) * n_metrics_rq2, 1.0)
                print("   ", g1, "vs", g2,
                      ": meandiff=", round(float(meandiff), 4),
                      ", p_tukey=", round(float(p_adj), 4),
                      ", p_tukey_bon=", round(p_tukey_bon, 4),
                      ", reject=", reject)

                oneway_rows.append({
                    "metric": "  " + metric,
                    "test": "Tukey HSD",
                    "statistic": round(float(meandiff), 4),
                    "p_raw": round(float(p_adj), 4),
                    "p_bonferroni": round(p_tukey_bon, 4),
                    "effect_size": "",
                    "effect_label": "",
                    "significant": bool(reject),
                    "pair": str(g1) + " vs " + str(g2)})

    else:
        #non-parametric path: Kruskal-Wallis
        all_groups = [pop_data[pop] for pop in pop_order_all if len(pop_data[pop]) > 0]
        h_stat, p_kw = stats.kruskal(*all_groups)
        p_bon = min(p_kw * n_metrics_rq2, 1.0)
        is_sig = p_bon < alpha

        n_total = sum(len(pop_data[pop]) for pop in pop_order_all if len(pop_data[pop]) > 0)
        k_pops = len([pop for pop in pop_order_all if len(pop_data[pop]) > 0])
        eps_sq = epsilon_sq_kw(h_stat, k_pops, n_total)

        print("  H =", round(h_stat, 3),
              ", p =", round(p_kw, 4),
              ", p_bonferroni =", round(p_bon, 4),
              ", epsilon_sq =", round(eps_sq, 3),
              ", significant =", is_sig)

        oneway_rows.append({
            "metric": metric,
            "test": "Kruskal-Wallis",
            "statistic": round(h_stat, 3),
            "p_raw": round(p_kw, 4),
            "p_bonferroni": round(p_bon, 4),
            "effect_size": round(eps_sq, 3),
            "effect_label": "epsilon_sq",
            "significant": is_sig,
            "pair": "all populations"})

        #if significant -> mann-whitney u pairwise across all 15 pairs
        if is_sig:
            #build all 15 pairwise combinations
            pairs = []
            for pi in range(len(pop_order_all)):
                for pj in range(pi + 1, len(pop_order_all)):
                    pairs.append((pop_order_all[pi], pop_order_all[pj]))

            for g1, g2 in pairs:
                mw_stat, mw_p = stats.mannwhitneyu(
                    pop_data[g1], pop_data[g2], alternative="two-sided")
                #correct for both metric family and number of pairs
                mw_p_bon = min(mw_p * n_metrics_rq2 * len(pairs), 1.0)

                r_rb = rank_biserial_r(pop_data[g1], pop_data[g2], mw_stat)

                print("  ", g1, "vs", g2,
                      ": p =", round(mw_p, 4),
                      ", p_bon =", round(mw_p_bon, 4),
                      ", r_rb =", round(r_rb, 3))

                oneway_rows.append({
                    "metric": "  " + metric,
                    "test": "Mann-Whitney U",
                    "statistic": round(mw_stat, 3),
                    "p_raw": round(mw_p, 4),
                    "p_bonferroni": round(mw_p_bon, 4),
                    "effect_size": round(r_rb, 3),
                    "effect_label": "rank_biserial_r",
                    "significant": mw_p_bon < alpha,
                    "pair": g1 + " vs " + g2})

    print()

df_oneway = pd.DataFrame(oneway_rows)
df_oneway.to_csv(results_dir / "rq2_oneway_N500.csv", index=False)
print("one-way test results saved to results/rq2_oneway_N500.csv")

"""
=== kruskal-wallis at N=500: population effect ===

--- Mean norm. cost (units/kWh) ---
  H = 165.162 , p = 0.0 , p_bonferroni = 0.0 , epsilon_sq = 0.92 , significant = True
   Habitual vs Progressive : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Habitual vs Tipping : p = 0.6627 , p_bon = 1.0 , r_rb = -0.067
   Habitual vs Balanced : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Habitual vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Habitual vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 0.938
   Progressive vs Tipping : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Progressive vs Balanced : p = 0.0 , p_bon = 0.0001 , r_rb = -0.74
   Progressive vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Progressive vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Tipping vs Balanced : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Tipping vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Tipping vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 0.947
   Balanced vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Balanced vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Price-Maximalist vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0

--- Mean price advantage ---
  H = 169.059 , p = 0.0 , p_bonferroni = 0.0 , epsilon_sq = 0.943 , significant = True
   Habitual vs Progressive : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Habitual vs Tipping : p = 0.217 , p_bon = 1.0 , r_rb = 0.187
   Habitual vs Balanced : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Habitual vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Habitual vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Progressive vs Tipping : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Progressive vs Balanced : p = 0.0 , p_bon = 0.0 , r_rb = 0.982
   Progressive vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Progressive vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Tipping vs Balanced : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Tipping vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Tipping vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Balanced vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Balanced vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Price-Maximalist vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 1.0

--- EMD vs baseline ---
  F= 733.735 , p= 0.0 , p_bonferroni= 0.0 , eta_sq= 0.955 , significant= True
  -> running Tukey HSD post-hoc comparisons
    Balanced vs Habitual : meandiff= -0.8029 , p_tukey= 0.0 , p_tukey_bon= 0.0 , reject= True
    Balanced vs Price-Maximalist : meandiff= 1.1295 , p_tukey= 0.0 , p_tukey_bon= 0.0 , reject= True
    Balanced vs Progressive : meandiff= 0.0574 , p_tukey= 0.6298 , p_tukey_bon= 1.0 , reject= False
    Balanced vs Social-Maximalist : meandiff= -0.2972 , p_tukey= 0.0 , p_tukey_bon= 0.0 , reject= True
    Balanced vs Tipping : meandiff= -0.7477 , p_tukey= 0.0 , p_tukey_bon= 0.0 , reject= True
    Habitual vs Price-Maximalist : meandiff= 1.9324 , p_tukey= 0.0 , p_tukey_bon= 0.0 , reject= True
    Habitual vs Progressive : meandiff= 0.8602 , p_tukey= 0.0 , p_tukey_bon= 0.0 , reject= True
    Habitual vs Social-Maximalist : meandiff= 0.5057 , p_tukey= 0.0 , p_tukey_bon= 0.0 , reject= True
    Habitual vs Tipping : meandiff= 0.0552 , p_tukey= 0.667 , p_tukey_bon= 1.0 , reject= False
    Price-Maximalist vs Progressive : meandiff= -1.0721 , p_tukey= 0.0 , p_tukey_bon= 0.0 , reject= True
    Price-Maximalist vs Social-Maximalist : meandiff= -1.4267 , p_tukey= 0.0 , p_tukey_bon= 0.0 , reject= True
    Price-Maximalist vs Tipping : meandiff= -1.8772 , p_tukey= 0.0 , p_tukey_bon= 0.0 , reject= True
    Progressive vs Social-Maximalist : meandiff= -0.3545 , p_tukey= 0.0 , p_tukey_bon= 0.0 , reject= True
    Progressive vs Tipping : meandiff= -0.805 , p_tukey= 0.0 , p_tukey_bon= 0.0 , reject= True
    Social-Maximalist vs Tipping : meandiff= -0.4505 , p_tukey= 0.0 , p_tukey_bon= 0.0 , reject= True

--- Peak-to-average ratio ---
  H = 149.99 , p = 0.0 , p_bonferroni = 0.0 , epsilon_sq = 0.833 , significant = True
   Habitual vs Progressive : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Habitual vs Tipping : p = 0.3632 , p_bon = 1.0 , r_rb = -0.138
   Habitual vs Balanced : p = 0.0 , p_bon = 0.0 , r_rb = 0.998
   Habitual vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -0.998
   Habitual vs Social-Maximalist : p = 0.2707 , p_bon = 1.0 , r_rb = 0.167
   Progressive vs Tipping : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Progressive vs Balanced : p = 0.0575 , p_bon = 1.0 , r_rb = -0.287
   Progressive vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Progressive vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Tipping vs Balanced : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Tipping vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Tipping vs Social-Maximalist : p = 0.0392 , p_bon = 1.0 , r_rb = 0.311
   Balanced vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Balanced vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -0.978
   Price-Maximalist vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 0.996

--- Mean flexibility (hrs/day/agent) ---
  H = 172.99 , p = 0.0 , p_bonferroni = 0.0 , epsilon_sq = 0.965 , significant = True
   Habitual vs Progressive : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Habitual vs Tipping : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Habitual vs Balanced : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Habitual vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Habitual vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Progressive vs Tipping : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Progressive vs Balanced : p = 0.0 , p_bon = 0.0 , r_rb = 0.889
   Progressive vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Progressive vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Tipping vs Balanced : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Tipping vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Tipping vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Balanced vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Balanced vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Price-Maximalist vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 1.0

--- Adjustment day-29 (hrs) ---
  H = 169.186 , p = 0.0 , p_bonferroni = 0.0 , epsilon_sq = 0.944 , significant = True
   Habitual vs Progressive : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Habitual vs Tipping : p = 0.0 , p_bon = 0.0037 , r_rb = -0.618
   Habitual vs Balanced : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Habitual vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Habitual vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Progressive vs Tipping : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Progressive vs Balanced : p = 0.0 , p_bon = 0.0 , r_rb = 0.802
   Progressive vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Progressive vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Tipping vs Balanced : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Tipping vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Tipping vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Balanced vs Price-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = -1.0
   Balanced vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 1.0
   Price-Maximalist vs Social-Maximalist : p = 0.0 , p_bon = 0.0 , r_rb = 1.0

one-way test results saved to results/rq2_oneway_N500.csv
11. Summary table
"""

#build summary at N=500: mean and sd per population per metric
summary_rows = []

for pop in pop_order_all:
    pop_df = df_500[df_500["pop_label"] == pop]
    row = {"population": pop}

    for metric in metrics_rq2:
        vals = pop_df[metric].dropna().values
        if len(vals) > 0:
            row[metric + "_mean"] = round(float(np.mean(vals)), 4)
            row[metric + "_sd"] = round(float(np.std(vals, ddof=1)), 4)
        else:
            row[metric + "_mean"] = np.nan
            row[metric + "_sd"] = np.nan

    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(results_dir / "rq2_summary_N500.csv", index=False)

print("=== RQ2 summary table (mean at N=500) ===")
print(df_summary)
print()
print("summary saved to results/rq2_summary_N500.csv")

"""
=== RQ2 summary table (mean at N=500) ===
          population  mean_cost_norm_mean  mean_cost_norm_sd  \
0           Habitual               8.7312             0.0323   
1        Progressive               8.4202             0.0294   
2            Tipping               8.7336             0.0328   
3           Balanced               8.4691             0.0326   
4   Price-Maximalist               8.0464             0.0228   
5  Social-Maximalist               8.6408             0.0311   

   mean_price_advantage_mean  mean_price_advantage_sd  emd_vs_baseline_mean  \
0                    -1.0464                   0.0105                1.1762   
1                    -0.7866                   0.0111                2.0364   
2                    -1.0490                   0.0099                1.2314   
3                    -0.8285                   0.0158                1.9790   
4                    -0.5025                   0.0104                3.1085   
5                    -0.9695                   0.0157                1.6819   

   emd_vs_baseline_sd  par_mean_mean  par_mean_sd  mean_flexibility_mean  \
0              0.1211         1.5495       0.0077                 1.5132   
1              0.1266         1.5195       0.0055                 2.8978   
2              0.1534         1.5505       0.0062                 1.6619   
3              0.1274         1.5228       0.0079                 2.7979   
4              0.1544         1.5812       0.0095                 4.9336   
5              0.1681         1.5468       0.0091                 2.2748   

   mean_flexibility_sd  mean_adjustment_mean  mean_adjustment_sd  
0               0.0467                1.0397              0.0219  
1               0.0440                1.8030              0.0319  
2               0.0358                1.0684              0.0209  
3               0.0476                1.7399              0.0356  
4               0.0628                2.7090              0.0729  
5               0.0323                1.4813              0.0484  

summary saved to results/rq2_summary_N500.csv


## Summary

**Figure 1** -> For each of the six populations, what does the aggregate load curve look like
at the end of the simulation

Early peak is universal, no difference for habit/price heavy, but evening peak is more spread out. Material for discussion, as I can't immediatly tell why that would be


**Figure 2** -> investigate the metrics across N

Mean norm. cost: Flat across N. Price-Maximalist gets out best, habitual the least. This hierarchy reflects composition: more price-responsive agents = lower costs. This hierarchy can be seen in all plots

Mean price advantage: Less negative = closer to or below the mean = better outcome. Price-Maximalist being least negative means they actually do successfully shift to cheaper hours and pay less. They are closest to beating the daily mean. This is exactly what’s expected  habitual being most negative means they're stuck in expensive peak hours, paying furthest above the daily mean. They shift the least, so they stay where prices are high. Why all negative -> Likely because the daily mean price is anchored low by the cheap off-peak hours, and even the best-shifting agents are still consuming during expensive hours. No population fully escapes peak-hour consumption. But will need to think more

EMD vs baseline: price-maximalist deviates most from baseline, habitual least. EMD measures "how different is day-29 from no-shift." Higher = more behavioral change happened OVERALL. Slight drop with N suggests larger networks dampen individual agent variation, stabilizing the shifted curve.

Peak-to-average ratio: Sharp drop from N=100 to N=500, then plateaus. Larger networks have smooth demand. Price-maximalist still peaks highest even at N=1000 because their concentrated shifting creates a sharp redistribution. Habitual flatter because they don't shift.

Total flexibility: Linear scaling with N.  This is expected from the code: total_flexibility is the sum across all agents. More agents = more total hours shifted. But per-agent flexibility is similar across N, otherwise it wouldn’t be linear

Adjustment day-29: Flat across N. Price-maximalist adjusts most, habitual least


**Figure 3** -> How did the peak hour move?

Expected results, both peak and midday peaks move earlier as seen 1, so the downward trend can be seen in fig 3 as well. I do not understand the outliers. Progressive and Balanced population are more spread/unpredictable

**Figure 4** -> Taking another look at EMD

Expected results, same data as in fig 1 and it follows the familiar structure with price maximalists on top and habit population on the bottom

--------------------------------------------

**Statistical Analysis**

*N effect*: Only PAR is significantly affected by network size. All other metrics are stable across N.

*Population effect*: All six metrics are significantly different across populations.
Effect sizes are very large across the board: epsilon_sq ranges from 0.833 (PAR) to 0.965 (flexibility),
and EMD via ANOVA gives eta_sq = 0.955 -> population composition explains >83% of variance in every metric.

*Pairwise structure*:
Price-Maximalist is separated from every other population on every metric with r_rb = +-1.0 -> complete
non-overlap

*EMD post-hoc (Tukey HSD)* reveals two non-significant pairs: Habitual vs Tipping and Balanced vs
Progressive. Despite having slightly different compositions, they produce statistically near identical
load shape changes relative to baseline 

*PAR* has a notable cluster of non-significant pairs: Habitual vs Tipping, Habitual vs Social-Maximalist,
Progressive vs Balanced, and Tipping vs Social-Maximalist.
Social-Maximalist shifts a lot by flexibility and adjustment measures but does not produce a meaningfully
different PAR from Habitual 

Progressive and Balanced are significantly different on cost, flexibility, and adjustment but the same on PAR. 
Habitual and Tipping cluster together on 4 of 6 metrics, only separating
weakly on adjustment

"""
