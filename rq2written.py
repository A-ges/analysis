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

#network sizes used in RQ2 (and generated with datagenerator)
n_values = [100, 250, 500, 750, 1000]

#total simulated days
days = 30

#significance threshold used in all tests
alpha = 0.05

print("notebook loaded and ready")

#-------------------------------------------

#load the main rq2 summary csv
df = pd.read_csv(results_dir / "rq2_run_summaries.csv")

print("rows:", len(df))
print("columns:", list(df.columns))
print()
print(df.groupby(["pop_label", "N"]).size().rename("n_runs").to_string())

#-----------------------------------------------
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


print("Functions defined: bootstrap_ci, cohens_d")
#--------------------------------------------------

#filter to n=500 for the assumption checks
df_500 = df[df["N"] == 500].copy()

#metrics to test
metrics_rq2 = [
    "mean_cost_norm",
    "mean_price_advantage",
    "emd_vs_baseline",
    "par_mean",
    "total_flexibility",
    "mean_adjustment"]

metric_titles = {
    "mean_cost_norm": "Mean norm. cost (units/kWh)",
    "mean_price_advantage": "Mean price advantage",
    "emd_vs_baseline": "EMD vs baseline",
    "par_mean": "Peak-to-average ratio",
    "total_flexibility": "Total flexibility (hrs/day)",
    "mean_adjustment": "Adjustment day-29 (hrs)"}

#store whether each metric passes assumptions
#  -> True means ANOVA can be used, False means use Kruskal-Wallis
use_parametric = {}

print("=== assumption checks at N=500 ===")
print()

for metric in metrics_rq2:
    print("--- metric:", metric_titles[metric], "---")

    #collect values per population
    group_data = {}
    for pop in pop_order_all:
        vals = df_500[df_500["pop_label"] == pop][metric].dropna().values
        group_data[pop] = vals

    #shapiro-wilk normality test for each population
    #  -> H0 = the data is normally distributed
    #  -> p > 0.05 means normality is not rejected
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

    #levene test for equal variances across populations
    #  -> H0 = all groups have equal variance
    #  -> p > 0.05 means equal variance is not rejected
    all_groups = []
    for pop in pop_order_all:
        if len(group_data[pop]) > 0:
            all_groups.append(group_data[pop])

    equal_var = True
    if len(all_groups) >= 2:
        lev_stat, p_lev = stats.levene(*all_groups)
        if p_lev < alpha:
            equal_var = False
            lev_result = "FAIL"
        else:
            lev_result = "pass"
        print("  levene (H0: equal variances): W=", round(lev_stat, 3), ", p=", round(p_lev, 4), "->", lev_result)

    #decision: only use parametric ANOVA if both normality and equal variance hold
    if all_normal and equal_var:
        use_parametric[metric] = True
        print("  -> will use: one-way ANOVA")
    else:
        use_parametric[metric] = False
        print("  -> will use: Kruskal-Wallis")
    print()

print("assumption check complete")
#----------------------------------------------
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
print("figure 1 saved")
#------------------------------------------
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
#----------------------------------------------------
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
#---------------------------------------------
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
    "RQ2 - EMD vs baseline strip plot (N=500)\n"
    "day-29 load vs median baseline | diamond =  mean + 95% CI")
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(figure_dir / "fig4_emd_strip.png", bbox_inches="tight")
plt.show()
#-------------------------------------------------------------------
#number of metrics for bonferroni correction
n_metrics_rq2 = len(metrics_rq2)

#collect results
kw_n_rows = []

print("=== kruskal-wallis: N effect (all populations pooled) ===")
print()

for metric in metrics_rq2:
    print("---", metric_titles[metric], "---")

    #collect values per N level, pooling all populations and variants
    n_groups = {}
    for n_val in n_values:
        cell = df[df["N"] == n_val][metric].dropna().values
        n_groups[n_val] = cell

    #build the input list for kruskal, skip any empty groups
    data_list = []
    for n_val in n_values:
        if len(n_groups[n_val]) > 0:
            data_list.append(n_groups[n_val])

    if len(data_list) < 2:
        print("  not enough groups, skipping")
        continue

    h_stat, p_kw = stats.kruskal(*data_list)
    p_bon = min(p_kw * n_metrics_rq2, 1.0)
    is_sig = p_bon < alpha

    if is_sig:
        flag = "SIGNIFICANT"
    else:
        flag = "not significant"

    print("  H =", round(h_stat, 3),
          "| p =", round(p_kw, 4),
          "| p_bonferroni =", round(p_bon, 4),
          "|", flag)
    print()

    kw_n_rows.append({
        "metric": metric,
        "test": "Kruskal-Wallis (N effect)",
        "H": round(h_stat, 3),
        "p_raw": round(p_kw, 4),
        "p_bonferroni": round(p_bon, 4),
        "significant": is_sig})

df_kw_n = pd.DataFrame(kw_n_rows)
df_kw_n.to_csv(results_dir / "rq2_kw_N_effect.csv", index=False)
print("kruskal-wallis N-effect results saved to results/rq2_kw_N_effect.csv")
#------------------------------------
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
        #parametric path: one-way ANOVA across populations
        groups_list = []
        for pop in pop_order_all:
            if len(pop_data[pop]) > 0:
                groups_list.append(pop_data[pop])
        f_stat, p_anova = stats.f_oneway(*groups_list)
        p_bon = min(p_anova * n_metrics_rq2, 1.0)
        is_sig = p_bon < alpha

        print("  F=", round(f_stat, 3),
              ", p=", round(p_anova, 4),
              ", p_bonferroni=", round(p_bon, 4),
              ", significant=", is_sig)

        oneway_rows.append({
            "metric": metric,
            "test": "ANOVA",
            "statistic": round(f_stat, 3),
            "p_raw": round(p_anova, 4),
            "p_bonferroni": round(p_bon, 4),
            "effect_size": "",
            "effect_label": "",
            "significant": is_sig,
            "pair": "all populations"})

        #post-hoc tukey if significant
        if is_sig:
            all_vals = []
            all_labels = []
            for pop in pop_order_all:
                for v in pop_data[pop]:
                    all_vals.append(v)
                    all_labels.append(pop)
            tukey = pairwise_tukeyhsd(all_vals, all_labels, alpha=alpha)
            print(tukey)

    else:
        #non-parametric path: kruskal-wallis across all six populations
        data_list = []
        for pop in pop_order_all:
            if len(pop_data[pop]) > 0:
                data_list.append(pop_data[pop])

        h_stat, p_kw = stats.kruskal(*data_list)
        p_bon = min(p_kw * n_metrics_rq2, 1.0)
        is_sig = p_bon < alpha

        print("  H=", round(h_stat, 3),
              ", p=", round(p_kw, 4),
              ", p_bonferroni=", round(p_bon, 4),
              ", significant=", is_sig)

        oneway_rows.append({
            "metric": metric,
            "test": "Kruskal-Wallis",
            "statistic": round(h_stat, 3),
            "p_raw": round(p_kw, 4),
            "p_bonferroni": round(p_bon, 4),
            "effect_size": "",
            "effect_label": "",
            "significant": is_sig,
            "pair": "all populations"})

        #if significant -> mann-whitney u pairwise across all 15 pairs
        if is_sig:
            #build all 15 pairwise combinations explicitly with for loops
            pairs = []
            for pi in range(len(pop_order_all)):
                for pj in range(pi + 1, len(pop_order_all)):
                    pairs.append((pop_order_all[pi], pop_order_all[pj]))

            for g1, g2 in pairs:
                mw_stat, mw_p = stats.mannwhitneyu(
                    pop_data[g1], pop_data[g2], alternative="two-sided")
                #correct for both metric family and number of pairs
                mw_p_bon = min(mw_p * n_metrics_rq2 * len(pairs), 1.0)
                d = cohens_d(pop_data[g1], pop_data[g2])
                oneway_rows.append({
                    "metric": "  " + metric,
                    "test": "Mann-Whitney U",
                    "statistic": round(mw_stat, 3),
                    "p_raw": round(mw_p, 4),
                    "p_bonferroni": round(mw_p_bon, 4),
                    "effect_size": round(d, 3) if not np.isnan(d) else "",
                    "effect_label": "cohens_d",
                    "significant": mw_p_bon < alpha,
                    "pair": g1 + " vs " + g2})
    print()

df_oneway = pd.DataFrame(oneway_rows)
df_oneway.to_csv(results_dir / "rq2_oneway_N500.csv", index=False)
print("one-way test results saved to results/rq2_oneway_N500.csv")
#-----------------------

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
print(df_summary.to_string(index=False))
print()
print("summary saved to results/rq2_summary_N500.csv")
