"""
# RQ1 - Behavioral type effects within Populations

**Research question:** How do different household behavioral decision architectures
(habit-driven, price-responsive, socially influenced) affect flexibility,
adjustment, and cost-comfort trade-offs within the ABM?

**Data source:** results/rq1_run_summaries.csv. Written by datagenerator.py
One row per (population, network_code, seed, dominant_group). With 3 variants
(a, b, c) and 10 seeds there are 30 rows per (population, group) cell.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#point to the project root
project_root = Path(".")
sys.path.insert(0, str(project_root))

#paths to the cached data and the output figure folder
results_dir = Path("results")
figure_dir = Path("figures/rq1")
figure_dir.mkdir(parents=True, exist_ok=True)

#consistent plot style for all figures in this notebook
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

#define one color per behavioral group used in all figures
group_colors = {
    "Habit-driven": "red",
    "Price-responsive": "green",
    "Social-influenced": "blue"}

group_short = {
    "Habit-driven": "Habit",
    "Price-responsive": "Price",
    "Social-influenced": "Social"}

groups = ["Habit-driven", "Price-responsive", "Social-influenced"]

pop_order = ["Habitual", "Progressive", "Tipping", "Balanced"]

metric_labels = {
    "flex_mean": "Mean flexibility (hrs/day)",
    "cost_norm_mean": "Mean norm. cost (units/kWh)",
    "adjustment_mean": "Adjustment day-29 (hrs)"}

print("notebook loaded and ready")

## 2. Load and inspect data

#load the summary CSV that datagenerator.py wrote for RQ1
df = pd.read_csv(results_dir / "rq1_run_summaries.csv")

print("shape:", df.shape)
print()

#Confirm the expected number of runs per cell
#  -> 3 variants (a, b, c) x 10 seeds = 30 runs per (population, group)
print("runs per (population, dominant_group):")
for pop in pop_order:
    print(" ", pop, ":")
    for group in groups:
        count = len(df[(df["pop_label"] == pop) & (df["dominant_group"] == group)])
        print("   ", group, ":", count, "runs (expected 30)")

print()
print("column names in the CSV:")
print(df.columns)

"""
shape: (360, 13)

runs per (population, dominant_group):
  Habitual :
    Habit-driven : 30 runs (expected 30)
    Price-responsive : 30 runs (expected 30)
    Social-influenced : 30 runs (expected 30)
  Progressive :
    Habit-driven : 30 runs (expected 30)
    Price-responsive : 30 runs (expected 30)
    Social-influenced : 30 runs (expected 30)
  Tipping :
    Habit-driven : 30 runs (expected 30)
    Price-responsive : 30 runs (expected 30)
    Social-influenced : 30 runs (expected 30)
  Balanced :
    Habit-driven : 30 runs (expected 30)
    Price-responsive : 30 runs (expected 30)
    Social-influenced : 30 runs (expected 30)

column names in the CSV:
Index(['pop_label', 'network_code', 'N', 'seed', 'dominant_group', 'flex_mean',
       'cost_norm_mean', 'adjustment_mean', 'price_contrib_mean',
       'social_contrib_mean', 'price_contrib_pct', 'social_contrib_pct',
       'savings_per_flex_mean'],
      dtype='object')

3. Functions
"""

#bootstrap confidence interval function to get the figure error bars
#  -> Used for visual error bars not for formal hypothesis testing
def bootstrap_ci(values, n_boot=2000, ci_level=0.95, seed=0):
    rng = np.random.default_rng(seed)

    #convert to a clean numpy array of floats with NaN values removed
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]

    #handle empty input case
    if len(vals) == 0:
        return np.nan, np.nan, np.nan

    #generate n_boot resampled means 
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boot_means[b] = sample.mean()

    #compute the actual mean and the percentile-based confidence interval
    mean_val = float(vals.mean())
    lo = float(np.percentile(boot_means, (1 - ci_level) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci_level) / 2 * 100))
    return mean_val, lo, hi


#cohen's d effect size between two groups if Anova
#  -> tells how large a difference is in units of pooled standard deviation
#  -> Convention is small = 0.2, medium = 0.5, large = 0.8
def cohens_d(vals_a, vals_b):
    a = np.array(vals_a, dtype=float)
    b = np.array(vals_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    #handle degenerate input cases
    if len(a) < 2 or len(b) < 2:
        return np.nan

    #pooled standard deviation across both groups
    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)

    #avoid division by zero if both groups happen to be constant
    if pooled_sd == 0:
        return np.nan

    return float((np.mean(a) - np.mean(b)) / pooled_sd)

def rank_biserial_r(x, y, u_stat):
    """
    Compute rank-biserial correlation as effect size for Mann-Whitney U.
 
    Parameters:
    -> x: array for group 1
    -> y: array for group 2
    -> u_stat: U statistic returned by scipy.stats.mannwhitneyu(x, y)
               this is U1, the count of how often x[i] > y[j]
 
    Returns rank-biserial r in [-1, 1]
    -> r > 0 means x tends to be larger than y
    -> r < 0 means y tends to be larger than x
    -> |r| >= 0.1 small, >= 0.3 medium, >= 0.5 large (Cohen, 1988)
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
    -> Interpretation: proportion of variance in ranks explained by group
    -> >= 0.01 small, >= 0.06 medium, >= 0.14 large (Cohen, 1988 adapted)
    """
    if n <= k:
        return np.nan
    return (h_stat - k + 1) / (n - k)

"""
4. Assumption checks before choosing a statistical test
Before running a one-way ANOVA we need to verify two assumptions:

Normality: Each of the three groups' run-level means should be approximately normally distributed.
Homogeneity of variance: The spread of run-level means should be similar across the three groups.
Why do these tests ANOVA works by comparing the variance between groups to the variance within groups. If a group has a skewed distribution or unusually large spread, the F-statistic ANOVA produces will be unreliable

In case the test fails If normality or equal variance is violated for a (population, metric) cell, use Kruskal-Wallis instead

Tests used:

Shapiro-Wilk tests whether a sample could plausibly come from a normal distribution. p > 0.05 means normality is not rejected
Levene tests whether groups have equal variances p > 0.05 means equality is not rejected
"""
#significance threshold used in all tests in this notebook
alpha = 0.05

#dictionary that stores the assumption check outcome per (pop, metric)
#  -> True means we ANOVA is used  
#  -> False means Kruskal-Wallis 
use_parametric = {}

print("=== assumption checks ===")
print("significance threshold: alpha =", alpha)
print()

#loop over every (population, metric) cell and run both tests
for pop in pop_order:
    pop_df = df[df["pop_label"] == pop]
    print("--- population:", pop, "---")

    for metric in metric_labels.keys():
        print("  metric:", metric)

        #collect the run-level values per group into a dict
        group_data = {}
        for group in groups:
            vals = pop_df[pop_df["dominant_group"] == group][metric].dropna().values
            group_data[group] = vals

        #shapiro-wilk normality test for each group
        #  -> H0 = the data is normally distributed
        #  -> all groups have to pass for a normal ANOVA
        all_normal = True
        print("    shapiro-wilk (H0: normal):")
        for group in groups:
            vals = group_data[group]
            sw_stat, sw_p = stats.shapiro(vals)
            is_normal = sw_p > alpha
            if not is_normal:
                all_normal = False
            if is_normal:
                result_label = "PASS"
            else:
                result_label = "FAIL"
            print("     ", group, ": W=", round(sw_stat, 3),
                  ", p=", round(sw_p, 4), "->", result_label)

        #levene test for homogeneity of variance
        #  -> H0 = all groups have equal variance
        data_list = []
        for group in groups:
            if len(group_data[group]) > 1:
                data_list.append(group_data[group])
        equal_var = True
        if len(data_list) >= 2:
            lev_stat, lev_p = stats.levene(*data_list)
            equal_var = lev_p > alpha
            if equal_var:
                lev_label = "PASS"
            else:
                lev_label = "FAIL"
            print("    levene (H0: equal variances): W=", round(lev_stat, 3),
                  ", p=", round(lev_p, 4), "->", lev_label)

        #decide which test to use for this cell
        go_parametric = all_normal and equal_var
        use_parametric[(pop, metric)] = go_parametric
        if go_parametric:
            test_name = "ANOVA"
        else:
            test_name = "Kruskal-Wallis"
        print("    -> will use:", test_name)
        print()

print("assumption check complete")
print("results stored in use_parametric dict")

"""
=== assumption checks ===
significance threshold: alpha = 0.05

--- population: Habitual ---
  metric: flex_mean
    shapiro-wilk (H0: normal):
      Habit-driven : W= 0.904 , p= 0.0103 -> FAIL
      Price-responsive : W= 0.894 , p= 0.006 -> FAIL
      Social-influenced : W= 0.848 , p= 0.0006 -> FAIL
    levene (H0: equal variances): W= 17.791 , p= 0.0 -> FAIL
    -> will use: Kruskal-Wallis

  metric: cost_norm_mean
    shapiro-wilk (H0: normal):
      Habit-driven : W= 0.842 , p= 0.0004 -> FAIL
      Price-responsive : W= 0.911 , p= 0.016 -> FAIL
      Social-influenced : W= 0.986 , p= 0.9595 -> PASS
    levene (H0: equal variances): W= 9.004 , p= 0.0003 -> FAIL
    -> will use: Kruskal-Wallis

  metric: adjustment_mean
    shapiro-wilk (H0: normal):
      Habit-driven : W= 0.926 , p= 0.0388 -> FAIL
      Price-responsive : W= 0.848 , p= 0.0006 -> FAIL
      Social-influenced : W= 0.911 , p= 0.0156 -> FAIL
    levene (H0: equal variances): W= 5.038 , p= 0.0085 -> FAIL
    -> will use: Kruskal-Wallis

--- population: Progressive ---
  metric: flex_mean
    shapiro-wilk (H0: normal):
      Habit-driven : W= 0.949 , p= 0.1548 -> PASS
      Price-responsive : W= 0.969 , p= 0.512 -> PASS
      Social-influenced : W= 0.976 , p= 0.7125 -> PASS
    levene (H0: equal variances): W= 13.042 , p= 0.0 -> FAIL
    -> will use: Kruskal-Wallis

  metric: cost_norm_mean
    shapiro-wilk (H0: normal):
      Habit-driven : W= 0.759 , p= 0.0 -> FAIL
      Price-responsive : W= 0.859 , p= 0.001 -> FAIL
      Social-influenced : W= 0.898 , p= 0.0075 -> FAIL
    levene (H0: equal variances): W= 0.789 , p= 0.4574 -> PASS
    -> will use: Kruskal-Wallis

  metric: adjustment_mean
    shapiro-wilk (H0: normal):
      Habit-driven : W= 0.977 , p= 0.7454 -> PASS
      Price-responsive : W= 0.953 , p= 0.1997 -> PASS
      Social-influenced : W= 0.928 , p= 0.0427 -> FAIL
    levene (H0: equal variances): W= 8.96 , p= 0.0003 -> FAIL
    -> will use: Kruskal-Wallis

--- population: Tipping ---
  metric: flex_mean
    shapiro-wilk (H0: normal):
      Habit-driven : W= 0.92 , p= 0.0272 -> FAIL
      Price-responsive : W= 0.87 , p= 0.0017 -> FAIL
      Social-influenced : W= 0.91 , p= 0.015 -> FAIL
    levene (H0: equal variances): W= 15.322 , p= 0.0 -> FAIL
    -> will use: Kruskal-Wallis

  metric: cost_norm_mean
    shapiro-wilk (H0: normal):
      Habit-driven : W= 0.825 , p= 0.0002 -> FAIL
      Price-responsive : W= 0.883 , p= 0.0033 -> FAIL
      Social-influenced : W= 0.976 , p= 0.7257 -> PASS
    levene (H0: equal variances): W= 2.351 , p= 0.1013 -> PASS
    -> will use: Kruskal-Wallis

  metric: adjustment_mean
    shapiro-wilk (H0: normal):
      Habit-driven : W= 0.941 , p= 0.096 -> PASS
      Price-responsive : W= 0.921 , p= 0.0282 -> FAIL
      Social-influenced : W= 0.972 , p= 0.5816 -> PASS
    levene (H0: equal variances): W= 17.037 , p= 0.0 -> FAIL
    -> will use: Kruskal-Wallis

--- population: Balanced ---
  metric: flex_mean
    shapiro-wilk (H0: normal):
      Habit-driven : W= 0.958 , p= 0.281 -> PASS
      Price-responsive : W= 0.984 , p= 0.9188 -> PASS
      Social-influenced : W= 0.968 , p= 0.4932 -> PASS
    levene (H0: equal variances): W= 10.576 , p= 0.0001 -> FAIL
    -> will use: Kruskal-Wallis

  metric: cost_norm_mean
    shapiro-wilk (H0: normal):
      Habit-driven : W= 0.863 , p= 0.0011 -> FAIL
      Price-responsive : W= 0.898 , p= 0.0077 -> FAIL
      Social-influenced : W= 0.912 , p= 0.0164 -> FAIL
    levene (H0: equal variances): W= 0.825 , p= 0.4416 -> PASS
    -> will use: Kruskal-Wallis

  metric: adjustment_mean
    shapiro-wilk (H0: normal):
      Habit-driven : W= 0.974 , p= 0.6639 -> PASS
      Price-responsive : W= 0.971 , p= 0.5789 -> PASS
      Social-influenced : W= 0.988 , p= 0.9778 -> PASS
    levene (H0: equal variances): W= 17.58 , p= 0.0 -> FAIL
    -> will use: Kruskal-Wallis

assumption check complete
results stored in use_parametric dict
5. Figure 1: headline 4x3 small-multiples
Each panel shows the distribution of run-level means across the 30 runs for each behavioral group, broken out by population (rows) and metric (columns).

Each dot is one simulation run
The diamond shows the mean across all runs in that group
The vertical line shows the 95% bootstrap confidence interval
"""
#metric keys and human readable labels for the columns
metric_keys = list(metric_labels.keys())
metric_names = list(metric_labels.values())

#build the 4 row x 3 column grid
fig, axes = plt.subplots(len(pop_order), len(metric_keys),
                         figsize=(12, 10),
                         sharey=False)

#loop over populations (rows) and metrics (columns)
for row_i, pop in enumerate(pop_order):
    pop_df = df[df["pop_label"] == pop]

    for col_i, metric in enumerate(metric_keys):
        ax = axes[row_i][col_i]

        for g_i, group in enumerate(groups):
            #get the run-level values for this group
            g_vals = pop_df[pop_df["dominant_group"] == group][metric].dropna().values
            x_pos = g_i + 1

            #add small horizontal jitter so overlapping dots stay visible
            jitter = np.random.default_rng(row_i * 10 + col_i + g_i).uniform(
                -0.15, 0.15, len(g_vals))
            ax.scatter(np.full(len(g_vals), x_pos) + jitter, g_vals,
                       color=group_colors[group], alpha=0.35, s=16, zorder=2)

            #compute mean and bootstrap CI for this group
            #This bar is so small that it is not visible in the final plot -> Model is very robust
            mean_v, ci_lo, ci_hi = bootstrap_ci(g_vals)

            #plot the mean as a diamond marker

            ax.plot(x_pos, mean_v,
                    marker="D", 
                    color=group_colors[group],
                    markersize=8,
                    markeredgecolor="black",
                    markeredgewidth=1.0,
                    zorder=4)
            #plot the CI as a vertical line through the diamond
            ax.vlines(x_pos, ci_lo, ci_hi,
                      color=group_colors[group], linewidth=1.8, zorder=3)

        #format individual panel
        ax.set_xticks([1, 2, 3])
        x_tick_labels = []
        for g in groups:
            x_tick_labels.append(group_short[g])
        ax.set_xticklabels(x_tick_labels, fontsize=8)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

        #left column gets the population name as y-axis label
        if col_i == 0:
            ax.set_ylabel(pop, fontsize=9, fontweight="bold", labelpad=8)

        #top row gets the metric name as title
        if row_i == 0:
            ax.set_title(metric_names[col_i], fontsize=9)

#shared legend below the figure
handles = []
for group in groups:
    handles.append(mpatches.Patch(color=group_colors[group], label=group))
fig.legend(handles=handles, loc="lower center", ncol=3,
           fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    "RQ1 - behavioral group differences across populations and metrics\n"
    "N=500 | dots=runs | diamond=mean | bar=95% bootstrap CI",
    fontsize=11)

plt.tight_layout()
plt.savefig(figure_dir / "fig1_rq1_headline.png", bbox_inches="tight")
plt.show()


"""
6. Figure 2: Shift Compositions depending on population
For each behavioral group within each population, what fraction of total flexibility came from price signals vs social influence?
"""
fig, axes = plt.subplots(1, len(pop_order), figsize=(13, 3.5), sharey=True)

for ax_i, pop in enumerate(pop_order):
    ax = axes[ax_i]
    pop_df = df[df["pop_label"] == pop]

    for y_pos, group in enumerate(groups):
        g_vals = pop_df[pop_df["dominant_group"] == group]

        #mean price and social percentage across all runs for this group
        price_pct = g_vals["price_contrib_pct"].mean(skipna=True)
        social_pct = g_vals["social_contrib_pct"].mean(skipna=True)

        #handle missing data  
        if np.isnan(price_pct):
            price_pct = 0.0
        if np.isnan(social_pct):
            social_pct = 0.0

        #normalise 100% for display
        total = price_pct + social_pct
        if total > 0:
            p_norm = price_pct / total * 100
            s_norm = social_pct / total * 100
        else:
            p_norm = 0.0
            s_norm = 0.0

        #plot price segment first
        ax.barh(y_pos, p_norm, color="green", alpha=0.85, height=0.55)
        #plot social segment starting where price ended
        ax.barh(y_pos, s_norm, left=p_norm, color="blue", alpha=0.85, height=0.55)

        #add percentage labels inside each segment if it is wide enough
        if p_norm > 8:
            ax.text(p_norm / 2, y_pos, str(int(round(p_norm))) + "%",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")
        if s_norm > 8:
            ax.text(p_norm + s_norm / 2, y_pos, str(int(round(s_norm))) + "%",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")

    #format panel
    y_tick_positions = []
    y_tick_labels = []
    for i, g in enumerate(groups):
        y_tick_positions.append(i)
        y_tick_labels.append(group_short[g])
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of total shift")
    ax.set_title(pop, fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")

#shared legend below the figure
p_patch = mpatches.Patch(color="green", alpha=0.85, label="Price-driven")
s_patch = mpatches.Patch(color="blue", alpha=0.85, label="Social-driven")
fig.legend(handles=[p_patch, s_patch], loc="lower center", ncol=2,
           frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.14))

fig.suptitle(
    "RQ1 - Shift origin: price-driven vs social-driven per group "
    "(mean % across runs, N=500)",
    fontsize=10.5)
plt.tight_layout()
plt.savefig(figure_dir / "fig2_shift_composition.png", bbox_inches="tight")
plt.show()fig, axes = plt.subplots(1, len(pop_order), figsize=(13, 3.5), sharey=True)

for ax_i, pop in enumerate(pop_order):
    ax = axes[ax_i]
    pop_df = df[df["pop_label"] == pop]

    for y_pos, group in enumerate(groups):
        g_vals = pop_df[pop_df["dominant_group"] == group]

        #mean price and social percentage across all runs for this group
        price_pct = g_vals["price_contrib_pct"].mean(skipna=True)
        social_pct = g_vals["social_contrib_pct"].mean(skipna=True)

        #handle missing data  
        if np.isnan(price_pct):
            price_pct = 0.0
        if np.isnan(social_pct):
            social_pct = 0.0

        #normalise 100% for display
        total = price_pct + social_pct
        if total > 0:
            p_norm = price_pct / total * 100
            s_norm = social_pct / total * 100
        else:
            p_norm = 0.0
            s_norm = 0.0

        #plot price segment first
        ax.barh(y_pos, p_norm, color="green", alpha=0.85, height=0.55)
        #plot social segment starting where price ended
        ax.barh(y_pos, s_norm, left=p_norm, color="blue", alpha=0.85, height=0.55)

        #add percentage labels inside each segment if it is wide enough
        if p_norm > 8:
            ax.text(p_norm / 2, y_pos, str(int(round(p_norm))) + "%",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")
        if s_norm > 8:
            ax.text(p_norm + s_norm / 2, y_pos, str(int(round(s_norm))) + "%",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")

    #format panel
    y_tick_positions = []
    y_tick_labels = []
    for i, g in enumerate(groups):
        y_tick_positions.append(i)
        y_tick_labels.append(group_short[g])
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of total shift")
    ax.set_title(pop, fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")

#shared legend below the figure
p_patch = mpatches.Patch(color="green", alpha=0.85, label="Price-driven")
s_patch = mpatches.Patch(color="blue", alpha=0.85, label="Social-driven")
fig.legend(handles=[p_patch, s_patch], loc="lower center", ncol=2,
           frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.14))

fig.suptitle(
    "RQ1 - Shift origin: price-driven vs social-driven per group "
    "(mean % across runs, N=500)",
    fontsize=10.5)
plt.tight_layout()
plt.savefig(figure_dir / "fig2_shift_composition.png", bbox_inches="tight")
plt.show()

"""
7. Figure 3: Cost-Comfort Trade-off Scatter
This figure visualises the trade-off in RQ1: do groups that achieve lower costs (x-axis) also end up with larger schedule disruptions (y-axis)? The slope of each color cloud shows whether there is a trade-off, paying less might mean shifting your schedule more which is less comfortable.

Figure 1 already captures both dimensions separately, but this scatter adds the relational view between them
"""

pop_markers = {
    "Habitual": "o",
    "Progressive": "s",
    "Tipping": "^",
    "Balanced": "D"}

fig, ax = plt.subplots(figsize=(8, 6))

#one cloud of points per behavioral group
for group in groups:
    g_df = df[df["dominant_group"] == group].dropna(
        subset=["cost_norm_mean", "adjustment_mean"])

    #scatter all points for this group across all populations
    for pop in pop_order:
        p_df = g_df[g_df["pop_label"] == pop]
        ax.scatter(p_df["cost_norm_mean"], p_df["adjustment_mean"],
                   color=group_colors[group],
                   marker=pop_markers[pop],
                   s=30, alpha=0.5, zorder=2)

    #fit a linear trend line per group
    if len(g_df) > 3:
        slope, intercept, r, p_val, se = stats.linregress(
            g_df["cost_norm_mean"], g_df["adjustment_mean"])
        x_range = np.linspace(g_df["cost_norm_mean"].min(),
                              g_df["cost_norm_mean"].max(), 100)
        ax.plot(x_range, slope * x_range + intercept,
                color=group_colors[group], linewidth=1.8,
                alpha=0.9, label=group + " (slope=" + str(round(slope, 2)) + ")")

ax.set_xlabel("Mean normalised cost (price units / kWh)")
ax.set_ylabel("Adjustment day-29 (mean hours from original schedule)")
ax.set_title("RQ1 - cost-comfort trade-off per behavioral group")

#two-part legend: groups by color, populations by marker
group_handles = []
for group in groups:
    group_handles.append(mpatches.Patch(color=group_colors[group], label=group))

pop_handles = []
for pop in pop_order:
    pop_handles.append(
        plt.Line2D([0], [0], marker=pop_markers[pop],
                   color="grey", linestyle="none",
                   markersize=6, label=pop))

leg1 = ax.legend(handles=group_handles, loc="upper right",
                 fontsize=8.5, frameon=True)
ax.add_artist(leg1)
ax.legend(handles=pop_handles, loc="lower left",
          fontsize=8, frameon=True,
          title="Population", title_fontsize=8)

ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(figure_dir / "fig3_cost_comfort_scatter.png", bbox_inches="tight")
plt.show()

"""
8. Statistical Tests
For each population and each metric the test path is chosen by the assumption check from step 4. All turned out to be Kruskal Wallis. I wrote the ANOVA code as well, left here in case the model changed, and if it needs reruns and turns out the outcomes are parametric

Kruskal-Wallis Rank-based equivalent of one-way ANOVA. Converts values to ranks and tests whether the rank distributions differ. Makes no normality assumption.

Mann-Whitney U pairwise Rank-based equivalent of a two-sample t-test, used as the post-hoc.

Bonferroni correction We test 3 metrics per population. Multiplying each p-value by 3 keeps the overall false positive rate at 5%.
"""

stat_rows = []
 
#loop over populations then metrics
for pop in pop_order:
    pop_df = df[df["pop_label"] == pop]
 
    for metric in metric_labels.keys():
        #collect the run-level values per group
        group_data = {}
        for group in groups:
            vals = pop_df[pop_df["dominant_group"] == group][metric].dropna().values
            group_data[group] = vals
 
        #look up which test path was chosen during step 4
        go_parametric = use_parametric.get((pop, metric), True)
        if go_parametric:
            test_used = "ANOVA"
        else:
            test_used = "Kruskal-Wallis"
 
        print("=" * 60)
        print("population:", pop, "| metric:", metric, "| test:", test_used)
 
        #ANOVA IS NOT USED, BUT LEFT HERE IN CASE I CHANGE THE MODEL AND HAVE TO REDO ANALYSIS
 
        if go_parametric:
            #--- parametric path: one-way ANOVA ---
            f_stat, p_anova = stats.f_oneway(
                group_data["Habit-driven"],
                group_data["Price-responsive"],
                group_data["Social-influenced"])
 
            #compute partial eta-squared as effect size for the ANOVA
            #  -> eta_squared = SS_between / SS_total
            #  -> tells us what fraction of total variance is explained by group membership
            all_vals = np.concatenate(list(group_data.values()))
            grand_mean = all_vals.mean()
            ss_between = 0.0
            ss_total = 0.0
            for group in groups:
                vals = group_data[group]
                ss_between = ss_between + len(vals) * (vals.mean() - grand_mean) ** 2
                ss_total = ss_total + ((vals - grand_mean) ** 2).sum()
            if ss_total > 0:
                eta_sq_p = ss_between / ss_total
            else:
                eta_sq_p = np.nan
 
            #bonferroni correction: multiply by number of metrics tested per population
            n_metrics = len(metric_labels)
            p_anova_bon = min(p_anova * n_metrics, 1.0)
            sig_after_bon = p_anova_bon < alpha
 
            print("  F=", round(f_stat, 3),
                  ", p=", round(p_anova, 4),
                  ", p_bonferroni=", round(p_anova_bon, 4),
                  ", eta_sq_p=", round(eta_sq_p, 3))
            print("  significant (after Bonferroni):", sig_after_bon)
 
            stat_rows.append({
                "population": pop,
                "metric": metric,
                "test": "ANOVA",
                "statistic": round(f_stat, 3),
                "p_raw": round(p_anova, 4),
                "p_bonferroni": round(p_anova_bon, 4),
                "effect_size": round(eta_sq_p, 3),
                "effect_label": "eta_sq_p",
                "significant": sig_after_bon,
                "pair": "all groups"})
 
            #if ANOVA significant -> run Tukey HSD pairwise
            if sig_after_bon:
                print("  -> running Tukey HSD post-hoc comparisons")
 
                all_vals_flat = np.concatenate(list(group_data.values()))
                all_labels_flat = []
                for group in groups:
                    for _ in range(len(group_data[group])):
                        all_labels_flat.append(group)
                all_labels_flat = np.array(all_labels_flat)
 
                tukey_result = pairwise_tukeyhsd(
                    all_vals_flat, all_labels_flat, alpha=alpha)
                tukey_table = tukey_result._results_table.data
 
                for row in tukey_table[1:]:
                    g1, g2, meandiff, p_adj, lower, upper, reject = row
                    print("   ", g1, "vs", g2,
                          ": meandiff=", round(meandiff, 3),
                          ", p_adj=", round(p_adj, 4),
                          ", reject=", reject)
 
                    stat_rows.append({
                        "population": "  " + pop,
                        "metric": metric,
                        "test": "Tukey HSD",
                        "statistic": round(meandiff, 3),
                        "p_raw": round(p_adj, 4),
                        "p_bonferroni": round(p_adj, 4),
                        "effect_size": "",
                        "effect_label": "",
                        "significant": bool(reject),
                        "pair": str(g1) + " vs " + str(g2)})
 
        else:
            #--- non-parametric path: Kruskal-Wallis ---
            h_stat, p_kw = stats.kruskal(
                group_data["Habit-driven"],
                group_data["Price-responsive"],
                group_data["Social-influenced"])
 
            n_metrics = len(metric_labels)
            p_kw_bon = min(p_kw * n_metrics, 1.0)
            sig_kw_bon = p_kw_bon < alpha
 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CHANGED: compute epsilon-squared for the Kruskal-Wallis omnibus test and
# add it to the print output and stat_rows dict. Previously effect_size was
# stored as "". k=3 groups, n = total obs across all three groups.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
            #total n across all groups for this (pop, metric) cell
            n_total = sum(len(group_data[g]) for g in groups)
            k_groups = len(groups)
            eps_sq = epsilon_sq_kw(h_stat, k_groups, n_total)
 
            print("  H=", round(h_stat, 3),
                  ", p=", round(p_kw, 4),
                  ", p_bonferroni=", round(p_kw_bon, 4),
                  ", epsilon_sq=", round(eps_sq, 3))
            print("  significant (after Bonferroni):", sig_kw_bon)
 
            stat_rows.append({
                "population": pop,
                "metric": metric,
                "test": "Kruskal-Wallis",
                "statistic": round(h_stat, 3),
                "p_raw": round(p_kw, 4),
                "p_bonferroni": round(p_kw_bon, 4),
                "effect_size": round(eps_sq, 3),
                "effect_label": "epsilon_sq",
                "significant": sig_kw_bon,
                "pair": "all groups"})
 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# END CHANGED SECTION
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
            #if KW significant -> run Mann-Whitney U pairwise
            if sig_kw_bon:
                print("  -> running Mann-Whitney U pairwise comparisons")
 
                #all 3 pairwise combinations of the 3 groups
                pairs = [
                    ("Habit-driven", "Price-responsive"),
                    ("Habit-driven", "Social-influenced"),
                    ("Price-responsive", "Social-influenced")]
 
                for g1, g2 in pairs:
                    mw_stat, mw_p = stats.mannwhitneyu(
                        group_data[g1], group_data[g2], alternative="two-sided")
                    #correct for both metric family AND number of pairs
                    mw_p_bon = min(mw_p * n_metrics * 3, 1.0)
 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CHANGED: replaced cohens_d(g1, g2) with rank_biserial_r for the pairwise
# Mann-Whitney effect size. The print label changes from "d=" to "r_rb=".
# rank_biserial_r needs the raw arrays and the U statistic from mannwhitneyu.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
                    r_rb = rank_biserial_r(group_data[g1], group_data[g2], mw_stat)
 
                    print("   ", g1, "vs", g2,
                          ": p=", round(mw_p, 4),
                          ", p_bon=", round(mw_p_bon, 4),
                          ", r_rb=", round(r_rb, 3))
 
                    stat_rows.append({
                        "population": "  " + pop,
                        "metric": metric,
                        "test": "Mann-Whitney U",
                        "statistic": round(mw_stat, 3),
                        "p_raw": round(mw_p, 4),
                        "p_bonferroni": round(mw_p_bon, 4),
                        "effect_size": round(r_rb, 3),
                        "effect_label": "rank_biserial_r",
                        "significant": mw_p_bon < alpha,
                        "pair": g1 + " vs " + g2})
print()
print("=== statistical tests complete ===")

"""
============================================================
population: Habitual | metric: flex_mean | test: Kruskal-Wallis
  H= 79.121 , p= 0.0 , p_bonferroni= 0.0 , epsilon_sq= 0.886
  significant (after Bonferroni): True
  -> running Mann-Whitney U pairwise comparisons
    Habit-driven vs Price-responsive : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Habit-driven vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Price-responsive vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= 1.0
============================================================
population: Habitual | metric: cost_norm_mean | test: Kruskal-Wallis
  H= 61.748 , p= 0.0 , p_bonferroni= 0.0 , epsilon_sq= 0.687
  significant (after Bonferroni): True
  -> running Mann-Whitney U pairwise comparisons
    Habit-driven vs Price-responsive : p= 0.0 , p_bon= 0.0 , r_rb= 1.0
    Habit-driven vs Social-influenced : p= 0.0207 , p_bon= 0.1861 , r_rb= -0.349
    Price-responsive vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
============================================================
population: Habitual | metric: adjustment_mean | test: Kruskal-Wallis
  H= 78.08 , p= 0.0 , p_bonferroni= 0.0 , epsilon_sq= 0.874
  significant (after Bonferroni): True
  -> running Mann-Whitney U pairwise comparisons
    Habit-driven vs Price-responsive : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Habit-driven vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= -0.973
    Price-responsive vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= 1.0
============================================================
population: Progressive | metric: flex_mean | test: Kruskal-Wallis
  H= 79.121 , p= 0.0 , p_bonferroni= 0.0 , epsilon_sq= 0.886
  significant (after Bonferroni): True
  -> running Mann-Whitney U pairwise comparisons
    Habit-driven vs Price-responsive : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Habit-driven vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Price-responsive vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= 1.0
============================================================
population: Progressive | metric: cost_norm_mean | test: Kruskal-Wallis
  H= 72.07 , p= 0.0 , p_bonferroni= 0.0 , epsilon_sq= 0.805
  significant (after Bonferroni): True
  -> running Mann-Whitney U pairwise comparisons
    Habit-driven vs Price-responsive : p= 0.0 , p_bon= 0.0 , r_rb= 1.0
    Habit-driven vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= 0.802
    Price-responsive vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
============================================================
population: Progressive | metric: adjustment_mean | test: Kruskal-Wallis
  H= 79.121 , p= 0.0 , p_bonferroni= 0.0 , epsilon_sq= 0.886
  significant (after Bonferroni): True
  -> running Mann-Whitney U pairwise comparisons
    Habit-driven vs Price-responsive : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Habit-driven vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Price-responsive vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= 1.0
============================================================
population: Tipping | metric: flex_mean | test: Kruskal-Wallis
  H= 79.121 , p= 0.0 , p_bonferroni= 0.0 , epsilon_sq= 0.886
  significant (after Bonferroni): True
  -> running Mann-Whitney U pairwise comparisons
    Habit-driven vs Price-responsive : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Habit-driven vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Price-responsive vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= 1.0
============================================================
population: Tipping | metric: cost_norm_mean | test: Kruskal-Wallis
  H= 60.26 , p= 0.0 , p_bonferroni= 0.0 , epsilon_sq= 0.67
  significant (after Bonferroni): True
  -> running Mann-Whitney U pairwise comparisons
    Habit-driven vs Price-responsive : p= 0.0 , p_bon= 0.0 , r_rb= 1.0
    Habit-driven vs Social-influenced : p= 0.1537 , p_bon= 1.0 , r_rb= 0.216
    Price-responsive vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
============================================================
population: Tipping | metric: adjustment_mean | test: Kruskal-Wallis
  H= 79.121 , p= 0.0 , p_bonferroni= 0.0 , epsilon_sq= 0.886
  significant (after Bonferroni): True
  -> running Mann-Whitney U pairwise comparisons
    Habit-driven vs Price-responsive : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Habit-driven vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Price-responsive vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= 1.0
============================================================
population: Balanced | metric: flex_mean | test: Kruskal-Wallis
  H= 79.121 , p= 0.0 , p_bonferroni= 0.0 , epsilon_sq= 0.886
  significant (after Bonferroni): True
  -> running Mann-Whitney U pairwise comparisons
    Habit-driven vs Price-responsive : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Habit-driven vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Price-responsive vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= 1.0
============================================================
population: Balanced | metric: cost_norm_mean | test: Kruskal-Wallis
  H= 68.91 , p= 0.0 , p_bonferroni= 0.0 , epsilon_sq= 0.769
  significant (after Bonferroni): True
  -> running Mann-Whitney U pairwise comparisons
    Habit-driven vs Price-responsive : p= 0.0 , p_bon= 0.0 , r_rb= 1.0
    Habit-driven vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= 0.696
    Price-responsive vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
============================================================
population: Balanced | metric: adjustment_mean | test: Kruskal-Wallis
  H= 79.121 , p= 0.0 , p_bonferroni= 0.0 , epsilon_sq= 0.886
  significant (after Bonferroni): True
  -> running Mann-Whitney U pairwise comparisons
    Habit-driven vs Price-responsive : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Habit-driven vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= -1.0
    Price-responsive vs Social-influenced : p= 0.0 , p_bon= 0.0 , r_rb= 1.0

=== statistical tests complete ===

9. Summary Table
"""

summary_rows = []
for pop in pop_order:
    pop_df = df[df["pop_label"] == pop]
    for metric, metric_name in metric_labels.items():
        row = {
            "population": pop,
            "metric": metric_name}
 
        #fill in the mean +/- SD per behavioral group
        for group in groups:
            vals = pop_df[pop_df["dominant_group"] == group][metric].dropna().values
            if len(vals) > 0:
                row[group_short[group]] = (str(round(vals.mean(), 3))
                                            + " +/- "
                                            + str(round(vals.std(ddof=1), 3)))
            else:
                row[group_short[group]] = "n/a"
 
        #pull the omnibus test result for this (pop, metric) cell
        match = []
        for r in stat_rows:
            if r["population"] == pop and r["metric"] == metric and r["pair"] == "all groups":
                match.append(r)
        if len(match) > 0:
            row["test"] = match[0]["test"]
            row["statistic"] = match[0]["statistic"]
            row["p_bonferroni"] = match[0]["p_bonferroni"] 
            row["effect_size"] = match[0]["effect_size"]
            row["effect_label"] = match[0]["effect_label"]
 
            if match[0]["significant"]:
                row["significant"] = "yes"
            else:
                row["significant"] = "no"
        summary_rows.append(row)
 
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(results_dir / "rq1_summary_table.csv", index=False)
 
print("=== RQ1 summary table (mean +/- SD per group) ===")
print(df_summary.to_string(index=False))
 
#also save the full statistical test results
df_stats = pd.DataFrame(stat_rows)
df_stats.to_csv(results_dir / "rq1_statistics.csv", index=False)

"""
=== RQ1 summary table (mean +/- SD per group) ===
 population                      metric           Habit           Price          Social           test  statistic  p_bonferroni  effect_size effect_label significant
   Habitual  Mean flexibility (hrs/day) 1.254 +/- 0.035  5.525 +/- 0.24 2.165 +/- 0.289 Kruskal-Wallis     79.121           0.0        0.886   epsilon_sq         yes
   Habitual Mean norm. cost (units/kWh) 8.772 +/- 0.033 7.929 +/- 0.055 8.798 +/- 0.043 Kruskal-Wallis     61.748           0.0        0.687   epsilon_sq         yes
   Habitual     Adjustment day-29 (hrs) 0.915 +/- 0.022 3.296 +/- 0.063 1.023 +/- 0.062 Kruskal-Wallis     78.080           0.0        0.874   epsilon_sq         yes
Progressive  Mean flexibility (hrs/day) 1.458 +/- 0.041  5.874 +/- 0.13 2.612 +/- 0.121 Kruskal-Wallis     79.121           0.0        0.886   epsilon_sq         yes
Progressive Mean norm. cost (units/kWh) 8.645 +/- 0.037 7.912 +/- 0.022 8.594 +/- 0.026 Kruskal-Wallis     72.070           0.0        0.805   epsilon_sq         yes
Progressive     Adjustment day-29 (hrs)  1.12 +/- 0.028 3.231 +/- 0.074 1.619 +/- 0.038 Kruskal-Wallis     79.121           0.0        0.886   epsilon_sq         yes
    Tipping  Mean flexibility (hrs/day) 1.246 +/- 0.032 5.529 +/- 0.292 2.052 +/- 0.108 Kruskal-Wallis     79.121           0.0        0.886   epsilon_sq         yes
    Tipping Mean norm. cost (units/kWh) 8.777 +/- 0.031  7.946 +/- 0.05 8.769 +/- 0.043 Kruskal-Wallis     60.260           0.0        0.670   epsilon_sq         yes
    Tipping     Adjustment day-29 (hrs) 0.917 +/- 0.021 3.306 +/- 0.087 1.044 +/- 0.029 Kruskal-Wallis     79.121           0.0        0.886   epsilon_sq         yes
   Balanced  Mean flexibility (hrs/day) 1.439 +/- 0.048 5.845 +/- 0.126 2.468 +/- 0.057 Kruskal-Wallis     79.121           0.0        0.886   epsilon_sq         yes
   Balanced Mean norm. cost (units/kWh) 8.663 +/- 0.035 7.933 +/- 0.036 8.617 +/- 0.041 Kruskal-Wallis     68.910           0.0        0.769   epsilon_sq         yes
   Balanced     Adjustment day-29 (hrs) 1.103 +/- 0.032   3.2 +/- 0.074 1.554 +/- 0.035 Kruskal-Wallis     79.121           0.0        0.886   epsilon_sq         yes

Summary
Figure 1 -> Shows mean flexibility (across full sim), mean normalized cost, and adjustment (which is always the final day) across populations

Observations: Social agents don’t really care about price when habit agents are the far majority Price agents show the same pattern across populations, just do their own thing I expected more activity from social in the tipping population (which has a lot of habit agents, 5% price agents and 30% social agents) apparently the noise from habitual agents didn’t give the social agents the opportunity to pivot. Same counts for the progressive plots. Habit withholds social agents from high shifting. Social just needs a good amount of price agents to actually act (the most gain for social agents was in the balanced and progressive setting, though the higher amount of price agents in progressive doesn’t necessarily mean more shifting) This is material for the discussion.

Figure 2 -> Shows shifting contribution per group per population

Observations: This proves the thing about social agents in figure 1. The social shift did not really change when there were more price agents they could copy from. I don’t really understand the price agents’ social shift in habitual and tipping, material for discussion

Figure 3 -> Shows the effect of the population type on the dominant group’s cost and flexibility

Observations:

Price responsive is very stable
Very clear distinction again between the price heavy populations and the less price heavy populations when it comes to social agents. Like with the last two figures: It does not matter how heavy this population is stuffed with price responsive agents. If there is a good amount, then they will show the somewhat same effects regardless of the magnitude of that amount
Habit agents show a similar behavior to social agents, but they are less reactive (as seen with the fitted line)
Social agents have higher adjustment in conservative and progressive populations. In the conservative population however, the price benefit compared to habit agents could turn out worse.

Flexibility: Price-responsive agents shift about 4x more per day than habit-driven agents across all populations. Social-influenced sit clearly in between. All three groups are fully separated in rank (r_rb = about 1.0 on all pairwise flexibility comparisons), confirmed by epsilon_sq values between 0.874 and 0.886 -> group membership explains around 88% of rank variance in flexibility regardless of which population the agents are embedded in.

Normalized cost: Price-responsive agents pay the least, habit-driven and social-influenced pay noticeably more. The separation is slightly weaker here (epsilon_sq between 0.670 and 0.805) because habit and social agents overlap more on cost than on flexibility or adjustment -> this is visible in the r_rb values, which are ~1.0 for Habit vs Price and Price vs Social, but drop to +-0.7 for Habit vs Social in some populations.

Adjustment: Same pattern as flexibility. Price-responsive agents have drifted furthest from their day-0 schedule by day 29, habit-driven the least, social-influenced in between. Again full rank separation. The consistency of these effect sizes across all four populations means the behavioral architecture effect is robust to population composition.

"""
