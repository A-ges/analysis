"""
datagenerator.py
This file runs the full simulation grid for both research questions and the qualitative
dynamics and saves summary CSVs for the analysis notebooks to load

every actual simulation run for RQ2 is paired with a matched baseline run
  -> matched means the same population composition and the same random state
  -> the only difference is that epsilon_price = epsilon_social = 0 in the baseline
  -> this isolates the effect of behavioral shifting from agent-composition effects

"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wasserstein_distance

#point to the project root so the simulation files can be imported
project_root = Path("..")
sys.path.insert(0, str(project_root))

#import run model and its default parameters that I have defined in the agent.py file
from run_model import run_model
from agent import default_epsilon_habit, default_epsilon_price, default_epsilon_social

#create the output directories where everything will be saved
results_dir = Path("results")
repr_dir = results_dir / "representative"
#storing baselines
baselines_dir = results_dir / "baselines"
results_dir.mkdir(parents=True, exist_ok=True)
repr_dir.mkdir(parents=True, exist_ok=True)
baselines_dir.mkdir(parents=True, exist_ok=True)

print("output directories ready")

#define population compositions for RQ1
#each entry is [habit_pct, price_pct, social_pct] and sums to 100
populations_rq1 = {
    "Habitual": [90, 5, 5],
    "Progressive": [60, 30, 10],
    "Tipping": [70, 5, 25],
    "Balanced": [50, 25, 25]}

#define population compositions for RQ2
#includes the four plausible populations from RQ1 plus two extreme populations to look into
populations_rq2 = {
    "Habitual": [90, 5, 5],
    "Progressive": [60, 30, 10],
    "Tipping": [70, 5, 25],
    "Balanced": [50, 25, 25],
    "Price-Maximalist": [20, 70, 10],
    "Social-Maximalist": [20, 10, 70]}

#network sizes used per research question
#use only 500 as a size to investigate RQ1, size does have an effect, mainly because networks will grow. But after a certain while
#the daily network will always be filled
#RQ2 sweeps the full size range to test how aggregate effects change with network size
n_rq1 = [500]
n_rq2 = [100, 250, 500, 750, 1000]

#using multiple variants prevents any single network topology from driving the results
variants = ["a", "b", "c"]

#Network variant used for baseline runs becasue network has no influence (social epsilon = 0)

baseline_variant = "a"

#build the list of random seeds explicitly with a for loop
#10 seeds per (population, network_code) cell gives 10 independent runs
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#days of simulation
#do 30 day sims
#  -> on day 0 only the habit shift is applied and price/social shifts have not happened yet
days = 30

#representative run parameters -> these runs are used by qualitativedynamics.ipynb
#picked as Progressive because it has all three groups well represented in a somewhat plausible form
#and shows the shifting dynamics

repr_pop = "Progressive"
repr_net = "500a"

#RQ1 needs no baselines because notebook RQ1.ipynb compares behavioral groups within a population
#RQ2 needs a matched baseline for every actual run to compute a fair EMD. The baseline has to be
#generated for every combination of habit parameter, N and random state
rq1_runs = len(populations_rq1) * len(n_rq1) * len(variants) * len(seeds)
rq2_actual = len(populations_rq2) * len(n_rq2) * len(variants) * len(seeds)
rq2_baselines = len(populations_rq2) * len(n_rq2) * len(seeds)
rq2_total = rq2_actual + rq2_baselines
repr_runs = len(seeds)

print("rq1 runs (no baselines)    :", rq1_runs)
print("rq2 actual runs            :", rq2_actual)
print("rq2 matched baseline runs  :", rq2_baselines, "(shared across variants on the same pop/N/seed)")
print("representative runs        :", repr_runs)
print("total                      :", rq1_runs + rq2_total + repr_runs)


#running a single simulation
def run_one(agents_pct, network_code, seed, is_baseline):
    #pick the right epsilons based on whether this is a baseline run or not
    #  baseline -> epsilon_price = epsilon_social = 0, agents do not shift
    #  actual   -> default epsilon values, agents respond to price and social
    if is_baseline:
        ep = 0.0
        es = 0.0
    else:
        ep = default_epsilon_price
        es = default_epsilon_social

    df_agents, df_daily, load_profiles, df_pricing = run_model(
        agents_pct=agents_pct,
        network_code=network_code,
        days=days,
        graphs=None,
        median_plot=False,
        random_state=seed,
        epsilon_habit=default_epsilon_habit,
        epsilon_price=ep,
        epsilon_social=es)

    return df_agents, df_daily, load_profiles, df_pricing


#CHANGE: helper that loads or creates the matched baseline for a given (pop_label, N, seed)
#  -> the baseline is generated on variant "a" and reused across variants b and c
#  -> cached to disk so that an interrupted run can resume without redoing the baseline
def get_or_create_baseline(pop_label, agents_pct, net_n, seed):
    #file paths for the cached baseline data
    #  -> daily.parquet stores df_daily, load.npy stores the (days x 96) aggregate load array
    #  -> these are the only two outputs needed by compute_rq2_row
    base_name = pop_label + "_N" + str(net_n) + "_seed" + str(seed)
    daily_path = baselines_dir / (base_name + "_daily.parquet")
    load_path = baselines_dir / (base_name + "_load.npy")

    #if both cache files exist load from disk and return immediately without rerunning
    if daily_path.exists() and load_path.exists():
        b_daily = pd.read_parquet(daily_path)
        b_load = np.load(load_path)
        return b_daily, b_load

    #otherwise generate the baseline run on variant "a" and save it for reuse
    net_code_baseline = str(net_n) + baseline_variant
    print("    -> generating baseline:", pop_label, "|", net_code_baseline, "| seed", seed)
    _, b_daily, b_load, _ = run_one(agents_pct, net_code_baseline, seed, True)

    #save both to disk for restart safety
    b_daily.to_parquet(daily_path, index=False)
    np.save(load_path, b_load)

    return b_daily, b_load


#computing the per-group summary rows for one RQ1 simulation run
#produces three rows (one per dominant group) per simulation run
def compute_rq1_rows(pop_label, network_code, seed, df_agents):
    #list to collect the output rows
    rows = []

    df_active = df_agents.copy() #to isolate processing from the raw simulation data

    #compute per-agent means
    grouped = df_active.groupby("agent_id")

    #build a list of one dict per agent
    per_agent_list = []
    for agent_id, agent_data in grouped:
        #every row for one agent has the same dominant_group so taking the first is safe
        group_name = agent_data["dominant_group"].iloc[0]

        #assemble the per-agent summary directly
        agent_row = {
            "agent_id": agent_id,
            "dominant_group": group_name,
            "individual_flexibility": agent_data["individual_flexibility"].mean(),
            "individual_cost_normalized": agent_data["individual_cost_normalized"].mean(),
            "price_shift_contribution": agent_data["price_shift_contribution"].mean(),
            "social_shift_contribution": agent_data["social_shift_contribution"].mean(),
            "savings_per_flex": agent_data["savings_per_flex"].mean()}
        per_agent_list.append(agent_row)

    #convert the list of dicts to a DataFrame for easy filtering
    per_agent = pd.DataFrame(per_agent_list)

    #the adjustment metric is only computed by the model on the last simulated day
    #  -> last day index is days-1
    df_last = df_agents[df_agents["day"] == days - 1].copy()

    #produce one summary row per behavioral group
    for group in ["Habit-driven", "Price-responsive", "Social-influenced"]:
        #pull the subset of agents that belong to this group
        g_active = per_agent[per_agent["dominant_group"] == group]
        g_last = df_last[df_last["dominant_group"] == group]

        #means across the agents in this group
        total_flex = g_active["individual_flexibility"].mean()
        price_contrib = g_active["price_shift_contribution"].mean()
        social_contrib = g_active["social_shift_contribution"].mean()
        total_shift = price_contrib + social_contrib

        #compute the percentage breakdown of price-driven vs social-driven shifting
        #  -> only meaningful when the group actually shifts at all
        #  -> habit-driven groups in low-shifting populations may have total_shift = 0
        if total_shift > 0:
            price_pct = (price_contrib / total_shift) * 100
            social_pct = (social_contrib / total_shift) * 100
        else:
            price_pct = np.nan
            social_pct = np.nan

        adjustment_value = g_last["individual_adjustment"].mean()

        #savings_per_flex can have NaN values when an agent did not shift much that day
        #  -> filter those out before computing the mean to avoid biased results
        savings_vals = g_active["savings_per_flex"].values
        savings_vals = savings_vals[~np.isnan(savings_vals)]
        if len(savings_vals) > 0:
            savings_mean = savings_vals.mean()
        else:
            savings_mean = np.nan

        #assemble the row as a dict for the final df
        row_dict = {
            "pop_label": pop_label,
            "network_code": network_code,
            "N": int(network_code[:-1]),
            "seed": seed,
            "dominant_group": group,
            "flex_mean": total_flex,
            "cost_norm_mean": g_active["individual_cost_normalized"].mean(),
            "adjustment_mean": adjustment_value,
            "price_contrib_mean": price_contrib,
            "social_contrib_mean": social_contrib,
            "price_contrib_pct": price_pct,
            "social_contrib_pct": social_pct,
            "savings_per_flex_mean": savings_mean}
        rows.append(row_dict)

    return rows


#computing the system-level row for one RQ2 simulation run
#takes both the actual run data and the matched baseline data
#  -> matched baseline = same population, same N, same seed (always variant "a"), no shifting
#  -> EMD is computed against the matched baseline so it isolates the effect of shifting
def compute_rq2_row(pop_label, network_code, seed, df_daily, load_profiles,
                    baseline_daily, baseline_load):

    #use all 30 days because total_flexibility on day 0 is genuinely zero
    #  -> aggregating across all days gives a stable system-level signal
    d = df_daily.copy()

    #adjustment is only computed on the last simulated day
    adj_last = df_daily[df_daily["day"] == days - 1]["mean_adjustment"].values
    if len(adj_last) > 0:
        adj_val = float(adj_last[0])
    else:
        adj_val = np.nan

    #CHANGE: EMD is now computed as the final shifted day vs the median of the baseline run
    #  -> run_curve  = the actual load profile on the last simulated day (day 29)
    #  -> baseline_curve = the median profile across all 30 baseline days (typical no-shift day)
    #  -> this captures how far the final shifted state has drifted from the baseline typical day
    #  -> the old version used median of both, which diluted the effect of learning across days
    run_curve = load_profiles[days - 1]
    baseline_curve = np.median(baseline_load, axis=0)

    #normalise both curves to probability distributions over the 96 quarter-hour slots
    #  -> this means EMD captures shape difference rather than magnitude difference
    #  -> protects against zero-sum corner cases with an explicit if check
    if run_curve.sum() > 0:
        run_p = run_curve / run_curve.sum()
    else:
        run_p = run_curve
    if baseline_curve.sum() > 0:
        baseline_p = baseline_curve / baseline_curve.sum()
    else:
        baseline_p = baseline_curve

    #scipy's wasserstein_distance expects the support points and weights
    #  -> support is the slot indices 0 to 95
    slots = np.arange(96)
    emd = wasserstein_distance(slots, slots, run_p, baseline_p)

    #peak hour shift compared to the matched baseline on the last day
    #  -> negative means the peak moved earlier in the day
    #  -> needed to interpret PAR (high PAR is fine if peak moved to cheap midday hours)
    baseline_ph = baseline_daily[baseline_daily["day"] == days - 1]["peak_hour"].values
    run_ph = df_daily[df_daily["day"] == days - 1]["peak_hour"].values

    if len(baseline_ph) > 0:
        baseline_ph_val = float(baseline_ph[0])
    else:
        baseline_ph_val = np.nan
    if len(run_ph) > 0:
        run_ph_val = float(run_ph[0])
    else:
        run_ph_val = np.nan
    peak_hour_shift = run_ph_val - baseline_ph_val

    #assemble the system-level row directly as a dict literal
    row_dict = {
        "pop_label": pop_label,
        "network_code": network_code,
        "N": int(network_code[:-1]),
        "seed": seed,
        "par_mean": float(d["par"].mean()),
        "peak_hour_mean": float(d["peak_hour"].mean()),
        "peak_hour_shift": peak_hour_shift,
        "mean_cost_norm": float(d["mean_cost_norm"].mean()),
        "mean_price_advantage": float(d["mean_price_advantage"].mean()),
        "total_flexibility": float(d["total_flexibility"].mean()),
        "mean_adjustment": adj_val,
        "emd_vs_baseline": emd}

    return row_dict


#run the representative simulations for all seeds in seeds
#  -> each seed is cached independently so a partial run resumes cleanly
#  -> all four output objects are saved per seed for full flexibility in qualitativedynamics.ipynb
#  -> CHANGE: no legacy migration needed since we are starting fresh with seeds [1..10]
for repr_seed in seeds:
    agents_path = repr_dir / ("agents_seed" + str(repr_seed) + ".parquet")
    daily_path = repr_dir / ("daily_seed" + str(repr_seed) + ".parquet")
    pricing_path = repr_dir / ("pricing_seed" + str(repr_seed) + ".parquet")
    load_path = repr_dir / ("load_seed" + str(repr_seed) + ".npy")

    #skip if all four files already exist from a previous run
    if agents_path.exists() and daily_path.exists() and pricing_path.exists() and load_path.exists():
        print("representative seed", repr_seed, "already cached - skipping")
        continue

    print("running representative:", repr_pop, "|", repr_net, "| seed", repr_seed)
    df_a, df_d, lp, df_pr = run_one(
        populations_rq1[repr_pop],
        repr_net,
        repr_seed,
        False)

    #write each output object to its own per-seed file in the representative directory
    df_a.to_parquet(agents_path, index=False)
    df_d.to_parquet(daily_path, index=False)
    df_pr.to_parquet(pricing_path, index=False)
    np.save(load_path, lp)
    print("  saved")


#run the RQ1 simulation grid
#  -> RQ1 does not need baselines because notebook RQ1.ipynb compares behavioral groups within
#     a population, not against a no-shift state
rq1_csv = results_dir / "rq1_run_summaries.csv"

#load any previous progress so we can resume if the script was interrupted
#  -> each row in the CSV corresponds to one (pop_label, network_code, seed, dominant_group)
#  -> RQ1 produces 3 rows per run (one per behavioral group) so we use a set of run keys
#  -> completed at the run level rather than the row level
completed_keys_rq1 = set()
if rq1_csv.exists():
    df_existing = pd.read_csv(rq1_csv)
    for i in range(len(df_existing)):
        key = (df_existing.iloc[i]["pop_label"],
               df_existing.iloc[i]["network_code"],
               int(df_existing.iloc[i]["seed"]))
        completed_keys_rq1.add(key)
    print("rq1 cache found with", len(completed_keys_rq1), "completed runs - will resume")

#build the rq1 grid as a list of tuples
rq1_grid = []
for net_n in n_rq1:
    for variant in variants:
        net_code = str(net_n) + variant
        for seed in seeds:
            for pop_label, pct in populations_rq1.items():
                rq1_grid.append((pop_label, pct, net_code, seed))

#progress tracking counters
total_rq1 = len(rq1_grid)
done_rq1 = 0

#run every cell in the grid that has not yet been completed
for pop_label, agents_pct, network_code, seed in rq1_grid:
    done_rq1 = done_rq1 + 1

    #skip if this exact run has already been completed in a previous invocation
    if (pop_label, network_code, seed) in completed_keys_rq1:
        print("[" + str(done_rq1) + "/" + str(total_rq1) + "] " + pop_label
              + " | " + network_code + " | seed " + str(seed) + " -> already done")
        continue

    #print a progress line and execute the run
    print("[" + str(done_rq1) + "/" + str(total_rq1) + "] " + pop_label
          + " | " + network_code + " | seed " + str(seed))
    df_a, df_d, lp, df_pr = run_one(agents_pct, network_code, seed, False)

    #compute the three summary rows for this run
    rows = compute_rq1_rows(pop_label, network_code, seed, df_a)

    #append immediately to the CSV so progress is preserved on every run
    #  -> if the script crashes the next iteration the resume logic picks up from here
    rows_df = pd.DataFrame(rows)
    header_needed = not rq1_csv.exists()
    rows_df.to_csv(rq1_csv, mode="a", header=header_needed, index=False)
    completed_keys_rq1.add((pop_label, network_code, seed))

print("rq1 complete ->", rq1_csv)


#run the RQ2 simulation grid
#  -> each (pop_label, N, seed) shares ONE matched baseline across variants a, b and c
#  -> the baseline is loaded or generated by get_or_create_baseline()
#  -> the only difference between baseline and actual is epsilon_price = epsilon_social = 0
#CHANGE: loop order is now (N -> seed -> pop -> variant) so the baseline is computed once per
#CHANGE: (pop, N, seed) and then reused for all three variants in the inner loop
rq2_csv = results_dir / "rq2_run_summaries.csv"

#load any previous progress
completed_keys_rq2 = set()
if rq2_csv.exists():
    df_existing = pd.read_csv(rq2_csv)
    for i in range(len(df_existing)):
        key = (df_existing.iloc[i]["pop_label"],
               df_existing.iloc[i]["network_code"],
               int(df_existing.iloc[i]["seed"]))
        completed_keys_rq2.add(key)
    print("rq2 cache found with", len(completed_keys_rq2), "completed runs - will resume")

#count total iterations for the progress display
#  -> total here is the number of ACTUAL runs (baseline runs are additional one-offs)
total_rq2 = len(populations_rq2) * len(n_rq2) * len(variants) * len(seeds)
done_rq2 = 0

#CHANGE: outer loops are N -> seed -> pop, inner loop is variant
#  -> for a fixed (N, seed, pop) the baseline is identical regardless of variant a, b or c
#  -> we check whether any variant in this cell still needs running before loading the baseline
for net_n in n_rq2:
    for seed in seeds:
        for pop_label, pct in populations_rq2.items():

            #check if at least one variant in this (N, seed, pop) cell still needs a run
            #  -> only then do we load the baseline into memory
            #CHANGE: explicit for loop instead of comprehension - no any() allowed
            cell_needs_run = False
            for variant in variants:
                net_code_check = str(net_n) + variant
                if (pop_label, net_code_check, seed) not in completed_keys_rq2:
                    cell_needs_run = True
                    break

            if cell_needs_run:
                #fetch or create the matched baseline ONCE for this (pop, N, seed)
                #  -> reused for all variants in the inner loop below
                baseline_daily, baseline_load = get_or_create_baseline(
                    pop_label, pct, net_n, seed)
            else:
                #all three variants are already done, no need to load anything
                baseline_daily = None
                baseline_load = None

            #now run all three variants against the same baseline
            for variant in variants:
                done_rq2 = done_rq2 + 1
                net_code = str(net_n) + variant

                #skip if this exact run has already been completed
                if (pop_label, net_code, seed) in completed_keys_rq2:
                    print("[" + str(done_rq2) + "/" + str(total_rq2) + "] " + pop_label
                          + " | " + net_code + " | seed " + str(seed) + " -> already done")
                    continue

                #print a progress line and run the actual simulation with shifting enabled
                print("[" + str(done_rq2) + "/" + str(total_rq2) + "] " + pop_label
                      + " | " + net_code + " | seed " + str(seed))
                df_a, df_d, lp, _ = run_one(pct, net_code, seed, False)

                #compute the system-level row using both the actual run and the shared baseline
                row = compute_rq2_row(
                    pop_label, net_code, seed, df_d, lp, baseline_daily, baseline_load)

                #append to the file immediately so progress is preserved on interruption
                row_df = pd.DataFrame([row])
                header_needed = not rq2_csv.exists()
                row_df.to_csv(rq2_csv, mode="a", header=header_needed, index=False)
                completed_keys_rq2.add((pop_label, net_code, seed))

print("rq2 complete ->", rq2_csv)


#load the final outputs for verification
df_rq1 = pd.read_csv(results_dir / "rq1_run_summaries.csv")
df_rq2 = pd.read_csv(results_dir / "rq2_run_summaries.csv")

#print a quick summary of what was produced
print("=== rq1 summary ===")
print(df_rq1.groupby(["pop_label", "dominant_group"]).size().rename("n_runs").to_string())
print()
print("=== rq2 summary ===")
print(df_rq2.groupby(["pop_label", "N"]).size().rename("n_runs").to_string())
print()
print("all simulations complete")
