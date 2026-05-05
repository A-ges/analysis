"""
rq2_aggregate_curves_generator.py

purpose-built generator for the rq2 figure 1 aggregate load curve overlay.
runs all 6 rq2 populations at n=500, seeds 1-10, network variants a/b/c.
saves only the day-29 load profile per run to one single .npz file.

output:
  RQ2 aggregate curves data/rq2_agg_curves.npz
  -> one array per population, shape (30, 96)
     rows = runs (10 seeds x 3 variants), columns = 96 quarter-hour slots
"""

import sys
import numpy as np
from pathlib import Path

#add project root to path so simulation files can be imported
project_root = Path("..")
sys.path.insert(0, str(project_root))

from run_model import run_model
from agent import default_epsilon_habit, default_epsilon_price, default_epsilon_social

#output folder - will be created if it does not exist
output_dir = Path("RQ2 aggregate curves data")
output_dir.mkdir(parents=True, exist_ok=True)

#fixed network size for this figure
net_n = 500

#same seeds and variants as in datagenerator.py
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
variants = ["a", "b", "c"]

#total days per run - same as the main generator
days = 30

#population compositions - same 6 as rq2 in datagenerator.py
populations_rq2 = {
    "Habitual": [90, 5, 5],
    "Progressive": [60, 30, 10],
    "Tipping": [70, 5, 25],
    "Balanced": [50, 25, 25],
    "Price-Maximalist": [20, 70, 10],
    "Social-Maximalist": [20, 10, 70]}

#output path for the single results file
output_path = output_dir / "rq2_agg_curves.npz"

#check if file already exists so we can avoid re-running completed populations
existing_data = {}
if output_path.exists():
    loaded = np.load(output_path, allow_pickle=False)
    for key in loaded.files:
        existing_data[key] = loaded[key]
    print("existing file found, loaded", list(existing_data.keys()))

#count total runs for progress display
#  -> 6 populations x 10 seeds x 3 variants = 180 runs
total_runs = len(populations_rq2) * len(seeds) * len(variants)
done_count = 0

#collect day-29 profiles: one list per population
all_profiles = {}

for pop_label in populations_rq2:
    agents_pct = populations_rq2[pop_label]

    #if this population was already computed and saved, reload and skip
    if pop_label in existing_data:
        all_profiles[pop_label] = existing_data[pop_label]
        done_count = done_count + len(seeds) * len(variants)
        print(pop_label, "-> already done, skipping")
        continue

    #collect day-29 load profiles for all runs of this population
    pop_profiles = []

    for seed in seeds:
        for variant in variants:
            done_count = done_count + 1
            net_code = str(net_n) + variant

            print("[" + str(done_count) + "/" + str(total_runs) + "]",
                  pop_label, "|", net_code, "| seed", seed)

            #run the simulation with shifting enabled (same epsilons as datagenerator.py)
            df_agents, df_daily, load_profiles, df_pricing = run_model(
                agents_pct=agents_pct,
                network_code=net_code,
                days=days,
                graphs=None,
                median_plot=False,
                random_state=seed,
                epsilon_habit=default_epsilon_habit,
                epsilon_price=default_epsilon_price,
                epsilon_social=default_epsilon_social)

            #take the last day (day index 29) load profile
            #  -> shape is (96,): aggregate kW per 15-minute slot
            day29_profile = load_profiles[days - 1]
            pop_profiles.append(day29_profile)

    #convert to numpy array of shape (30, 96) and store
    all_profiles[pop_label] = np.array(pop_profiles)

    #save incrementally after each population so progress is not lost on interruption
    #  -> re-include any populations already saved before
    save_dict = {}
    for key in all_profiles:
        save_dict[key] = all_profiles[key]

    np.savez(output_path, **save_dict)
    print("  saved progress after", pop_label)

print()
print("all done ->", output_path)
print("array shapes per population:")

for pop_label in all_profiles:
    arr = all_profiles[pop_label]
    print(" ", pop_label, "->", arr.shape, "(runs x slots)")
