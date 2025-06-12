
import warnings
import time
warnings.filterwarnings("ignore")
import streamlit as st
import numpy as np
import random as randomx
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import requests, zipfile, io
import scipy.stats as stats
from bs4 import BeautifulSoup
import re

r = requests.get("https://www.worldcubeassociation.org/api/v0/export/public").json()
sql_url = r["sql_url"]
response = requests.get(sql_url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    for name in z.namelist():
        if name.endswith(".sql"):
            z.extract(name, ".")
with open('WCA_export.sql', 'r') as file:
    all_lines = file.readlines()
st.success("✅ Data Loaded!")

st.title("Rubik's Cube Competitor Analysis")
st.write("Similar to sports statisticians, we are working hard to make metrics that accurately predict real-world performance. This project wanted to make a weighted estimated rank based on recent solves instead of lifetime best solves.")

st.markdown("### Choose how you want to input competitor data")
input_method = st.radio("Select input method:", ["Upload HTML File", "Enter WCA IDs Manually"])
user_list = []

if input_method == "Upload HTML File":
    uploaded_file = st.file_uploader("Upload the saved HTML file from a WCA registration page", type="html")
    if uploaded_file:
        soup = BeautifulSoup(uploaded_file, "html.parser")
        links = soup.find_all("a", href=True)
        user_list = sorted({
            match.group(1)
            for link in links
            if (match := re.search(r"/persons/([0-9]{4}[A-Z]{4}[0-9]{2})", link["href"]))
        })
        if user_list:
            df = pd.DataFrame(user_list, columns=["WCA ID"])
            st.success(f"✅ Extracted {len(user_list)} WCA IDs")
            st.dataframe(df)
        else:
            st.warning("⚠️ No WCA IDs found in the uploaded HTML file.")
elif input_method == "Enter WCA IDs Manually":
    user_input = st.text_area("Enter WCA IDs separated by commas (e.g., 2018SAIT06, 2022CHAI02)")
    if user_input:
        user_list = [id.strip() for id in user_input.split(",") if id.strip()]
        if user_list:
            st.success(f"✅ Collected {len(user_list)} WCA IDs")
            st.write(user_list)

def describe_solver(data):
    mean = np.mean(data)
    std = np.std(data)
    cv = std / mean if mean > 0 else 0
    return mean, std, cv

def build_adaptive_kde(data):
    mean, std, cv = describe_solver(data)
    base_bw = 0.2
    scaled_bw = base_bw + 0.3 * cv
    return gaussian_kde(data, bw_method=scaled_bw)

def build_percentile_sampler(data, kde):
    x_values = np.linspace(min(data) - 1, max(data) + 1, 1000)
    pdf_values = kde(x_values)
    cdf_values = cumulative_trapezoid(pdf_values, x_values, initial=0)
    cdf_values /= cdf_values[-1]
    cdf_interpolator = interp1d(cdf_values, x_values, bounds_error=False, fill_value=(x_values[0], x_values[-1]))
    return lambda percentile: float(cdf_interpolator(percentile / 100))

def fast_simtournament(sampler, base_noise=0.15, heavy_tail_chance=0.05):
    percentiles = np.random.rand(5) * 100
    samples = []
    for p in percentiles:
        if np.random.rand() < heavy_tail_chance:
            val = np.random.uniform(10, 16)
        else:
            val = sampler(p) + np.random.normal(0, base_noise)
        samples.append(val)
    samples = sorted(samples)
    return round(np.mean(samples[1:4]), 2)

def simulate_rounds_behavioral(data_list, player_names, num_simulations=1000, r1_cutoff=60, r2_cutoff=20):
    kde_list = [build_adaptive_kde(data) for data in data_list]
    samplers = [build_percentile_sampler(data, kde) for data, kde in zip(data_list, kde_list)]
    all_results = []
    for sim_id in range(num_simulations):
        r1_ao5 = [fast_simtournament(s) for s in samplers]
        r1_sorted = np.argsort(r1_ao5)
        r2_indices = r1_sorted[:min(r1_cutoff, len(r1_ao5))]
        r2_ao5 = [fast_simtournament(samplers[i]) for i in r2_indices]
        r2_sorted = np.argsort(r2_ao5)
        final_indices = [r2_indices[i] for i in r2_sorted[:min(r2_cutoff, len(r2_ao5))]]
        final_ao5 = [fast_simtournament(samplers[i]) for i in final_indices]
        final_rankings = {player_names[i]: rank+1 for rank, (i, _) in enumerate(sorted(zip(final_indices, final_ao5), key=lambda x: x[1]))}
        for i, name in enumerate(player_names):
            all_results.append({
                "Simulation_ID": sim_id + 1,
                "Competitor": name,
                "Ao5_Round1": r1_ao5[i] if i in r1_sorted else np.nan,
                "Ao5_Round2": r2_ao5[r2_indices.tolist().index(i)] if i in r2_indices else np.nan,
                "Ao5_Final": final_ao5[final_indices.index(i)] if i in final_indices else np.nan,
                "Advanced_R1": i in r2_indices,
                "Advanced_R2": i in final_indices,
                "Final_Placement": final_rankings.get(name, np.nan)
            })
    return pd.DataFrame(all_results)

def summarize_simulation_results(df):
    df_summary = df.groupby('Competitor').agg({
        'Ao5_Round1': 'mean',
        'Ao5_Round2': 'mean',
        'Ao5_Final': 'mean',
        'Advanced_R1': 'mean',
        'Advanced_R2': 'mean',
        'Final_Placement': 'mean'
    }).reset_index()

    # Handle NaNs gracefully
    if df_summary['Final_Placement'].isna().all():
        df_summary['Estimated_Rank'] = np.nan
        df_summary['Estimated_Rank_Display'] = "Not Ranked"
    else:
        df_summary['Estimated_Rank'] = df_summary['Final_Placement'].rank(method="min")
        df_summary['Estimated_Rank_Display'] = df_summary['Estimated_Rank'].apply(
            lambda x: str(int(x)) if not pd.isna(x) else "Not Ranked"
        )

    return df_summary.sort_values('Estimated_Rank', na_position="last")

def display_summary_tables(summary_df):
    st.subheader("📊 Full Summary")
    st.dataframe(summary_df.astype({"Estimated_Rank_Display": "string"}).style.format(precision=2))

    st.subheader("🏆 Final Estimated Rankings")
    if 'Estimated_Rank' in summary_df.columns:
        summary_df = summary_df.sort_values('Estimated_Rank', na_position="last")

    st.table(summary_df[['Competitor', 'Final_Placement', 'Estimated_Rank_Display']])

    st.subheader("🔁 Advancement Probabilities")
    adv_df = summary_df[['Competitor', 'Advanced_R1', 'Advanced_R2']].rename(columns={
        "Advanced_R1": "Advanced to Round 2",
        "Advanced_R2": "Made Finals"
    })
    adv_df = adv_df.sort_values("Made Finals", ascending=False)
    st.table(adv_df.style.format({
        "Advanced to Round 2": "{:.0%}",
        "Made Finals": "{:.0%}"
    }))

def get_recent_times_and_name(player_id, cube_category, times_amount, all_lines):
    pulled_lines = []
    for line in all_lines:
        if player_id in line:
            parts = line.split(',')
            if len(parts) > 2 and parts[1].strip().strip("'") == cube_category.strip().strip("'"):
                pulled_lines.append(line)
    if not pulled_lines:
        return None, None
    most_recent = pulled_lines[times_amount:]
    times = []
    name = None
    for entry in most_recent:
        parts = entry.split(',')
        times += parts[10:15]
        if not name:
            name = parts[6].strip().strip("'")
    try:
        int_list = np.asarray([int(x) for x in times if x.strip().isdigit()])
    except ValueError:
        return None, name
    int_list = int_list[int_list > 0]
    if len(int_list) == 0:
        return None, name
    return [x / 100 for x in int_list], name

# Options
event_map = {
    "2x2": "222", "3x3": "'333'", "4x4": "'444'", "5x5": "'555'", "6x6": "666", "7x7": "777",
    "3x3 Blindfolded": "333bf", "FMC": "333fm", "3x3 OH": "'333oh'", "Clock": "clock",
    "Megaminx": "minx", "Pyraminx": "pyram", "Skewb": "skewb", "Square-1": "sq1",
    "4x4 Blindfolded": "444bf", "5x5 Blindfolded": "555bf"
}
option = st.selectbox("Which event would you like to analyze?", list(event_map.keys()))
new_option = event_map[option]

times = st.slider("How many solves (competitor's most recent solves) would you like to include in the model?", 5, 200, 5, step=5)
times_amount = int((times / 5) * -1)
simulations = st.slider("How many simulations would you like to include?", 10, 500, 50)

if st.button("Submit"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    data_list, kde_list, valid_names = [], [], []
    for i, pid in enumerate(user_list):
        status_text.markdown(f"🔍 Processing {pid} ({i+1}/{len(user_list)})")
        data, name = get_recent_times_and_name(pid, new_option, times_amount, all_lines)
        if data is None or len(data) < 2:
            continue
        kde = gaussian_kde(data, bw_method=0.2)
        data_list.append(data)
        kde_list.append(kde)
        valid_names.append(f"{name} ({pid})")
        progress_bar.progress((i + 1) / len(user_list))
    st.success("✅ Finished Getting KDE + Solves")
    df_simulated = simulate_rounds_behavioral(data_list, valid_names, simulations)
    summary_df = summarize_simulation_results(df_simulated)

selected = st.multiselect("📈 Select competitors to view KDE graph and stats:", valid_names)
for j, name in enumerate(valid_names):
    if name not in selected:
        continue
    data = data_list[j]
    kde = kde_list[j]
    x_values = np.linspace(min(data) - 1, max(data) + 1, 1000)
    pdf_values = kde(x_values)
    mean, std, n = np.mean(data), np.std(data, ddof=1), len(data)
    z = stats.norm.ppf(0.975)
    ci_lower, ci_upper = mean - z * std / np.sqrt(n), mean + z * std / np.sqrt(n)
    pi_lower, pi_upper = mean - z * std * np.sqrt(1 + 1/n), mean + z * std * np.sqrt(1 + 1/n)
    fig, ax = plt.subplots()
    ax.plot(x_values, pdf_values, label="Estimated PDF")
    ax.axvline(mean, color='blue', label='Mean')
    ax.axvline(ci_lower, color='green', linestyle='--', label='95% CI')
    ax.axvline(ci_upper, color='green', linestyle='--')
    ax.axvline(pi_lower, color='orange', linestyle=':', label='95% PI')
    ax.axvline(pi_upper, color='orange', linestyle=':')
    ax.set_xlabel("Solve Time (seconds)")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"KDE for {name}")
    ax.legend()
    st.markdown(f"### 📈 Stats for {name}")
    st.write(f"**Mean:** {mean:.2f} seconds")
    st.write(f"**95% Confidence Interval:** ({ci_lower:.2f}, {ci_upper:.2f})")
    st.write(f"**95% Prediction Interval:** ({pi_lower:.2f}, {pi_upper:.2f})")
    st.pyplot(fig)
display_summary_tables(summary_df)
