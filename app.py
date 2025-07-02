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
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from bs4 import BeautifulSoup
import re
import pandas as pd
import gdown



st.title("Rubik's Cube Competitor Analysis")
st.markdown("This is an independent project made by Ryan Saito and not affiliated with the WCA in any way.")
st.write("Similar to sports statisticians, I am working hard to make metrics that accurately predict real-world performance. This project seeks to make a weighted estimated rank based on recent solves instead of lifetime best solves.")
#st.write("### Simulate a future competition: You pick the competitors, you against you and those who have signed up, create your own field")
# st.image("https://i.imgur.com/OYvs0v0.png", use_container_width=True)

# Let user choose input method
st.write("### Step 1: Choose what type of competition you would like to simulate?")
#st.write("If you would like to simulate a real future competition, follow the instructions above and upload an HTML file. If you would like to simulate a competition among certain competitors, enter their WCA IDs manually.")
input_method = st.radio("Choose one:", ["If you would like to simulate a future WCA competition, select this option to upload an HTML file of the competition.", "If you would like to simulate a competition among specific competitors that you choose, select this option to enter their WCA IDs manually."])

user_list = []

if input_method == "If you would like to simulate a future WCA competition, select this option to upload an HTML file of the competition.":
    st.markdown("### Step 2: Load the Data")
    st.write("Go to the World Cube Association website (www.worldcubeassociation.org) and choose a competition under “Competitions” tab.")
    st.write("Once you find the competition you want to simulate, select that competition and click on the “Competitors” tab.")
    st.write("Press CTRL + S to save the HTML file and press Enter while noting where you saved the file. Return back to the Streamlit website to upload the file (not the folder). It should have extracted the WCA IDs.")
    uploaded_file = st.file_uploader("Upload the saved HTML file from a WCA registration page", type="html")
    #st.write("DO **CTRL/CMD + S** TO SAVE HTML FILE")
    #st.image("https://i.imgur.com/xHw6NNt.png", caption="Saint John's Warm Up 2025 - Registrants", use_container_width=True)

    if uploaded_file:
        soup = BeautifulSoup(uploaded_file, "html.parser")
        links = soup.find_all("a", href=True)
        user_list = sorted({
            match.group(1)
            for link in links
            if (match := re.search(r"/persons/([0-9]{4}[A-Z]{4}[0-9]{2})", link["href"]))
        })

        if user_list:
            #df = pd.DataFrame(user_list, columns=["WCA ID"])
            st.success(f"✅ Extracted {len(user_list)} WCA IDs")
            #st.dataframe(df)
        else:
            st.warning("⚠️ No WCA IDs found in the uploaded HTML file.")

elif input_method == "If you would like to simulate a competition among specific competitors that you choose, select this option to enter their WCA IDs manually.":
    st.markdown("### Step 2: Load the Data")
    user_input = st.text_area("Enter WCA IDs separated by commas (e.g., 2018SAIT06, 2022CHAI02)")
    if user_input:
        user_list = [id.strip() for id in user_input.split(",") if id.strip()]
        if user_list:
            st.success(f"✅ Collected {len(user_list)} WCA IDs")
            st.write(user_list)

# --- Behavior-Aware KDE Builder ---
def describe_solver(data):
    mean = np.mean(data)
    std = np.std(data)
    cv = std / mean if mean > 0 else 0
    return mean, std, cv

def build_adaptive_kde(data):
    mean, std, cv = describe_solver(data)
    base_bw = 0.2
    scaled_bw = base_bw + 0.3 * cv  # adapt bandwidth to variability
    return gaussian_kde(data, bw_method=scaled_bw)

# --- Summary and Display ---
def summarize_simulation_results(df):
    # Group and compute mean values for each round
    df_summary = df.groupby('Competitor').agg({
        'Ao5_Round1': 'mean',
        'Ao5_Round2': 'mean',
        'Ao5_Final': 'mean',
        'Advanced_R1': 'mean',
        'Advanced_R2': 'mean',
        'Final_Placement': lambda x: np.nanmean(x)
    }).reset_index()

    # Create a fallback column for ranking: Final > Round2 > Round1
    df_summary['Estimated_Performance'] = (
        df_summary['Ao5_Final']
        .fillna(df_summary['Ao5_Round2'])
        .fillna(df_summary['Ao5_Round1'])
    )

    # Assign rank based on best available performance
    df_summary['Estimated_Rank'] = df_summary['Estimated_Performance'].rank(method="min")

    # Clean display format
    df_summary['Estimated_Rank_Display'] = df_summary['Estimated_Rank'].apply(
        lambda x: int(x) if not pd.isna(x) else "Not Ranked"
    )

    return df_summary.sort_values('Estimated_Rank', na_position="last")

def display_summary_table(summary_df):
    st.subheader("📊 Full Summary")
    st.dataframe(summary_df.style.format(precision=2))

def display_top_rankings(summary_df):
    st.subheader("🏆 Final Estimated Rankings")
    ranked_df = summary_df.dropna(subset=['Estimated_Rank'])  # removes "Not Ranked"
    ranked_df = ranked_df.sort_values('Estimated_Rank')
    display_cols = ['Competitor', 'Estimated_Rank_Display']
    st.table(ranked_df[display_cols].reset_index(drop=True).round(2))

def display_advancement_stats(summary_df):
    adv_df = summary_df[['Competitor','Advanced_R2']].copy()
    adv_df = adv_df.rename(columns={
        "Advanced_R2": "Made Finals"
    })

    adv_df = adv_df.sort_values("Made Finals", ascending=False)

    st.subheader("🔁 Advancement Probabilities (R1 → R2 → Final)")
    st.table(adv_df.reset_index(drop=True).style.format({
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


def build_adaptive_kde(data):
    mean, std, cv = describe_solver(data)
    base_bw = 0.2
    scaled_bw = base_bw + 0.3 * cv  # adapt bandwidth to variability
    return gaussian_kde(data, bw_method=scaled_bw)

def build_percentile_sampler(data, kde):
    x_values = np.linspace(min(data) - 1, max(data) + 1, 1000)
    pdf_values = kde(x_values)
    cdf_values = cumulative_trapezoid(pdf_values, x_values, initial=0)
    cdf_values /= cdf_values[-1]
    cdf_interpolator = interp1d(cdf_values, x_values, bounds_error=False, fill_value=(x_values[0], x_values[-1]))
    return lambda percentile: float(cdf_interpolator(percentile / 100))

# --- Fast Ao5 with Behavioral Variability ---
def fast_simtournament(sampler, base_noise=0.15, heavy_tail_chance=0.05):
    percentiles = np.random.rand(5) * 100
    heavy_mask = np.random.rand(5) < heavy_tail_chance
    base_samples = np.array([sampler(p) for p in percentiles])
    noise = np.random.normal(0, base_noise, 5)
    values = np.where(heavy_mask, np.random.uniform(10, 16, 5), base_samples + noise)
    return round(np.mean(np.sort(values)[1:4]), 2)

# --- Main Simulation ---
def simulate_rounds_behavioral(data_list, player_names, num_simulations, r1_cutoff=60, r2_cutoff=20):
    kde_list = [build_adaptive_kde(data) for data in data_list]
    samplers = [build_percentile_sampler(data, kde) for data, kde in zip(data_list, kde_list)]

    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    for sim_num in range(num_simulations):
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
                "Competitor": name,
                "Ao5_Round1": r1_ao5[i] if i in r1_sorted else np.nan,
                "Ao5_Round2": r2_ao5[r2_indices.tolist().index(i)] if i in r2_indices else np.nan,
                "Ao5_Final": final_ao5[final_indices.index(i)] if i in final_indices else np.nan,
                "Advanced_R1": i in r2_indices,
                "Advanced_R2": i in final_indices,
                "Final_Placement": final_rankings.get(name, np.nan)
            })

        # Update progress every loop
        progress = (sim_num + 1) / num_simulations
        progress_bar.progress(progress)
        status_text.markdown(f"🌀 Running simulation {sim_num+1} of {num_simulations}...")

    end_time = time.time()
    status_text.markdown(f"✅ Finished all {num_simulations} simulations in **{end_time - start_time:.1f} seconds**")
    return pd.DataFrame(all_results)


def get_cstimer_times(file, event, num_solves=25):
    data = file.read().decode("utf-8").strip()
    dictionary = json.loads(data)
    session_data = json.loads(dictionary['properties']['sessionData'].strip())

    session_name = None
    j = 1
    for i in range(1, len(session_data.keys())):
        if session_data[str(i)]['name'] == event:
            session_name = f'session{j}'
            break
        j += 1

    if session_name is None:
        st.error(f"❌ No session matching event '{event}' found in csTimer file.")
        return []

    times_raw = [
        dictionary[session_name][i][0][1] / 1000
        for i in range(1, len(dictionary[session_name]))
    ]
    
    # Show the actual times for debugging
    trimmed_times = times_raw[-num_solves:]
    st.write(f"📋 **csTimer Times Used ({len(trimmed_times)}):** {trimmed_times}")

    return trimmed_times

def build_data_and_kde(group_list, cube_category, times_amount, all_lines, min_solves=10, ):
    data_list = []
    kde_list = []
    valid_names = []

    for player_id in group_list:
        data = get_recent_times(player_id, cube_category, times_amount, all_lines)
        if data is None:
            st.warning(f"⚠️ Skipping {player_id} due to missing/short data")
            continue

        kde = gaussian_kde(data, bw_method=0.2)
        data_list.append(data)
        kde_list.append(kde)
        valid_names.append(player_id)

    return data_list, kde_list, valid_names

def build_data_and_kde_with_progress(group_list, cube_category, times_amount, all_lines, min_solves=10):
    data_list = []
    kde_list = []
    valid_names = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    timer_text = st.empty()
    start_time = time.time()

    total = len(group_list)

    for i, player_id in enumerate(group_list):
        elapsed = time.time() - start_time
        timer_text.markdown(f"⏱️ Elapsed Time: **{elapsed:.1f} seconds**")
        status_text.markdown(f"🔍 Processing `{player_id}` ({i+1} of {total})")

        data, name = get_recent_times_and_name(player_id, cube_category, times_amount, all_lines)

        # Skip if not enough valid times
        if data is None or len(data) < 2:
            st.warning(f"⚠️ Skipping {name or player_id} – not enough valid solves")
            continue

        kde = gaussian_kde(data, bw_method=0.2)
        data_list.append(data)
        kde_list.append(kde)
        valid_names.append(f"{name} ({player_id})")

        progress_bar.progress((i + 1) / total)

    status_text.markdown(f"✅ Done! Processed **{len(valid_names)} competitors**.")
    elapsed = time.time() - start_time
    timer_text.markdown(f"⏱️ Final Elapsed Time: **{elapsed:.1f} seconds**")

    return data_list, kde_list, valid_names


def load_sql_lines_filtered(event_code, user_list, buffer_size=10_000_000, zip_path="file.zip"):
    wca_id_set = set(user_list)
    filtered_lines = []

    progress = st.empty()
    status = st.empty()
    timer = st.empty()
    start_time = time.time()

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                with zip_ref.open(file_name) as f:
                    total = 0
                    for line in f:
                        total += 1
                        if total % 10000 == 0:
                            elapsed = time.time() - start_time
                            status.markdown(f"🔍 Parsed {total:,} lines...")
                            timer.markdown(f"⏱️ Elapsed: {elapsed:.1f} sec")
                            time.sleep(0.01)  # Let Streamlit update

                        try:
                            decoded_line = line.decode("utf-8")
                            if event_code in decoded_line and any(wca_id in decoded_line for wca_id in wca_id_set):
                                filtered_lines.append(decoded_line)
                        except UnicodeDecodeError:
                            continue
    except zipfile.BadZipFile:
        st.error("❌ Invalid ZIP file.")
        st.stop()

    elapsed = time.time() - start_time
    status.markdown(f"✅ Done! Found {len(filtered_lines):,} relevant lines.")
    timer.markdown(f"⏱️ SQL Filtering Time: **{elapsed:.2f} sec**")

    return filtered_lines

    elapsed = time.time() - start_time
    status.markdown(f"✅ Done! Found {len(filtered_lines):,} relevant lines.")
    timer.markdown(f"⏱️ SQL Filtering Time: **{elapsed:.2f} sec**")

    return filtered_lines


def download_file_from_google_drive(file_id, destination="file.zip"):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

st.markdown("### Step 3: Pick your Event")
option = st.selectbox("Which event would you like to analyze?", ("2x2", "3x3", "4x4",'5x5','6x6','7x7','3x3 Blindfolded','FMC','3x3 OH','Clock','Megaminx','Pyraminx','Skewb','Square-1','4x4 Blindfolded','5x5 Blindfolded'),)
new_option = ''
if option == "2x2":
  new_option = '222'
elif option == "3x3":
  new_option = "'333'"
elif option == "4x4":
  new_option = "'444'"
elif option == "5x5":
  new_option = "'555'"
elif option =='6x6':
  new_option = '666'
elif option =='7x7':
  new_option = '777'
elif option =='3x3 Blindfolded':
  new_option = '333bf'
elif option =='FMC':
  new_option = '333fm'
elif option =='3x3 OH':
  new_option = "'333oh'"
elif option =='Clock':
  new_option = 'clock'
elif option =='Megaminx':
  new_option = 'minx'
elif option =='Pyraminx':
  new_option = 'pyra'
  new_option = 'pyram'
elif option =='Skewb':
  new_option = 'skewb'
elif option =='Square-1':
  new_option = 'sq1'
elif option =='4x4 Blindfolded':
  new_option = '444bf'
elif option =='5x5 Blindfolded':
  new_option = '555bf'

#st.write(new_option)

#user_id = st.text_input("WCA ID(s)", "2018SAIT07")
#split_list = user_id.split(', ')
#user_list = []
#for i in split_list:
  #user_list.append(i.strip())

#if st.button("Add User"):
  #st.write(user_list)

st.markdown("### Step 4: Choose your Parameters")

times = st.slider("How many solves of the competitor's most recent solves would you like to include?", 5, 200, 25, step = 5)
new_times = (times / 5) * -1
times_amount = int(new_times)
simulations = st.slider("How many times would you like to simulate this competition?", 10, 500, 250)

st.markdown("### Step 5: Do you want to use your csTimer data as one of the competitors?")
include_cstimer = st.checkbox("Include csTimer times?")
st.checkbox("Do not include csTimer times")
cstimer_file = None

if include_cstimer:
    cstimer_file = st.file_uploader("Upload csTimer File", type=['txt'])
    num_cstimer_solves = st.slider(
        "Number of most recent csTimer solves to include",
        min_value=50, max_value=1000, value=200, step=25
    )

if st.button("Submit"):
    start_time = time.time()
    st.write("⏳ Loading...")
    download_file_from_google_drive("10EPfQTJeFw3hx1Vj_HxKS8sXNx6QRD09", "file.zip")
    st.success(f"✅ Data Loaded!")
    all_lines = load_sql_lines_filtered(new_option, user_list, zip_path="file.zip")
    
    if not include_cstimer:
        data_list, kde_list, player_names = build_data_and_kde_with_progress(user_list, new_option, times_amount, all_lines, simulations)
    else:
        data_list, kde_list, player_names = build_data_and_kde_with_progress(user_list, new_option, times_amount, all_lines, simulations)

    if include_cstimer and cstimer_file is not None:
        grabbed_times = get_cstimer_times(cstimer_file, option, num_cstimer_solves)
        if grabbed_times:
            data_list.append(grabbed_times)
            kde_list.append(build_adaptive_kde(grabbed_times))
            player_names.append("csTimer User")
            st.success("✅ csTimer times loaded and added to simulation")
        else:
            st.warning("⚠️ Could not extract valid csTimer times for this event.")
            
    st.success("✅ Finished Getting KDE + Solves")
    df_simulated = simulate_rounds_behavioral(data_list, player_names, simulations)
    summary_df = summarize_simulation_results(df_simulated)

    st.success("✅ Finished Simulating and Summarizing")

    end_time = time.time()  # ⏱️ End timer
    total_time = end_time - start_time

    st.info(f"🧠 **Processed data for {len(player_names)} competitors**")
    st.info(f"⏲️ **Total runtime: {total_time:.2f} seconds**")

    # Display

    display_top_rankings(summary_df)
    #display_advancement_stats(summary_df)
    #display_summary_table(summary_df)
  # Sample data
    for j, data in enumerate(data_list):

      kde = kde_list[j]
      x_values = np.linspace(min(data) - 1, max(data) + 1, 1000)
      pdf_values = kde(x_values)

      # CDF (for future use if needed)
      cdf_values = cumulative_trapezoid(pdf_values, x_values, initial=0)
      cdf_values /= cdf_values[-1]
      cdf_interpolator = interp1d(x_values, cdf_values)

      # Compute statistics
      mean = np.mean(data)
      std = np.std(data, ddof=1)
      n = len(data)
      z = stats.norm.ppf(0.975)

      ci_lower = mean - z * std / np.sqrt(n)
      ci_upper = mean + z * std / np.sqrt(n)
      pi_lower = mean - z * std * np.sqrt(1 + 1/n)
      pi_upper = mean + z * std * np.sqrt(1 + 1/n)

      # Plot
      fig, ax = plt.subplots(figsize=(7, 4))
      ax.plot(x_values, pdf_values, label="Estimated PDF")
      ax.axvline(mean, color='blue', label='Mean')
      ax.axvline(ci_lower, color='green', linestyle='--', label='95% CI')
      ax.axvline(ci_upper, color='green', linestyle='--')
      ax.axvline(pi_lower, color='orange', linestyle=':', label='95% PI')
      ax.axvline(pi_upper, color='orange', linestyle=':')

      ax.set_xlabel("Solve Time (seconds)")
      ax.set_ylabel("Probability Density")
      ax.set_title(f"KDE for {player_names[j]}")
      ax.legend()
      ax.grid(True)

      # Stats as text
      st.markdown(f"### 📈 Stats for {player_names[j]}")
      st.write(f"**Mean:** {mean:.2f} seconds")
      st.write(f"**95% Confidence Interval:** ({ci_lower:.2f}, {ci_upper:.2f})")
      st.write("A competitor's next average of 5 (Ao5) will fall in the range between the green dotted lines")
      st.write(f"**95% Prediction Interval:** ({pi_lower:.2f}, {pi_upper:.2f})")
      st.write("A competitor's next single solve will fall in the range between the orange dotted lines")

      st.pyplot(fig)
