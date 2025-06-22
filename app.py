import warnings
import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import seaborn as sns
import pandas as pd
import json
import requests, zipfile, io
import scipy.stats as stats
from bs4 import BeautifulSoup
import re

warnings.filterwarnings("ignore")

# ---------------- Streamlit Header ----------------
st.title("Rubik's Cube Competitor Analysis")
st.write("""
Similar to sports statisticians, we are working hard to make metrics that accurately predict real-world performance.
This project builds a weighted estimated rank based on recent solves instead of lifetime best solves.
""")
st.image("https://i.imgur.com/OYvs0v0.png", use_container_width=True)

# ---------------- Input Method ----------------
st.markdown("### Choose how you want to input competitor data")
st.write("Upload a WCA registration HTML file or enter WCA IDs manually.")
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
            if (match := re.search(r"/persons/([0-9]{4}[A-Z]{4}[0-9]{2})", link["href"))
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

# ---------------- KDE Helper Functions ----------------
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
    heavy_mask = np.random.rand(5) < heavy_tail_chance
    base_samples = np.array([sampler(p) for p in percentiles])
    noise = np.random.normal(0, base_noise, 5)
    values = np.where(heavy_mask, np.random.uniform(10, 16, 5), base_samples + noise)
    return round(np.mean(np.sort(values)[1:4]), 2)

# ---------------- csTimer Integration ----------------
def get_cstimer_times(file, event):
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

    times_list = [dictionary[session_name][i][0][1] / 1000 for i in range(1, len(dictionary[session_name]))]
    return times_list


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

times = st.slider("How many solves (competitor's most recent solves) would you like to include in the model?", 5, 200, 5, step = 5)
new_times = (times / 5) * -1
times_amount = int(new_times)
simulations = st.slider("How many simulations would you like to include?", 10, 500, 50)

include_cstimer = st.checkbox("Include csTimer times?")
cstimer_file = None
if include_cstimer:
    cstimer_file = st.file_uploader("Upload csTimer File", type=['txt'])

if st.button("Submit"):
    start_time = time.time()  # ⏱️ Start timer
    st.write("⏳ Loading...")

    # Step 1: Get latest export info
    r = requests.get("https://www.worldcubeassociation.org/api/v0/export/public").json()
    sql_url = r["sql_url"]

    # Step 2: Download and unzip
    response = requests.get(sql_url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for name in z.namelist():
            if name.endswith(".sql"):
                z.extract(name, ".")

    # Step 3: Read SQL content
    with open('WCA_export.sql', 'r') as file:
        all_lines = file.readlines()

    st.success("✅ Data Loaded!")

    if not include_cstimer:
        data_list, kde_list, player_names = build_data_and_kde_with_progress(user_list, new_option, times_amount, all_lines, simulations)
    else:
        data_list, kde_list, player_names = build_data_and_kde_with_progress(user_list, new_option, times_amount, all_lines, simulations)

    if include_cstimer and cstimer_file is not None:
        grabbed_times = get_cstimer_times(cstimer_file, option)
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
    display_advancement_stats(summary_df)
    display_summary_table(summary_df)
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
      st.write(f"**95% Prediction Interval:** ({pi_lower:.2f}, {pi_upper:.2f})")

      st.pyplot(fig)
