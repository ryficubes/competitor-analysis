import warnings
import time
import json
import re
import io
import base64

warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import scipy.stats as stats
from bs4 import BeautifulSoup


# ============================================================
# Configuration
# ============================================================

WCA_API_BASE = "https://wca-rest-api.robiningelbrecht.be"

st.set_page_config(
    page_title="Rubik's Cube Competitor Analysis",
    layout="wide",
)


# ============================================================
# Page header
# ============================================================

st.title("Rubik's Cube Competitor Analysis")
st.markdown(
    "This is an independent project made by Ryan Saito and not affiliated "
    "with the WCA in any way."
)
st.write(
    "Similar to sports statisticians, I am working hard to make metrics that "
    "accurately predict real-world performance. This project seeks to make a "
    "weighted estimated rank based on recent solves instead of lifetime best solves."
)


# ============================================================
# Step 1 / Step 2: competitors
# ============================================================

st.write("### Step 1: Choose what type of competition you would like to simulate?")

input_method = st.radio(
    "Choose one:",
    [
        "If you would like to simulate a future WCA competition, select this option to upload an HTML file of the competition.",
        "If you would like to simulate a competition among specific competitors that you choose, select this option to enter their WCA IDs manually.",
    ],
)

user_list = []

if input_method.startswith("If you would like to simulate a future WCA competition"):
    st.markdown("### Step 2: Load the Data")
    st.image("https://i.imgur.com/9ATfnS8.gif", use_container_width=True)

    st.write(
        "Go to the World Cube Association website "
        "(www.worldcubeassociation.org/competitions) and choose a competition "
        "that you want to simulate."
    )
    st.write(
        "Once you find the competition you want to simulate, select that "
        "competition and click on the “Competitors” tab."
    )
    st.write(
        "Press **CTRL/CMD + S** to save the HTML file. Return here and upload "
        "the **HTML FILE (not the folder)**."
    )

    uploaded_file = st.file_uploader(
        "Upload the saved HTML file from a WCA registration page",
        type="html",
    )

    if uploaded_file:
        soup = BeautifulSoup(uploaded_file, "html.parser")
        links = soup.find_all("a", href=True)

        user_list = sorted(
            {
                match.group(1)
                for link in links
                if (
                    match := re.search(
                        r"/persons/([0-9]{4}[A-Z]{4}[0-9]{2})",
                        link["href"],
                    )
                )
            }
        )

        if user_list:
            st.success(f"✅ Extracted {len(user_list)} WCA IDs")
            st.dataframe(pd.DataFrame(user_list, columns=["WCA ID"]))
        else:
            st.warning("⚠️ No WCA IDs found in the uploaded HTML file.")

else:
    st.markdown("### Step 2: Load the Data")

    user_input = st.text_area(
        "Enter WCA IDs separated by commas (e.g., 2018SAIT06, 2022CHAI02)"
    )

    if user_input:
        user_list = [
            wca_id.strip().upper()
            for wca_id in user_input.split(",")
            if wca_id.strip()
        ]

        if user_list:
            st.success(f"✅ Collected {len(user_list)} WCA IDs")


# ============================================================
# KDE / simulation helpers
# ============================================================

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

    cdf_values = cumulative_trapezoid(
        pdf_values,
        x_values,
        initial=0,
    )

    if cdf_values[-1] <= 0:
        raise ValueError("Could not build a valid probability distribution.")

    cdf_values /= cdf_values[-1]

    # interp1d requires increasing x values. KDE CDF values can occasionally
    # contain tiny duplicate regions, so remove duplicates first.
    cdf_values, unique_indices = np.unique(cdf_values, return_index=True)
    x_values = x_values[unique_indices]

    cdf_interpolator = interp1d(
        cdf_values,
        x_values,
        bounds_error=False,
        fill_value=(x_values[0], x_values[-1]),
    )

    return lambda percentile: float(cdf_interpolator(percentile / 100))


def fast_simtournament(sampler, base_noise=0.15, heavy_tail_chance=0.05):
    percentiles = np.random.rand(5) * 100
    base_samples = np.array([sampler(p) for p in percentiles])

    # Small solve-to-solve noise.
    noise = np.random.normal(0, base_noise, 5)
    values = base_samples + noise

    # Preserve the original idea of occasional bad solves, but make them
    # relative to the solver rather than forcing every event into 10-16 sec.
    heavy_mask = np.random.rand(5) < heavy_tail_chance
    if np.any(heavy_mask):
        normal_center = np.median(base_samples)
        bad_solve_floor = max(normal_center * 1.15, normal_center + base_noise)
        bad_solve_ceiling = max(normal_center * 1.60, bad_solve_floor + base_noise)
        values[heavy_mask] = np.random.uniform(
            bad_solve_floor,
            bad_solve_ceiling,
            heavy_mask.sum(),
        )

    # Times / scores should never become negative from random noise.
    values = np.maximum(values, 0.01)

    # Ao5 = drop best and worst, average middle 3.
    return round(np.mean(np.sort(values)[1:4]), 2)


def simulate_rounds_behavioral(
    data_list,
    player_names,
    num_simulations,
    r1_cutoff=60,
    r2_cutoff=20,
):
    kde_list = [build_adaptive_kde(data) for data in data_list]
    samplers = [
        build_percentile_sampler(data, kde)
        for data, kde in zip(data_list, kde_list)
    ]

    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    start_time = time.time()

    for sim_num in range(num_simulations):
        r1_ao5 = [fast_simtournament(sampler) for sampler in samplers]

        r1_sorted = np.argsort(r1_ao5)
        r2_indices = r1_sorted[: min(r1_cutoff, len(r1_ao5))]

        r2_ao5 = [
            fast_simtournament(samplers[i])
            for i in r2_indices
        ]

        r2_sorted = np.argsort(r2_ao5)

        final_indices = [
            r2_indices[i]
            for i in r2_sorted[: min(r2_cutoff, len(r2_ao5))]
        ]

        final_ao5 = [
            fast_simtournament(samplers[i])
            for i in final_indices
        ]

        final_rankings = {
            player_names[i]: rank + 1
            for rank, (i, _) in enumerate(
                sorted(
                    zip(final_indices, final_ao5),
                    key=lambda x: x[1],
                )
            )
        }

        r2_index_lookup = {
            competitor_index: round_index
            for round_index, competitor_index in enumerate(r2_indices)
        }

        final_index_lookup = {
            competitor_index: round_index
            for round_index, competitor_index in enumerate(final_indices)
        }

        for i, name in enumerate(player_names):
            all_results.append(
                {
                    "Competitor": name,
                    "Ao5_Round1": r1_ao5[i],
                    "Ao5_Round2": (
                        r2_ao5[r2_index_lookup[i]]
                        if i in r2_index_lookup
                        else np.nan
                    ),
                    "Ao5_Final": (
                        final_ao5[final_index_lookup[i]]
                        if i in final_index_lookup
                        else np.nan
                    ),
                    "Advanced_R1": i in r2_index_lookup,
                    "Advanced_R2": i in final_index_lookup,
                    "Final_Placement": final_rankings.get(name, np.nan),
                }
            )

        progress_bar.progress((sim_num + 1) / num_simulations)
        status_text.markdown(
            f"🌀 Running simulation {sim_num + 1} of {num_simulations}..."
        )

    elapsed = time.time() - start_time

    status_text.markdown(
        f"✅ Finished all {num_simulations} simulations in "
        f"**{elapsed:.1f} seconds**"
    )

    return pd.DataFrame(all_results)


def summarize_simulation_results(df):
    df_summary = (
        df.groupby("Competitor")
        .agg(
            {
                "Ao5_Round1": "mean",
                "Ao5_Round2": "mean",
                "Ao5_Final": "mean",
                "Advanced_R1": "mean",
                "Advanced_R2": "mean",
                "Final_Placement": lambda x: np.nanmean(x),
            }
        )
        .reset_index()
    )

    df_summary["Estimated_Performance"] = (
        df_summary["Ao5_Final"]
        .fillna(df_summary["Ao5_Round2"])
        .fillna(df_summary["Ao5_Round1"])
    )

    df_summary["Estimated_Rank"] = df_summary[
        "Estimated_Performance"
    ].rank(method="min")

    df_summary["Estimated_Rank_Display"] = df_summary[
        "Estimated_Rank"
    ].apply(
        lambda x: int(x) if not pd.isna(x) else "Not Ranked"
    )

    return df_summary.sort_values(
        "Estimated_Rank",
        na_position="last",
    )


def display_top_rankings(summary_df):
    st.subheader("🏆 Final Estimated Rankings")

    ranked_df = summary_df.dropna(
        subset=["Estimated_Rank"]
    ).sort_values("Estimated_Rank")

    display_cols = [
        "Competitor",
        "Estimated_Rank_Display",
    ]

    st.table(
        ranked_df[display_cols]
        .reset_index(drop=True)
    )


# ============================================================
# WCA API helpers
# ============================================================

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_person_data(player_id):
    """
    Fetch one competitor's static JSON file from the unofficial WCA REST API.
    The API is updated daily, so caching for 12 hours is reasonable.
    """
    url = f"{WCA_API_BASE}/persons/{player_id}.json"

    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "competitor-analysis/1.0"},
    )

    if response.status_code == 404:
        return None

    response.raise_for_status()
    return response.json()


def convert_wca_solve_value(value, event_code):
    """
    Convert a WCA solve value into the units used by this app.

    For timed events, WCA values are stored in centiseconds, so divide by 100.
    FMC values are move counts, so keep them as-is.

    DNF / DNS / missing values are negative or zero and are ignored.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None

    if value <= 0:
        return None

    if event_code == "333fm":
        return float(value)

    return value / 100.0


def get_recent_times_from_api(
    player_id,
    event_code,
    num_solves,
):
    """
    Return the competitor's most recent valid solves for an event.

    The person endpoint groups results by competition. We follow the API's
    competitionIds ordering, then flatten the selected event's rounds and keep
    the last num_solves valid attempts.
    """
    person = fetch_person_data(player_id)

    if not person:
        return None, None

    name = person.get("name", player_id)
    results = person.get("results", {}) or {}

    # Prefer competitionIds so we use the API's intended competition ordering.
    competition_ids = [
        comp_id
        for comp_id in person.get("competitionIds", [])
        if comp_id in results
    ]

    # Include any result keys not present in competitionIds.
    seen = set(competition_ids)
    competition_ids.extend(
        comp_id
        for comp_id in results.keys()
        if comp_id not in seen
    )

    solves = []

    for comp_id in competition_ids:
        competition_results = results.get(comp_id, {})
        event_results = competition_results.get(event_code, [])

        if not isinstance(event_results, list):
            continue

        for round_result in event_results:
            if not isinstance(round_result, dict):
                continue

            for raw_solve in round_result.get("solves", []):
                solve = convert_wca_solve_value(
                    raw_solve,
                    event_code,
                )

                if solve is not None:
                    solves.append(solve)

    if not solves:
        return None, name

    return solves[-num_solves:], name


def build_data_and_kde_with_progress(
    group_list,
    event_code,
    num_solves,
):
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

        timer_text.markdown(
            f"⏱️ Elapsed Time: **{elapsed:.1f} seconds**"
        )

        status_text.markdown(
            f"🔍 Loading {player_id} ({i + 1} of {total})"
        )

        try:
            data, name = get_recent_times_from_api(
                player_id,
                event_code,
                num_solves,
            )
        except requests.RequestException as exc:
            st.warning(
                f"⚠️ Could not load {player_id}: {exc}"
            )
            progress_bar.progress((i + 1) / total)
            continue

        if data is None or len(data) < 2:
            st.warning(
                f"⚠️ Skipping {name or player_id} — "
                "not enough valid solves for this event."
            )
            progress_bar.progress((i + 1) / total)
            continue

        # gaussian_kde can fail if all values are identical.
        if np.std(data) == 0:
            st.warning(
                f"⚠️ Skipping {name or player_id} — "
                "all selected solve values are identical."
            )
            progress_bar.progress((i + 1) / total)
            continue

        kde = gaussian_kde(
            data,
            bw_method=0.2,
        )

        data_list.append(data)
        kde_list.append(kde)
        valid_names.append(
            f"{name} ({player_id})"
        )

        progress_bar.progress((i + 1) / total)

    elapsed = time.time() - start_time

    status_text.markdown(
        f"✅ Done! Processed **{len(valid_names)} competitors**."
    )

    timer_text.markdown(
        f"⏱️ Data Loading Time: **{elapsed:.1f} seconds**"
    )

    return data_list, kde_list, valid_names


# ============================================================
# csTimer helper
# ============================================================

def get_cstimer_times(file, event, num_solves=25):
    data = file.read().decode("utf-8").strip()
    dictionary = json.loads(data)

    session_data = json.loads(
        dictionary["properties"]["sessionData"].strip()
    )

    session_name = None
    j = 1

    for i in range(1, len(session_data.keys())):
        if session_data[str(i)]["name"] == event:
            session_name = f"session{j}"
            break

        j += 1

    if session_name is None:
        st.error(
            f"❌ No session matching event '{event}' "
            "found in csTimer file."
        )
        return []

    times_raw = [
        dictionary[session_name][i][0][1] / 1000
        for i in range(1, len(dictionary[session_name]))
    ]

    trimmed_times = times_raw[-num_solves:]

    st.write(
        f"📋 **csTimer Times Used ({len(trimmed_times)}):** "
        f"{trimmed_times}"
    )

    return trimmed_times


# ============================================================
# Step 3: event
# ============================================================

st.markdown("### Step 3: Pick your Event")

EVENT_CODES = {
    "2x2": "222",
    "3x3": "333",
    "4x4": "444",
    "5x5": "555",
    "6x6": "666",
    "7x7": "777",
    "3x3 Blindfolded": "333bf",
    "FMC": "333fm",
    "3x3 OH": "333oh",
    "Clock": "clock",
    "Megaminx": "minx",
    "Pyraminx": "pyram",
    "Skewb": "skewb",
    "Square-1": "sq1",
    "4x4 Blindfolded": "444bf",
    "5x5 Blindfolded": "555bf",
}

option = st.selectbox(
    "Which event would you like to analyze?",
    tuple(EVENT_CODES.keys()),
)

new_option = EVENT_CODES[option]


# ============================================================
# Step 4: parameters
# ============================================================

st.markdown("### Step 4: Choose your Parameters")

times = st.slider(
    "How many of each competitor's most recent solves would you like to include?",
    min_value=5,
    max_value=200,
    value=25,
    step=5,
)

simulations = st.slider(
    "How many times would you like to simulate this competition?",
    min_value=100,
    max_value=1000,
    value=500,
)


# ============================================================
# Step 5: csTimer
# ============================================================

st.markdown(
    "### Step 5: Do you want to use your csTimer data "
    "as one of the competitors?"
)

include_cstimer = st.checkbox(
    "Include csTimer times?"
)

cstimer_file = None
num_cstimer_solves = 200

if include_cstimer:
    cstimer_file = st.file_uploader(
        "Upload csTimer File",
        type=["txt"],
    )

    num_cstimer_solves = st.slider(
        "Number of most recent csTimer solves to include",
        min_value=50,
        max_value=1000,
        value=200,
        step=25,
    )


# ============================================================
# Submit
# ============================================================

if st.button("Submit"):
    try:
        if not user_list:
            st.error(
                "Please provide at least one WCA ID "
                "(via HTML upload or manual entry)."
            )
            st.stop()

        start_time = time.time()

        st.write(
            "🔎 Loading competitor results from the WCA results API..."
        )

        data_list, kde_list, player_names = (
            build_data_and_kde_with_progress(
                user_list,
                new_option,
                times,
            )
        )

        if include_cstimer:
            if cstimer_file is None:
                st.warning(
                    "⚠️ csTimer is enabled, but no csTimer file was uploaded."
                )
            else:
                grabbed_times = get_cstimer_times(
                    cstimer_file,
                    option,
                    num_cstimer_solves,
                )

                if len(grabbed_times) >= 2 and np.std(grabbed_times) > 0:
                    data_list.append(grabbed_times)
                    kde_list.append(
                        build_adaptive_kde(grabbed_times)
                    )
                    player_names.append("csTimer User")
                    st.success(
                        "✅ csTimer times loaded and added"
                    )
                else:
                    st.warning(
                        "⚠️ Could not extract enough valid csTimer "
                        "times for this event."
                    )

        if not data_list:
            st.error(
                "No valid time series were built. Check the WCA IDs "
                "and selected event."
            )
            st.stop()

        st.success("✅ Finished Getting KDE + Solves")

        df_simulated = simulate_rounds_behavioral(
            data_list,
            player_names,
            simulations,
        )

        summary_df = summarize_simulation_results(
            df_simulated
        )

        st.success(
            "✅ Finished Simulating and Summarizing"
        )

        total_time = time.time() - start_time

        st.info(
            f"🧠 Processed {len(player_names)} competitors — "
            f"⏲️ {total_time:.2f}s total"
        )

        display_top_rankings(summary_df)

        # ====================================================
        # Plots
        # ====================================================

        for j, data in enumerate(data_list):
            kde = kde_list[j]

            x_values = np.linspace(
                min(data) - 1,
                max(data) + 1,
                1000,
            )

            pdf_values = kde(x_values)

            mean = np.mean(data)
            std = np.std(data, ddof=1)
            n = len(data)

            z = stats.norm.ppf(0.975)

            ci_lower = mean - z * std / np.sqrt(n)
            ci_upper = mean + z * std / np.sqrt(n)

            pi_lower = (
                mean
                - z * std * np.sqrt(1 + 1 / n)
            )

            pi_upper = (
                mean
                + z * std * np.sqrt(1 + 1 / n)
            )

            fig, ax = plt.subplots(
                figsize=(12, 8)
            )

            ax.plot(
                x_values,
                pdf_values,
                label="Estimated PDF",
            )

            ax.axvline(
                mean,
                label="Mean",
            )

            ax.axvline(
                ci_lower,
                linestyle="--",
                label="95% CI",
            )

            ax.axvline(
                ci_upper,
                linestyle="--",
            )

            ax.axvline(
                pi_lower,
                linestyle=":",
                label="95% PI",
            )

            ax.axvline(
                pi_upper,
                linestyle=":",
            )

            unit_label = (
                "Moves"
                if new_option == "333fm"
                else "Solve Time (s)"
            )

            ax.set_xlabel(unit_label)
            ax.set_ylabel("Density")
            ax.set_title(
                f"KDE for {player_names[j]}"
            )

            ax.legend()
            ax.grid(True)

            st.markdown(
                f"### 📈 Stats for {player_names[j]}"
            )

            unit_suffix = (
                " moves"
                if new_option == "333fm"
                else "s"
            )

            st.write(
                f"**Mean:** {mean:.2f}{unit_suffix}"
            )

            st.write(
                f"**95% CI:** "
                f"({ci_lower:.2f}, {ci_upper:.2f})"
            )

            st.write(
                f"**95% PI:** "
                f"({pi_lower:.2f}, {pi_upper:.2f})"
            )

            fig.tight_layout()

            # st.pyplot is simpler and avoids embedding a huge base64 string.
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    except Exception as e:
        st.error(
            "Unexpected error while running the simulation."
        )
        st.exception(e)
