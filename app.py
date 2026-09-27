import warnings
import time
import json
import re
import csv
import zipfile
import os
import io

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

WCA_EXPORT_META_URL = "https://www.worldcubeassociation.org/api/v0/export/public"
REQUEST_HEADERS = {
    "User-Agent": "competitor-analysis/1.0 (personal analytics project)"
}

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

    # Remove duplicate CDF values so interpolation always has increasing x.
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

    noise = np.random.normal(0, base_noise, 5)
    values = base_samples + noise

    # Occasional poor solves, scaled to the competitor/event instead of
    # hard-coding 10-16 seconds for every event.
    heavy_mask = np.random.rand(5) < heavy_tail_chance
    if np.any(heavy_mask):
        center = np.median(base_samples)
        bad_floor = max(center * 1.15, center + base_noise)
        bad_ceiling = max(center * 1.60, bad_floor + base_noise)

        values[heavy_mask] = np.random.uniform(
            bad_floor,
            bad_ceiling,
            heavy_mask.sum(),
        )

    values = np.maximum(values, 0.01)

    # Ao5: remove the best and worst, average the middle 3.
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

        r2_lookup = {
            competitor_index: round_index
            for round_index, competitor_index in enumerate(r2_indices)
        }

        final_lookup = {
            competitor_index: round_index
            for round_index, competitor_index in enumerate(final_indices)
        }

        for i, name in enumerate(player_names):
            all_results.append(
                {
                    "Competitor": name,
                    "Ao5_Round1": r1_ao5[i],
                    "Ao5_Round2": (
                        r2_ao5[r2_lookup[i]]
                        if i in r2_lookup
                        else np.nan
                    ),
                    "Ao5_Final": (
                        final_ao5[final_lookup[i]]
                        if i in final_lookup
                        else np.nan
                    ),
                    "Advanced_R1": i in r2_lookup,
                    "Advanced_R2": i in final_lookup,
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
    # Base averages across all simulations.
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

    # Probability of reaching the final.
    # Advanced_R2 is True when the competitor appears in final_indices.
    finals_probability = (
        df.groupby("Competitor")["Advanced_R2"]
        .mean()
        .reset_index(name="Finals_Probability")
    )

    # Win probability = percentage of all simulations where placement == 1.
    win_probability = (
        df.assign(Won=df["Final_Placement"].eq(1))
        .groupby("Competitor")["Won"]
        .mean()
        .reset_index(name="Win_Probability")
    )

    # Podium probability = percentage of all simulations where placement is 1-3.
    podium_probability = (
        df.assign(
            Podium=df["Final_Placement"].notna()
            & df["Final_Placement"].le(3)
        )
        .groupby("Competitor")["Podium"]
        .mean()
        .reset_index(name="Podium_Probability")
    )

    df_summary = (
        df_summary
        .merge(finals_probability, on="Competitor", how="left")
        .merge(win_probability, on="Competitor", how="left")
        .merge(podium_probability, on="Competitor", how="left")
    )

    # Estimated rank still uses the best available simulated performance:
    # Final > Round 2 > Round 1.
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

    # Friendly percentage strings for the Streamlit table.
    df_summary["Win %"] = (
        df_summary["Win_Probability"] * 100
    ).map(lambda x: f"{x:.1f}%")

    df_summary["Podium %"] = (
        df_summary["Podium_Probability"] * 100
    ).map(lambda x: f"{x:.1f}%")

    df_summary["Finals %"] = (
        df_summary["Finals_Probability"] * 100
    ).map(lambda x: f"{x:.1f}%")

    return df_summary.sort_values(
        "Estimated_Rank",
        na_position="last",
    )


def display_top_rankings(summary_df):
    st.subheader("🏆 Final Estimated Rankings")

    ranked_df = summary_df.dropna(
        subset=["Estimated_Rank"]
    ).sort_values("Estimated_Rank")

    display_df = ranked_df[
        [
            "Competitor",
            "Estimated_Rank_Display",
            "Win %",
            "Podium %",
            "Finals %",
        ]
    ].copy()

    display_df = display_df.rename(
        columns={
            "Estimated_Rank_Display": "Estimated Rank",
        }
    )

    st.table(
        display_df.reset_index(drop=True)
    )


# ============================================================
# Official WCA results-export loader (v2)
# ============================================================

def _normalize_column_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _column_index(header, *candidates, required=True):
    normalized = {
        _normalize_column_name(name): i
        for i, name in enumerate(header)
    }

    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}. "
            f"Available columns: {header}"
        )

    return None


def _find_tsv_member(names, required_words):
    """
    Find a TSV member without hard-coding the full export filename.
    This makes the code tolerant of prefixes such as WCA_export_....
    """
    candidates = []

    for name in names:
        lower = name.lower()

        if not lower.endswith(".tsv"):
            continue

        base = os.path.basename(lower)

        if all(word in base for word in required_words):
            candidates.append(name)

    if not candidates:
        raise ValueError(
            f"Could not find TSV file containing {required_words} "
            "inside the WCA export."
        )

    # Prefer the shortest matching filename in case the archive has extras.
    return sorted(candidates, key=len)[0]


def _iter_tsv_rows(zip_ref, member_name):
    """
    Yield rows from one TSV file inside the ZIP without extracting it.
    """
    with zip_ref.open(member_name) as raw:
        text = io.TextIOWrapper(
            raw,
            encoding="utf-8-sig",
            newline="",
        )

        reader = csv.reader(
            text,
            delimiter="\t",
        )

        for row in reader:
            yield row


@st.cache_resource(ttl=60 * 60 * 6, show_spinner=False)
def get_wca_export_file():
    """
    Download the current official WCA v2 TSV export once per Cloud Run
    instance (and refresh the cache every 6 hours).

    The large ZIP is streamed to /tmp rather than loaded into RAM.
    """
    meta_response = requests.get(
        WCA_EXPORT_META_URL,
        timeout=60,
        headers=REQUEST_HEADERS,
    )
    meta_response.raise_for_status()
    meta = meta_response.json()

    tsv_url = meta["tsv_url"]
    export_date = meta.get("export_date", "unknown")
    export_format_version = meta.get(
        "export_format_version",
        "unknown",
    )

    safe_date = re.sub(
        r"[^0-9A-Za-z_-]",
        "_",
        str(export_date),
    )

    zip_path = f"/tmp/wca_export_{safe_date}.tsv.zip"

    if not os.path.exists(zip_path):
        with requests.get(
            tsv_url,
            stream=True,
            timeout=600,
            headers=REQUEST_HEADERS,
        ) as response:
            response.raise_for_status()

            with open(zip_path, "wb") as out:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024,
                ):
                    if chunk:
                        out.write(chunk)

    return (
        zip_path,
        export_date,
        export_format_version,
    )


def _competition_date_map(zip_ref, member_name, wanted_competitions):
    """
    Return sortable YYYY-MM-DD-like keys for just the competitions
    represented in the selected competitors' results.
    """
    rows = _iter_tsv_rows(
        zip_ref,
        member_name,
    )

    header = next(rows)

    id_idx = _column_index(
        header,
        "id",
    )

    start_date_idx = _column_index(
        header,
        "start_date",
        "startDate",
        required=False,
    )

    year_idx = _column_index(
        header,
        "year",
        required=False,
    )
    month_idx = _column_index(
        header,
        "month",
        required=False,
    )
    day_idx = _column_index(
        header,
        "day",
        required=False,
    )

    dates = {}

    for row in rows:
        if len(row) <= id_idx:
            continue

        competition_id = row[id_idx]

        if competition_id not in wanted_competitions:
            continue

        if (
            start_date_idx is not None
            and start_date_idx < len(row)
            and row[start_date_idx]
        ):
            dates[competition_id] = row[start_date_idx]
            continue

        if (
            year_idx is not None
            and month_idx is not None
            and day_idx is not None
            and max(year_idx, month_idx, day_idx) < len(row)
        ):
            try:
                dates[competition_id] = (
                    f"{int(row[year_idx]):04d}-"
                    f"{int(row[month_idx]):02d}-"
                    f"{int(row[day_idx]):02d}"
                )
            except ValueError:
                dates[competition_id] = ""

    return dates


def _person_name_map(zip_ref, member_name, wanted_ids):
    rows = _iter_tsv_rows(
        zip_ref,
        member_name,
    )

    header = next(rows)

    wca_id_idx = _column_index(
        header,
        "wca_id",
        "wcaId",
        "id",
    )

    name_idx = _column_index(
        header,
        "name",
    )

    sub_id_idx = _column_index(
        header,
        "sub_id",
        "subid",
        required=False,
    )

    names = {}

    for row in rows:
        if len(row) <= max(wca_id_idx, name_idx):
            continue

        wca_id = row[wca_id_idx]

        if wca_id not in wanted_ids:
            continue

        if (
            sub_id_idx is not None
            and sub_id_idx < len(row)
            and row[sub_id_idx] not in {"", "1"}
        ):
            continue

        names[wca_id] = row[name_idx]

    return names


def _convert_export_attempt(value, event_code):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None

    # WCA export: -1 = DNF, -2 = DNS, 0 = no result.
    if value <= 0:
        return None

    # FMC attempt values are raw move counts.
    if event_code == "333fm":
        return float(value)

    # Normal timed events are centiseconds.
    return value / 100.0


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_recent_solves_from_export(
    player_ids_tuple,
    event_code,
    num_solves,
):
    """
    Read the official WCA Results Export v2 and return only the data
    needed by this simulation.

    Important v2 detail:
      - results contains one row per person/event/round
      - result_attempts contains the individual solves
      - result_attempts.result_id links back to results.id
    """
    player_ids = list(player_ids_tuple)
    player_set = set(player_ids)

    (
        zip_path,
        export_date,
        export_format_version,
    ) = get_wca_export_file()

    # Fail early if WCA introduces a new major export version.
    major_version = str(export_format_version).split(".")[0]

    if major_version not in {"2", "unknown"}:
        raise ValueError(
            "This app currently supports WCA Results Export v2, "
            f"but received version {export_format_version}."
        )

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        names = zip_ref.namelist()

        # result_attempts also contains the word "results" in some naming
        # schemes, so choose results carefully.
        attempts_member = _find_tsv_member(
            names,
            ["result", "attempt"],
        )

        results_candidates = [
            name
            for name in names
            if name.lower().endswith(".tsv")
            and "result" in os.path.basename(name).lower()
            and "attempt" not in os.path.basename(name).lower()
        ]

        if not results_candidates:
            raise ValueError(
                "Could not find the results TSV file in the WCA export."
            )

        results_member = sorted(
            results_candidates,
            key=len,
        )[0]

        competitions_member = _find_tsv_member(
            names,
            ["competition"],
        )

        persons_member = _find_tsv_member(
            names,
            ["person"],
        )

        # ----------------------------------------------------
        # 1) Scan RESULTS once and retain only selected users/event
        # ----------------------------------------------------
        results_rows = _iter_tsv_rows(
            zip_ref,
            results_member,
        )

        results_header = next(results_rows)

        result_id_idx = _column_index(
            results_header,
            "id",
        )

        person_idx = _column_index(
            results_header,
            "person_id",
            "personId",
        )

        event_idx = _column_index(
            results_header,
            "event_id",
            "eventId",
        )

        competition_idx = _column_index(
            results_header,
            "competition_id",
            "competitionId",
        )

        selected_results = {
            player_id: []
            for player_id in player_ids
        }

        wanted_result_ids = set()
        wanted_competitions = set()

        for row in results_rows:
            if len(row) <= max(
                result_id_idx,
                person_idx,
                event_idx,
                competition_idx,
            ):
                continue

            person_id = row[person_idx]

            if person_id not in player_set:
                continue

            if row[event_idx] != event_code:
                continue

            result_id = row[result_id_idx]
            competition_id = row[competition_idx]

            selected_results[person_id].append(
                {
                    "result_id": result_id,
                    "competition_id": competition_id,
                }
            )

            wanted_result_ids.add(result_id)
            wanted_competitions.add(competition_id)

        if not wanted_result_ids:
            return (
                {
                    player_id: {
                        "name": player_id,
                        "solves": [],
                    }
                    for player_id in player_ids
                },
                export_date,
                export_format_version,
            )

        # ----------------------------------------------------
        # 2) Read competition dates for chronological sorting
        # ----------------------------------------------------
        competition_dates = _competition_date_map(
            zip_ref,
            competitions_member,
            wanted_competitions,
        )

        # ----------------------------------------------------
        # 3) Get display names
        # ----------------------------------------------------
        person_names = _person_name_map(
            zip_ref,
            persons_member,
            player_set,
        )

        # ----------------------------------------------------
        # 4) Scan RESULT_ATTEMPTS once and keep selected result IDs
        # ----------------------------------------------------
        attempt_rows = _iter_tsv_rows(
            zip_ref,
            attempts_member,
        )

        attempts_header = next(attempt_rows)

        attempt_result_idx = _column_index(
            attempts_header,
            "result_id",
            "resultId",
        )

        attempt_number_idx = _column_index(
            attempts_header,
            "attempt_number",
            "attemptNumber",
        )

        attempt_value_idx = _column_index(
            attempts_header,
            "value",
        )

        attempts_by_result = {}

        for row in attempt_rows:
            if len(row) <= max(
                attempt_result_idx,
                attempt_number_idx,
                attempt_value_idx,
            ):
                continue

            result_id = row[attempt_result_idx]

            if result_id not in wanted_result_ids:
                continue

            value = _convert_export_attempt(
                row[attempt_value_idx],
                event_code,
            )

            if value is None:
                continue

            try:
                attempt_number = int(
                    row[attempt_number_idx]
                )
            except ValueError:
                attempt_number = 999

            attempts_by_result.setdefault(
                result_id,
                [],
            ).append(
                (attempt_number, value)
            )

        # ----------------------------------------------------
        # 5) Assemble newest solves for each competitor
        # ----------------------------------------------------
        output = {}

        for player_id in player_ids:
            result_entries = selected_results.get(
                player_id,
                [],
            )

            # Newest competitions first. Result ID is used as a
            # deterministic secondary key for multiple rounds on one date.
            result_entries.sort(
                key=lambda entry: (
                    competition_dates.get(
                        entry["competition_id"],
                        "",
                    ),
                    int(entry["result_id"])
                    if str(entry["result_id"]).isdigit()
                    else 0,
                ),
                reverse=True,
            )

            solves = []

            for entry in result_entries:
                attempts = attempts_by_result.get(
                    entry["result_id"],
                    [],
                )

                attempts.sort(
                    key=lambda item: item[0]
                )

                solves.extend(
                    value
                    for _, value in attempts
                )

                if len(solves) >= num_solves:
                    break

            output[player_id] = {
                "name": person_names.get(
                    player_id,
                    player_id,
                ),
                "solves": solves[:num_solves],
            }

        return (
            output,
            export_date,
            export_format_version,
        )


def build_data_and_kde_with_progress(
    group_list,
    event_code,
    num_solves,
):
    progress_bar = st.progress(0)
    status_text = st.empty()
    timer_text = st.empty()

    start_time = time.time()

    status_text.markdown(
        "📦 Loading the official WCA Results Export..."
    )

    (
        competitor_data,
        export_date,
        export_format_version,
    ) = load_recent_solves_from_export(
        tuple(group_list),
        event_code,
        num_solves,
    )

    progress_bar.progress(0.75)

    data_list = []
    kde_list = []
    valid_names = []

    total = len(group_list)

    for i, player_id in enumerate(group_list):
        info = competitor_data.get(
            player_id,
            {
                "name": player_id,
                "solves": [],
            },
        )

        name = info["name"]
        data = info["solves"]

        if data is None or len(data) < 2:
            st.warning(
                f"⚠️ Skipping {name} ({player_id}) — "
                "not enough valid solves for this event."
            )
            continue

        if np.std(data) == 0:
            st.warning(
                f"⚠️ Skipping {name} ({player_id}) — "
                "all selected solve values are identical."
            )
            continue

        try:
            kde = gaussian_kde(
                data,
                bw_method=0.2,
            )
        except Exception as exc:
            st.warning(
                f"⚠️ Skipping {name} ({player_id}) — "
                f"KDE could not be built: {exc}"
            )
            continue

        data_list.append(data)
        kde_list.append(kde)
        valid_names.append(
            f"{name} ({player_id})"
        )

        progress_bar.progress(
            0.75 + 0.25 * ((i + 1) / total)
        )

    elapsed = time.time() - start_time

    status_text.markdown(
        f"✅ Done! Processed **{len(valid_names)} competitors** "
        f"using WCA export **v{export_format_version}**."
    )

    timer_text.markdown(
        f"⏱️ Data Loading Time: **{elapsed:.1f} seconds**  \n"
        f"📅 WCA export date: **{export_date}**"
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
            "🔎 Loading recent solves from the official WCA Results Export..."
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

            padding = max((max(data) - min(data)) * 0.1, 1.0)

            x_values = np.linspace(
                max(0, min(data) - padding),
                max(data) + padding,
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
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    except Exception as e:
        st.error(
            "Unexpected error while running the simulation."
        )
        st.exception(e)
