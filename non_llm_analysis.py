import pandas as pd
import numpy as np
import csv
import io
import warnings
import re
from collections import Counter
from scipy.stats import entropy
import json
import matplotlib.pyplot as plt

# --- Global Variables & Configuration ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 0)

# This would be passed as an argument or configured for cloud environment
# For now, defaulting to a placeholder.
DEFAULT_INPUT_CSV_PATH = "global_dialogue_data.csv"
PADDING_ROWS = 9 # Specific to the CSV structure in the notebook

# Codebook for Thematic Coding
codebook = {
    "ECON_OPPORTUNITY": {
        "keywords": ["job creation", "economic growth", "increased productivity", "new business opportunities", "boost our economy", "new kinds of jobs"],
        "description": "Captures mentions of positive economic impacts, such as job creation, economic growth, increased productivity, or new business opportunities."
    },
    "ECON_THREAT": {
        "keywords": ["job displacement", "wage depression", "increased economic inequality", "losing my job to a robot", "job loss"],
        "description": "Captures mentions of negative economic impacts, such as job displacement, wage depression, or increased economic inequality."
    },
    "GOVERNANCE_REGULATION": {
        "keywords": ["rules", "laws", "government oversight", "control over AI", "regulations", "governance"],
        "description": "Captures mentions of the need for rules, laws, government oversight, or control over AI development and deployment."
    },
    "ETHICS_BIAS": {
        "keywords": ["fairness", "discrimination", "algorithmic bias", "moral considerations", "biased against certain groups", "ethical"],
        "description": "Captures mentions of fairness, discrimination, algorithmic bias, or moral considerations."
    },
    "HEALTHCARE_BENEFIT": {
        "keywords": ["health", "medicine", "disease diagnosis", "drug discovery", "cures for diseases", "healthcare"],
        "description": "Captures mentions of positive impacts on health, medicine, disease diagnosis, or drug discovery."
    },
    "EXISTENTIAL_RISK": {
        "keywords": ["catastrophic", "humanity-level risks", "loss of human control", "superintelligence", "dangerous for humanity", "existential risk"],
        "description": "Captures mentions of large-scale, catastrophic, or humanity-level risks, including loss of human control or superintelligence."
    },
    "SURVEILLANCE_PRIVACY": {
        "keywords": ["monitoring", "data privacy", "government surveillance", "corporate surveillance", "use my data"],
        "description": "Captures mentions of monitoring, data privacy, and government or corporate surveillance."
    }
}

# Continent Mapping for Location Standardization
continent_mapping = {
    # North America
    'USA': 'North America', 'United States': 'North America', 'United States of America': 'North America', 'US': 'North America',
    'Canada': 'North America', 'Mexico': 'North America', 'North America': 'North America',
    # Europe
    'UK': 'Europe', 'United Kingdom': 'Europe', 'Great Britain': 'Europe', 'England': 'Europe', 'Scotland': 'Europe', 'Wales': 'Europe', 'Northern Ireland': 'Europe',
    'Germany': 'Europe', 'France': 'Europe', 'Italy': 'Europe', 'Spain': 'Europe', 'Poland': 'Europe', 'Ukraine': 'Europe',
    'Netherlands': 'Europe', 'Belgium': 'Europe', 'Sweden': 'Europe', 'Norway': 'Europe', 'Finland': 'Europe', 'Denmark': 'Europe',
    'Ireland': 'Europe', 'Switzerland': 'Europe', 'Austria': 'Europe', 'Portugal': 'Europe', 'Greece': 'Europe',
    'Czech Republic': 'Europe', 'Hungary': 'Europe', 'Romania': 'Europe', 'Bulgaria': 'Europe', 'Serbia': 'Europe', 'Croatia': 'Europe',
    'Europe': 'Europe', 'Western Europe': 'Europe', 'Eastern Europe': 'Europe', 'Northern Europe': 'Europe', 'Southern Europe': 'Europe',
    # Asia
    'China': 'Asia', 'India': 'Asia', 'Japan': 'Asia', 'South Korea': 'Asia', 'Indonesia': 'Asia', 'Pakistan': 'Asia',
    'Bangladesh': 'Asia', 'Philippines': 'Asia', 'Vietnam': 'Asia', 'Turkey': 'Asia', 'Iran': 'Asia', 'Thailand': 'Asia',
    'Saudi Arabia': 'Asia', 'UAE': 'Asia', 'Israel': 'Asia', 'Singapore': 'Asia', 'Malaysia': 'Asia', 'Hong Kong': 'Asia',
    'Asia': 'Asia', 'Southeast Asia': 'Asia', 'South Asia': 'Asia', 'East Asia': 'Asia', 'Middle East': 'Asia',
    # Africa
    'Nigeria': 'Africa', 'Egypt': 'Africa', 'South Africa': 'Africa', 'Kenya': 'Africa', 'Ethiopia': 'Africa', 'Algeria': 'Africa',
    'Ghana': 'Africa', 'Morocco': 'Africa', 'Tanzania': 'Africa', 'Uganda': 'Africa',
    'Africa': 'Africa', 'North Africa': 'Africa', 'Sub-Saharan Africa': 'Africa',
    # South America
    'Brazil': 'South America', 'Argentina': 'South America', 'Colombia': 'South America', 'Peru': 'South America', 'Chile': 'South America',
    'Venezuela': 'South America', 'Ecuador': 'South America', 'Bolivia': 'South America',
    'South America': 'South America', 'Latin America': 'South America',
    # Oceania
    'Australia': 'Oceania', 'New Zealand': 'Oceania', 'Papua New Guinea': 'Oceania', 'Fiji': 'Oceania',
    'Oceania': 'Oceania',
    # Default / Other
    'Global': 'Global/Unknown', 'Worldwide': 'Global/Unknown', 'International': 'Global/Unknown',
    'Native American': 'North America',
    'Caribbean': 'North America',
    'Central America': 'North America'
}

# --- Data Loading and Preprocessing Functions ---
def p2f(x):
    """Converts percentage string to float."""
    try:
        if x == ' - ':
            return float("nan")
        else:
            return float(x.strip('%')) / 100
    except:
        return x

def load_data_from_csv(filename, padding_rows=PADDING_ROWS):
    """Loads and preprocesses data from the Remesh-style CSV."""
    with open(filename, 'r', encoding='utf-8') as file: # Added encoding
        csvreader = csv.reader(file)
        r = 1
        data = []
        qdata = []
        for row in csvreader:
            if r > padding_rows:
                if len(row) == 0 or not any(field.strip() for field in row): # Handle empty or all-whitespace rows
                    if qdata: # only append if qdata is not empty
                        data.append(qdata)
                        qdata = []
                else:
                    qdata.append(row)
            r = r + 1
        if qdata: # Append last qdata if not empty
            data.append(qdata)

    if not data or not data[0]: # Handle cases where data might be empty or first element is empty
        print("Warning: No data loaded after initial CSV processing. Check PADDING_ROWS and file content.")
        return [], pd.DataFrame()

    # Handle potential blank row at the beginning if PADDING_ROWS is inexact
    if not data[0][0][0].strip(): # if the first cell of the first question's header is blank
         data = data[1:]
         if not data:
            print("Warning: Data became empty after removing initial blank row.")
            return [], pd.DataFrame()

    qs = []
    meta_list = [["question type", "question text"]]
    for i, d_block in enumerate(data):
        if not d_block or len(d_block) < 2 or len(d_block[0])==0 : # Skip empty or malformed blocks
            print(f"Warning: Skipping malformed data block at index {i}.")
            continue

        # Ensure header row and first data row exist and have enough columns
        header_row = d_block[0]
        first_data_row = d_block[1] if len(d_block) > 1 else header_row # Use header if only one row

        if len(first_data_row) < 3:
            print(f"Warning: Data block {i} has insufficient columns in its first data row. Skipping.")
            continue

        m = [first_data_row[1], first_data_row[2]] # Question Type, Question Text
        meta_list.append(m)

        df_data = []
        if d_block[1][1] == 'Poll Single Select':
            for r_idx in range(1, len(d_block)):
                row_copy = list(d_block[r_idx]) # Make a copy to modify
                for c_idx in range(4, len(header_row)): # Iterate based on header length
                    if c_idx < len(row_copy):
                         row_copy[c_idx] = p2f(row_copy[c_idx])
                df_data.append(row_copy)
        elif d_block[1][1] == 'Ask Opinion':
            for r_idx in range(1, len(d_block)):
                row_copy = list(d_block[r_idx])
                # Adjust range to avoid index out of bounds for segment columns
                # Max column index for segments is len(header_row)-4
                for c_idx in range(6, min(len(header_row) - 3, len(row_copy))):
                    row_copy[c_idx] = p2f(row_copy[c_idx])
                df_data.append(row_copy)
        else: # For other types or if only header exists
             df_data = d_block[1:]


        if not df_data: # If df_data became empty (e.g. only header was there)
            print(f"Warning: No data rows for DataFrame in block {i} ('{m[1]}'). Creating empty DataFrame with headers.")
            df = pd.DataFrame(columns=header_row)

        else:
            df = pd.DataFrame(df_data, columns=header_row)
        qs.append(df)

    qmeta = pd.DataFrame(meta_list[1:], columns=meta_list[0])
    return qs, qmeta

def load_data_from_json(filepath):
    """Loads data from a JSON file where data was saved as list of dicts."""
    with open(filepath, 'r', encoding='utf-8') as f:
        loaded_list = json.load(f)
    qs = [pd.DataFrame(df_dict) for df_dict in loaded_list]

    # Reconstruct qmeta (assuming standard structure)
    meta_list = [["question type", "question text"]]
    for df in qs:
        if not df.empty and 'Question Type' in df.columns and 'Question' in df.columns and len(df) > 1:
             # Use iloc[0] if header is part of data, or iloc[1] if separate header row in data part
            q_type_row_idx = 0 if df.iloc[0]['Question Type'] in ['Poll Single Select', 'Ask Opinion', 'Ask Experience'] else 1
            if len(df) > q_type_row_idx :
                 meta_list.append([df['Question Type'].iloc[q_type_row_idx], df['Question'].iloc[q_type_row_idx]])
            else: # Fallback if structure is unusual
                meta_list.append(["Unknown", "Unknown Question"])

        elif not df.empty and 'Question Type' in df.columns and 'Question' in df.columns and len(df) ==1:
             meta_list.append([df['Question Type'].iloc[0], df['Question'].iloc[0]])
        else:
            meta_list.append(["Unknown", "Unknown Question"])

    qmeta = pd.DataFrame(meta_list[1:], columns=meta_list[0])
    return qs, qmeta


# --- Thematic Coding Functions ---
def apply_thematic_coding(text, codebook_dict):
    applied_codes = []
    if not isinstance(text, str): # Handle non-string inputs
        return applied_codes
    for code, details in codebook_dict.items():
        for keyword in details["keywords"]:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                if code not in applied_codes:
                    applied_codes.append(code)
                break
    return applied_codes

# --- Sentiment Analysis Functions ---
# Ensure vaderSentiment and textblob are installed: pip install vaderSentiment textblob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

def get_vader_sentiment_score(text):
    if not isinstance(text, str): return 0.0
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    return vs['compound']

def get_textblob_sentiment_score(text):
    if not isinstance(text, str): return 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity

# --- Index Calculation Functions ---
def calculate_aioi(sentiment_scores, positive_threshold=0.05, negative_threshold=-0.05):
    if not sentiment_scores: return None
    scores_array = np.array(sentiment_scores)
    total_responses = len(scores_array)
    if total_responses == 0: return None # Avoid division by zero
    num_positive = np.sum(scores_array > positive_threshold)
    num_negative = np.sum(scores_array < negative_threshold)
    percent_positive = (num_positive / total_responses) * 100
    percent_negative = (num_negative / total_responses) * 100
    aioi = percent_positive - percent_negative
    return aioi

def calculate_eai(country_coded_responses):
    all_codes_flat = [code for sublist in country_coded_responses for code in sublist]
    code_counts = Counter(all_codes_flat)
    freq_econ_threat = code_counts.get("ECON_THREAT", 0)
    freq_econ_opportunity = code_counts.get("ECON_OPPORTUNITY", 0)
    denominator = freq_econ_threat + freq_econ_opportunity
    if denominator == 0: return None
    eai = freq_econ_threat / denominator
    return eai

def calculate_geci(country_coded_responses):
    all_codes_flat = [code for sublist in country_coded_responses for code in sublist]
    if not all_codes_flat: return None
    code_counts = Counter(all_codes_flat)
    freq_gov_reg = code_counts.get("GOVERNANCE_REGULATION", 0)
    freq_ethics_bias = code_counts.get("ETHICS_BIAS", 0)
    total_codes_applied = len(all_codes_flat)
    if total_codes_applied == 0: return None
    geci = (freq_gov_reg + freq_ethics_bias) / total_codes_applied
    return geci

def calculate_dss(country_coded_responses):
    all_codes_flat = [code for sublist in country_coded_responses for code in sublist]
    if not all_codes_flat: return 0.0
    code_counts = Counter(all_codes_flat)
    num_unique_codes = len(code_counts)
    if num_unique_codes <= 1: return 0.0
    probabilities = np.array(list(code_counts.values())) / len(all_codes_flat)
    actual_entropy = entropy(probabilities, base=2)
    max_entropy = np.log2(num_unique_codes)
    if max_entropy == 0: return 0.0
    dss = actual_entropy / max_entropy
    return dss

# --- Location Standardization ---
def standardize_location_to_continent(location_string, mapping):
    if not isinstance(location_string, str):
        return 'Unknown/Other'
    location_string_lower = location_string.lower().strip()
    for key, continent in mapping.items():
        if key.lower() == location_string_lower:
            return continent
    # A more advanced version might check for keywords if direct match fails
    # e.g., if "asia" in location_string_lower and not specific country found
    return 'Unknown/Other'

# --- Analysis Library (Non-LLM) ---
import math # Already imported but good to have near its usage

def show_questions(qmeta_df): # Takes qmeta DataFrame
  return qmeta_df

def show_segments(qs_list): # Takes list of DataFrames
  segments = []
  if not qs_list: return pd.DataFrame(segments)
  q0 = qs_list[0] # Use the first question's DataFrame to determine segment columns

  # Determine actual header row for question type
  q_type_row_idx = 0
  if 'Question Type' in q0.columns and len(q0['Question Type']) > 1:
      if not (q0['Question Type'].iloc[0] == 'Poll Single Select' or q0['Question Type'].iloc[0] == 'Ask Opinion'):
          q_type_row_idx = 1 # Header is likely the second row in the data part
  elif not ('Question Type' in q0.columns and len(q0['Question Type']) > 0):
      print("Warning: 'Question Type' column not found or empty in the first DataFrame for show_segments.")
      return pd.DataFrame(segments)


  if len(q0) <= q_type_row_idx:
      print(f"Warning: DataFrame for show_segments has fewer than {q_type_row_idx+1} rows. Cannot determine question type.")
      return pd.DataFrame(segments)

  question_type_val = q0["Question Type"].iloc[q_type_row_idx]

  if question_type_val == 'Poll Single Select':
    # Segments start from column index 4
    for c_idx in range(4, len(q0.columns)):
      segments.append(q0.columns[c_idx])
  elif question_type_val == 'Ask Opinion':
    # Segments start from column index 7 (as per notebook, but seems like 5 in csv loading)
    # The notebook's `table_ask` and `bridging_ask` use df.columns[7+segs[i]]
    # but the CSV loading logic for 'Ask Opinion' processes segments from column 6.
    # Let's assume segments for Ask Opinion start after 'Participant Session ID', 'Response ID', 'English Responses', 'Agree', 'Disagree', 'Skip'
    # which typically means starting from column index 6 if 0-indexed.
    # The notebook has `df.columns[7+segs[i]]` for `bridging_ask`, this might be an offset if `segs` are indices.
    # For `show_segments`, it used `df.columns[5:len(q0.columns)-3]`. This implies segments are from index 5 up to 3 from end.
    # Let's use the logic from the notebook's show_segments for consistency here.
    # The original notebook code was:
    # if q0["Question Type"][1] == 'Poll Single Select':
    #   for c in range(4,len(q0.columns)): segments.append(q0.columns[c])
    # if q0["Question Type"][1] == 'Ask Opinion':
    #   for c in range(5,len(q0.columns)-3): segments.append(q0.columns[c])
    # This implies that for Ask Opinion, segments are from index 5 up to the last 3 columns.
    # Let's adjust to use the column names directly if they are segment columns.
    # This is tricky without knowing the exact fixed columns before segments.
    # Assuming the notebook's original `show_segments` logic was correct for its data structure:
    # For Ask Opinion, segments are often after 'English Responses', 'Agree', 'Disagree', 'Skip', and potentially 'Embedding' if added later.
    # The provided notebook's `show_segments` uses `range(5,len(q0.columns)-3)` for Ask Opinion.
    # Let's stick to that for now, assuming fixed leading columns and 3 trailing columns (like embedding, etc.)
    start_col_idx = 5 # As per original notebook for Ask Opinion in show_segments
    end_col_idx = len(q0.columns) - 3
    if 'embedding' in q0.columns: # If embedding column is present, it's usually last or near last.
        # This part is heuristic. If 'embedding' is one of the last 3, the original logic is fine.
        # If it's *before* other segments, this needs adjustment.
        # For simplicity, let's assume the original range was robust for its data.
        pass

    for c_idx in range(start_col_idx, end_col_idx):
        segments.append(q0.columns[c_idx])
  else:
    print(f"Segment display not configured for question type: {question_type_val}")

  return pd.DataFrame(segments, columns=["Segment Name"])


def plot_poll(df, seg_indices_or_names, q_text): # Added q_text
  print(q_text)
  plt.close("all") # Close previous plots

  segs_incl_cols = ['Responses'] # This should be the options column

  # Determine if segs are indices or names
  if all(isinstance(s, int) for s in seg_indices_or_names): # Segment indices
      for seg_idx in seg_indices_or_names:
          if 4 + seg_idx < len(df.columns):
              segs_incl_cols.append(df.columns[4 + seg_idx])
          else:
              print(f"Warning: Segment index {seg_idx} is out of bounds for DataFrame columns.")
  elif all(isinstance(s, str) for s in seg_indices_or_names): # Segment names
      for seg_name in seg_indices_or_names:
          if seg_name in df.columns:
              segs_incl_cols.append(seg_name)
          else:
              print(f"Warning: Segment name '{seg_name}' not found in DataFrame columns.")
  else:
      print("Warning: 'segs' parameter in plot_poll must be a list of integers (indices) or strings (names).")
      return pd.DataFrame()

  if len(segs_incl_cols) == 1: # Only 'Responses' column, nothing to plot
      print("Warning: No valid segments selected for plotting.")
      return pd.DataFrame()

  # Check if 'Responses' (options) column exists and is correctly identified
  # In Remesh CSVs, for Polls, the options are often in a column named 'Response' or 'Option Text'
  # The first column of dfplt should be the options.
  # Let's find the most likely options column.
  # Heuristic: it's usually the first column that is not a segment value column.
  # For Polls, this is typically df.columns[1] (Option Text) or df.columns[3] (Response) if structure is consistent

  # The notebook used df[segs_incl] where segs_incl started with 'Responses'.
  # This implies 'Responses' is the name of the column containing poll options.
  # Let's verify this column exists.
  options_col_name = None
  if 'Responses' in df.columns: # This was used in the notebook
      options_col_name = 'Responses'
  elif 'Option Text' in df.columns: # A common alternative
      options_col_name = 'Option Text'
  elif 'Response' in df.columns and df['Response'].nunique() < len(df): # Check if 'Response' is categorical options
      options_col_name = 'Response'

  if not options_col_name:
      print("Error: Could not identify the poll options column (e.g., 'Responses', 'Option Text').")
      return pd.DataFrame()

  # Reconstruct segs_incl_cols with the correct options column name
  plot_cols = [options_col_name] + [col for col in segs_incl_cols if col != 'Responses' and col in df.columns]


  dfplt = df[plot_cols].copy() # Use copy to avoid SettingWithCopyWarning

  # Convert segment columns to numeric, coercing errors
  for col in plot_cols[1:]: # Skip the options column
      dfplt[col] = pd.to_numeric(dfplt[col], errors='coerce')

  dfplt = dfplt.set_index(options_col_name)
  dfplt.plot.barh()
  plt.title(q_text[:80] + "...") # Add title to plot
  plt.tight_layout()
  # In a cloud script, you'd save this plot to a file instead of showing
  # plt.savefig("poll_plot.png")
  print("Plot generated. In a script, this would be saved to a file.")
  return dfplt

def make_pretty(styler):
  styler.background_gradient(axis=None, vmin=0, vmax=1, cmap="RdYlGn")
  styler.format(precision=2)
  return styler

def table_ask(df, seg_indices_or_names, n, q_text=""): # Added q_text for context
  if q_text: print(q_text)

  segs_incl_cols = ['English Responses']
  # This function in the notebook assumes segs are indices relative to a starting point (column 7)
  # df.columns[7+segs[i]]. This is quite specific.
  # Let's make it more robust by allowing names or direct indices if possible.

  if all(isinstance(s, int) for s in seg_indices_or_names): # Segment indices
      # Original notebook logic for 'segs' being indices from a base:
      # Here, 'segs' elements are direct indices if mapping to the notebook's use of `df.columns[7+segs[i]]`
      # If segs = [0, 231, ...], then 0 maps to overall, 231 to a specific segment column.
      # This requires knowing the actual column names for these "segment IDs".
      # This part is hard to generalize without the segment ID to column name mapping.
      # For now, let's assume seg_indices_or_names are actual column names if strings, or direct column indices if int.
      # If they are the "segment IDs" like 0, 231, etc., they need to be mapped to actual column names first.
      # The notebook seems to use these IDs as offsets from a base (e.g., column 7).
      # This is too fragile. Let's assume seg_indices_or_names are column names or direct indices.

      # If seg_indices_or_names are direct column indices:
      for seg_idx in seg_indices_or_names:
          if seg_idx < len(df.columns):
              segs_incl_cols.append(df.columns[seg_idx])
          else:
              print(f"Warning: Segment index {seg_idx} out of bounds.")
  elif all(isinstance(s, str) for s in seg_indices_or_names): # Segment names
      for seg_name in seg_indices_or_names:
          if seg_name in df.columns:
              segs_incl_cols.append(seg_name)
          else:
               print(f"Warning: Segment name '{seg_name}' not found.")
  else:
      print("Warning: 'segs' in table_ask must be list of column names or direct indices.")
      # Fallback to notebook's specific logic if segs = [0, 231, ...] was intended with base 7
      # This part is very specific to the notebook's implicit structure if segs are not names/direct_indices
      # print("Attempting fallback for segment IDs as offsets (notebook specific)...")
      # temp_segs_incl = ['English Responses']
      # for seg_id_offset in seg_indices_or_names: # e.g. seg_id_offset = 0 or 231
      #     actual_col_idx = 7 + seg_id_offset # This is from the notebook
      #     if actual_col_idx < len(df.columns):
      #         temp_segs_incl.append(df.columns[actual_col_idx])
      #     else:
      #         print(f"Warning: Fallback segment offset {seg_id_offset} (idx {actual_col_idx}) out of bounds.")
      # segs_incl_cols = temp_segs_incl
      # if len(segs_incl_cols) == 1:
      return pd.DataFrame().style # Return empty styled DataFrame

  if 'English Responses' not in df.columns:
      print("Error: 'English Responses' column not found in DataFrame for table_ask.")
      return pd.DataFrame().style

  dfplt = df[segs_incl_cols]
  return dfplt.iloc[:n].style.pipe(make_pretty)

# --- Bridging, Polarization, Divergence Metrics ---
# These functions depend on `table_ask`'s column selection logic.
# The original `bridging_ask` used `df.columns[7+segs[i]]`.
# This needs to be consistent or made more robust.

def _get_segment_columns_for_metrics(df, seg_indices_or_names_or_offsets):
    """Helper to get actual segment column names for metric calculations."""
    segment_value_cols = []
    # Try to be robust: are these names, direct indices, or notebook-style offsets?
    if all(isinstance(s, str) for s in seg_indices_or_names_or_offsets): # Names
        for name in seg_indices_or_names_or_offsets:
            if name in df.columns and name != 'English Responses':
                segment_value_cols.append(name)
    elif all(isinstance(s, int) for s in seg_indices_or_names_or_offsets):
        # This is ambiguous: direct indices or offsets?
        # Let's assume direct indices first if they are valid column indices for numeric data.
        # Heuristic: if values are small (like 0, 1, 2), they might be offsets.
        # If values are large (like 7, 238), they might be direct indices or notebook offsets.
        # The notebook used `df.columns[7+segs[i]]` where `segs` was like `[0,231,232,...]`
        # So `segs[0]=0` meant `df.columns[7]`, `segs[1]=231` meant `df.columns[7+231]`.
        # This is very fragile.
        # A better approach: `seg_indices_or_names_or_offsets` should be a list of actual segment column NAMES.
        print("Warning: Using integer list for segments in metric functions. Please provide column names for robustness.")
        print("Assuming these are direct column indices for now.")
        for idx in seg_indices_or_names_or_offsets:
            if idx < len(df.columns) and df.columns[idx] != 'English Responses':
                 # Check if this column looks like a numeric segment column
                if pd.api.types.is_numeric_dtype(df[df.columns[idx]].dropna()):
                    segment_value_cols.append(df.columns[idx])
                # else:
                #     print(f"  Skipping column {df.columns[idx]} as it's not numeric for metrics.")
            # Fallback to notebook's offset logic if direct indices don't make sense or are few.
            # This part is difficult to make robust without more context on `segs` content.
            # For now, we rely on the user passing column names. If they pass ints, we try direct indices.
    else:
        print("Error: Segments for metrics must be a list of column names or direct indices.")

    if not segment_value_cols:
         print("No valid segment columns found for metrics calculation. Check 'segs' input.")
    return segment_value_cols


def min_bridge(row, metric_segment_cols):
  b = 1.0
  if not metric_segment_cols: return 0.0 # No segments to compare
  for s_col in metric_segment_cols:
    b_ = row[s_col]
    if pd.isna(b_): continue # Skip NaN values
    b = min(b, b_)
  return b

def polarization_metric(row, metric_segment_cols): # Renamed from 'polarization' to avoid conflict
  mx = 0.0
  mn = 1.0
  if not metric_segment_cols: return 0.0
  valid_scores = [row[s_col] for s_col in metric_segment_cols if pd.notna(row[s_col])]
  if not valid_scores: return 0.0
  mx = max(valid_scores)
  mn = min(valid_scores)
  return mx - mn

def symmetric_divergence(row, metric_segment_cols):
  if not metric_segment_cols: return 0.0
  valid_scores = [row[s_col] for s_col in metric_segment_cols if pd.notna(row[s_col])]
  if not valid_scores: return 0.0
  mx = max(valid_scores)
  mn = min(valid_scores)
  mx_div = max(mx - 0.5, 0)
  mn_div = max(0.5 - mn, 0)
  return math.sqrt(mx_div * mn_div)

def bridging_ask(df, segs_config_list): # segs_config_list should be list of actual segment column names
  df_copy = df.copy()
  if 'English Responses' not in df_copy.columns:
      print("Error: 'English Responses' column missing.")
      return pd.DataFrame()

  metric_segment_cols = _get_segment_columns_for_metrics(df_copy, segs_config_list)
  if not metric_segment_cols:
      print("Error: No valid segment columns identified for bridging_ask metrics.")
      # Add empty metric columns to avoid errors later if expected
      df_copy["bridge"] = 0.0
      df_copy["polarization"] = 0.0
      df_copy["divergence"] = 0.0
      return df_copy.sort_values(by=["bridge"], ascending=False)


  df_copy["bridge"] = df_copy.apply(lambda row: min_bridge(row, metric_segment_cols), axis=1)
  df_copy["polarization"] = df_copy.apply(lambda row: polarization_metric(row, metric_segment_cols), axis=1)
  df_copy["divergence"] = df_copy.apply(lambda row: symmetric_divergence(row, metric_segment_cols), axis=1)

  # Columns to display: English Responses, the metric_segment_cols, and the new metric columns
  display_cols = ['English Responses'] + metric_segment_cols + ["bridge", "polarization", "divergence"]
  # Filter out any columns not actually in df_copy (e.g. if 'English Responses' was missing)
  display_cols = [col for col in display_cols if col in df_copy.columns]

  return df_copy[display_cols].sort_values(by=["bridge"], ascending=False)


def get_bridging_responses(df, segs_config_list, thresh):
  bdf = bridging_ask(df, segs_config_list)
  return bdf.loc[bdf['bridge'] > thresh]

def get_polarizing_responses(df, segs_config_list, n):
  bdf = bridging_ask(df, segs_config_list)
  bdfp = bdf.sort_values(by=["polarization"], ascending=False)
  return bdfp.iloc[:n]

def get_divergent_responses(df, segs_config_list, n):
  bdf = bridging_ask(df, segs_config_list)
  bdfp = bdf.sort_values(by=["divergence"], ascending=False)
  return bdfp.iloc[:n]

def _summary_text_common(df_subset, metric_segment_cols, response_col='English Responses'):
    summary_lines = []
    for idx in df_subset.index:
        first_col_text = df_subset.loc[idx, response_col]

        valid_scores_dict = {col: df_subset.loc[idx, col] for col in metric_segment_cols if pd.notna(df_subset.loc[idx, col])}
        if not valid_scores_dict:
            min_col_name, max_col_name = "N/A", "N/A"
            min_value, max_value = float('nan'), float('nan')
        else:
            min_col_name = min(valid_scores_dict, key=valid_scores_dict.get)
            max_col_name = max(valid_scores_dict, key=valid_scores_dict.get)
            min_value = valid_scores_dict[min_col_name]
            max_value = valid_scores_dict[max_col_name]

        summary_lines.append(first_col_text)
        summary_lines.append(f"Low : {min_value*100:.0f}% -- {min_col_name}")
        summary_lines.append(f"High : {max_value*100:.0f}% -- {max_col_name}")
        summary_lines.append(" ")
    return "\n".join(summary_lines)

def polarization_summary(df, segs_config_list, n):
  metric_segment_cols = _get_segment_columns_for_metrics(df, segs_config_list)
  if not metric_segment_cols: return "Error: No segment columns for polarization summary."
  pa = get_polarizing_responses(df, segs_config_list, n)
  return _summary_text_common(pa, metric_segment_cols)

def divergence_summary(df, segs_config_list, n):
  metric_segment_cols = _get_segment_columns_for_metrics(df, segs_config_list)
  if not metric_segment_cols: return "Error: No segment columns for divergence summary."
  da = get_divergent_responses(df, segs_config_list, n)
  return _summary_text_common(da, metric_segment_cols)

def save_qs_to_json(qs_list, filename):
  qx = [df.to_dict(orient='records') for df in qs_list] # More standard way to save df to dict list
  with open(filename, 'w', encoding='utf-8') as f:
      json.dump(qx, f, indent=2) # Added indent for readability
  print(f"Data saved to {filename}")

# --- Main Processing Logic ---
def main_non_llm_analysis(input_filepath, output_dir="."):
    """
    Main function to run the non-LLM analysis pipeline.
    """
    print("Starting Non-LLM Analysis Pipeline...")

    # Step 1: Load Data
    # Determine if input is CSV or JSON based on extension
    if input_filepath.lower().endswith(".csv"):
        print(f"Loading data from CSV: {input_filepath}")
        qs, qmeta = load_data_from_csv(input_filepath)
    elif input_filepath.lower().endswith(".json"):
        print(f"Loading data from JSON: {input_filepath}")
        qs, qmeta = load_data_from_json(input_filepath)
    else:
        print(f"Error: Unsupported file type: {input_filepath}. Please use .csv or .json.")
        return

    if not qs:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded {len(qs)} question DataFrames.")
    # print("Question Metadata:")
    # print(show_questions(qmeta).head())

    # Step 2 & 3: Apply Thematic Coding and Sentiment Analysis
    print("\nApplying Thematic Coding and Sentiment Analysis...")
    ask_opinion_question_indices = []
    participant_id_col_name_global = None # To store the identified PID column name

    for i, q_df_orig in enumerate(qs):
        q_df = q_df_orig.copy() # Work on a copy

        # Determine question type from the DataFrame itself (more robust)
        q_type = None
        q_text = f"Question {i}" # Default question text

        # Try to get Question Type and Question Text from the DataFrame
        # Assuming the first row (index 0) of the DataFrame contains this metadata
        # This needs to be robust to how DataFrames were constructed by load_data_*
        current_q_meta_row = qmeta[qmeta['question text'] == q_df['Question'].iloc[1 if len(q_df)>1 and q_df['Question'].iloc[0] == q_df.columns[2] else 0] ] if 'Question' in q_df.columns and not q_df.empty else pd.DataFrame()


        if not current_q_meta_row.empty:
            q_type = current_q_meta_row["question type"].iloc[0]
            q_text = current_q_meta_row["question text"].iloc[0]
        elif 'Question Type' in q_df.columns and not q_df.empty: # Fallback
             # Check if header is the first row of data or a separate header
            header_is_data_row0 = (len(q_df) > 1 and q_df['Question Type'].iloc[0] == q_df['Question Type'].iloc[1]) or (len(q_df)==1)
            q_type_row_idx = 0 if header_is_data_row0 else 1
            if len(q_df) > q_type_row_idx:
                q_type = q_df['Question Type'].iloc[q_type_row_idx]
                q_text = q_df['Question'].iloc[q_type_row_idx] if 'Question' in q_df.columns else q_text


        if q_type == 'Ask Opinion':
            ask_opinion_question_indices.append(i)
            print(f"  Processing Question ID (index): {i} - Type: Ask Opinion - Text: {q_text[:50]}...")
            if 'English Responses' in q_df.columns:
                q_df['English Responses'] = q_df['English Responses'].astype(str).fillna('')
                q_df['thematic_codes'] = q_df['English Responses'].apply(
                    lambda x: apply_thematic_coding(x, codebook) if pd.notna(x) and x.strip() != '' else []
                )
                q_df['vader_sentiment_score'] = q_df['English Responses'].apply(
                    lambda x: get_vader_sentiment_score(x) if pd.notna(x) and x.strip() != '' else 0.0
                )
                qs[i] = q_df # Update the DataFrame in the list
                # print(q_df[['English Responses', 'thematic_codes', 'vader_sentiment_score']].head())
            else:
                print(f"    Warning: 'English Responses' column not found in DataFrame for question index {i}.")

        # Identify participant ID column name (globally, assuming it's consistent)
        # This is done once, typically from the country question or first available.
        if not participant_id_col_name_global:
            possible_pid_cols = ['Participant Session ID', 'Participant ID', 'Sub ID', 'Participant Response ID']
            for col in possible_pid_cols:
                if col in q_df.columns:
                    participant_id_col_name_global = col
                    print(f"  Identified Participant ID column: '{participant_id_col_name_global}'")
                    break

    if not participant_id_col_name_global:
        print("  Warning: Could not identify a Participant ID column. Continent mapping might fail.")


    # Step 4: Initial Location Data Extraction
    print("\nExtracting Raw Location Data...")
    raw_location_data = []
    country_question_idx = 6 # As identified in notebook: "What country or region do you most identify with?"

    # Find the actual index for the country question based on qmeta
    country_q_text_short = "What country or region"
    try:
        country_question_idx = qmeta[qmeta['question text'].str.contains(country_q_text_short, case=False, na=False)].index[0]
        print(f"  Located country question at index: {country_question_idx} - '{qmeta.loc[country_question_idx, 'question text']}'")
    except IndexError:
        print(f"  Warning: Country question containing '{country_q_text_short}' not found in qmeta. Using default index {country_question_idx}.")


    if country_question_idx < len(qs):
        country_df_orig = qs[country_question_idx]
        country_df = country_df_orig.copy()

        current_q_meta_row_country = qmeta.iloc[[country_question_idx]]
        country_q_type = current_q_meta_row_country["question type"].iloc[0] if not current_q_meta_row_country.empty else None

        if country_q_type == 'Poll Single Select':
            reported_location_col_name = None
            # For Polls, the response is often in 'Response' or 'Option Text' if the rows are participants.
            # Or, it could be the column name itself if rows are options and columns are segments.
            # The notebook's logic for qs[6] (country question) implies rows are participants.
            possible_loc_cols = ['Response', 'Option Text', 'Responses'] # 'Responses' was also used

            for col in possible_loc_cols:
                if col in country_df.columns:
                    # Heuristic: if 'Responses' col has many unique values, it might be options.
                    # A participant's choice is often in 'Response'.
                    if col == 'Responses' and country_df[col].nunique() > 50 and len(country_df) > country_df[col].nunique():
                        pass # Likely options list
                    else:
                        reported_location_col_name = col
                        break

            if participant_id_col_name_global and reported_location_col_name:
                print(f"  Using '{participant_id_col_name_global}' for PID and '{reported_location_col_name}' for Location from qs[{country_question_idx}].")
                for _, row in country_df.iterrows():
                    pid = row[participant_id_col_name_global]
                    loc = row[reported_location_col_name]
                    if pd.notna(pid) and pd.notna(loc):
                        raw_location_data.append({'pid': pid, 'reported_location': str(loc).strip()})
                print(f"  Extracted {len(raw_location_data)} raw location entries.")
                if raw_location_data: print(f"  Sample: {raw_location_data[:3]}")
            else:
                print(f"  Could not find PID ('{participant_id_col_name_global}') or Location ('{reported_location_col_name}') column in country question DataFrame.")
        else:
            print(f"  Country question (idx {country_question_idx}) is not 'Poll Single Select'. Type: {country_q_type}. Skipping raw location extraction.")
    else:
        print(f"  Country question index ({country_question_idx}) is out of bounds. Skipping raw location extraction.")

    # Step 5 & 6: Standardize Location to Continent
    print("\nStandardizing Locations to Continents...")
    participant_to_continent_map = {}
    if raw_location_data:
        for item in raw_location_data:
            pid = item['pid']
            reported_loc = item['reported_location']
            standardized_continent = standardize_location_to_continent(reported_loc, continent_mapping)
            participant_to_continent_map[pid] = standardized_continent
        print(f"  Created participant_to_continent_map with {len(participant_to_continent_map)} entries.")
        if participant_to_continent_map:
            continent_counts = Counter(participant_to_continent_map.values())
            print(f"  Continent distribution: {continent_counts}")
    else:
        print("  No raw location data to process for continent standardization.")

    # Step 7: Consolidate Responses with Continent Info
    print("\nConsolidating Responses with Continent Info...")
    all_responses_with_continent_info = []
    if participant_to_continent_map and participant_id_col_name_global:
        for q_idx in ask_opinion_question_indices:
            df = qs[q_idx] # This is already a copy from earlier
            if participant_id_col_name_global in df.columns and \
               'English Responses' in df.columns and \
               'thematic_codes' in df.columns and \
               'vader_sentiment_score' in df.columns:
                for _, row in df.iterrows():
                    pid = row[participant_id_col_name_global]
                    continent = participant_to_continent_map.get(pid, "Unknown/Other")
                    all_responses_with_continent_info.append({
                        'continent': continent,
                        'response_text': row['English Responses'],
                        'thematic_codes': row['thematic_codes'],
                        'sentiment_score': row['vader_sentiment_score']
                    })
            else:
                 print(f"  Skipping question {q_idx} for continent aggregation due to missing required columns.")
        print(f"  Consolidated {len(all_responses_with_continent_info)} responses with continent information.")
    else:
        print("  No participant_to_continent_map or PID column available. Cannot consolidate by continent.")

    continent_aggregated_df = pd.DataFrame()
    if all_responses_with_continent_info:
        continent_aggregated_df = pd.DataFrame(all_responses_with_continent_info)
        # print("Sample of consolidated data with continent:")
        # print(continent_aggregated_df.head())

    # Step 8: Calculate Continent-Level Indices
    print("\nCalculating Continent-Level Indices...")
    continent_indices_data = []
    if not continent_aggregated_df.empty:
        unique_continents = continent_aggregated_df['continent'].unique()
        for continent_name in unique_continents:
            current_continent_data = continent_aggregated_df[continent_aggregated_df['continent'] == continent_name]
            if current_continent_data.empty: continue

            sentiment_scores_list = current_continent_data['sentiment_score'].tolist()
            coded_responses_list = current_continent_data['thematic_codes'].tolist()

            aioi_score = calculate_aioi(sentiment_scores_list)
            eai_score = calculate_eai(coded_responses_list)
            geci_score = calculate_geci(coded_responses_list)
            dss_score = calculate_dss(coded_responses_list)

            continent_indices_data.append({
                'Continent': continent_name,
                'AIOI': aioi_score, 'EAI': eai_score, 'GECI': geci_score, 'DSS': dss_score,
                'Number_of_Responses': len(current_continent_data)
            })

    continent_indices_df = pd.DataFrame()
    if continent_indices_data:
        continent_indices_df = pd.DataFrame(continent_indices_data)
        print("--- Calculated Continent-Level Indices ---")
        print(continent_indices_df.sort_values(by='Number_of_Responses', ascending=False))
        continent_indices_df.to_csv(f"{output_dir}/continent_indices.csv", index=False)
        print(f"Continent indices saved to {output_dir}/continent_indices.csv")
    else:
        print("No continent-level indices were calculated.")

    # Save processed qs (list of DataFrames) for potential use by LLM script
    if qs:
        save_qs_to_json(qs, f"{output_dir}/processed_qs_data.json")

    print("\nNon-LLM Analysis Pipeline Finished.")
    return qs, qmeta, continent_indices_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Non-LLM Analysis Pipeline.")
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_CSV_PATH,
                        help="Path to the input CSV or JSON file (e.g., data/my_survey.csv or data/processed_data.json).")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save output files (e.g., results/).")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # For testing locally if the default CSV/JSON needs a dummy creation
    if args.input_file == DEFAULT_INPUT_CSV_PATH:
        try:
            with open(DEFAULT_INPUT_CSV_PATH, 'r', encoding='utf-8') as f: # Added encoding
                pass
            print(f"{DEFAULT_INPUT_CSV_PATH} found.")
        except FileNotFoundError:
            print(f"{DEFAULT_INPUT_CSV_PATH} not found. Creating a dummy CSV file for testing structure.")
            dummy_header = ["ID","Question Type","Question","Response","Segment1","Segment2", "Participant Session ID", "English Responses"]
            dummy_q1_meta = ["1","Poll Single Select","Country?","USA","0.5","0.6","pid1","N/A"]
            dummy_q1_data1 = ["","","","USA","0.5","0.6","pid1","N/A"]
            dummy_q1_data2 = ["","","","Canada","0.4","0.3","pid2","N/A"]
            dummy_q2_meta = ["2","Ask Opinion","Your thoughts?","Okay","0.7","0.8","pid1","This is a thought"]
            dummy_q2_data1 = ["","","","Okay","0.7","0.8","pid1","This is a thought"]

            with open(DEFAULT_INPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as f: # Added encoding
                writer = csv.writer(f)
                for _ in range(PADDING_ROWS):
                    writer.writerow([])
                writer.writerow(dummy_header)
                writer.writerow(dummy_q1_meta)
                writer.writerow(dummy_q1_data1)
                writer.writerow(dummy_q1_data2)
                writer.writerow([])
                writer.writerow(dummy_header)
                writer.writerow(dummy_q2_meta)
                writer.writerow(dummy_q2_data1)
            print(f"Dummy {DEFAULT_INPUT_CSV_PATH} created. Please replace with actual data.")

    main_non_llm_analysis(args.input_file, args.output_dir)
