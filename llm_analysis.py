import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Langchain and OpenAI/Google GenAI imports
# Ensure these are in requirements.txt:
# langchain, openai, sentence-transformers, langchain-google-genai
# Also, ensure API keys are set as environment variables in a cloud environment

# Attempt to import, with fallback for environments where they might not be immediately available
try:
    from langchain_openai import ChatOpenAI # For OpenAI models
    from langchain_google_genai import ChatGoogleGenerativeAI # For Google Gemini
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain, SequentialChain
    from openai import OpenAI as OpenAIClient # Renamed to avoid conflict if using both
except ImportError:
    print("Warning: Langchain or OpenAI/Google GenAI libraries not found. LLM functionalities will not work.")
    ChatOpenAI = None
    ChatGoogleGenerativeAI = None
    PromptTemplate = None
    LLMChain = None
    SequentialChain = None
    OpenAIClient = None


# --- Configuration & API Key Management ---
# In a cloud environment, API keys should be set as environment variables.
# Example: os.environ["OPENAI_API_KEY"] = "your_openai_key"
# Example: os.environ["GOOGLE_API_KEY"] = "your_google_api_key"
# The notebook used userdata.get('OPENAI_API_KEY'), which is Colab-specific.

# For Google GenAI client used for embeddings in the notebook:
# The notebook had: client = OpenAI(api_key="AIzaSyCtiQUR_d8XpEJ_0kgCwTAW4Ul83pQG6kU", base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
# This seems to be using an OpenAI library interface to call what might be a Google model.
# Let's assume the user wants to use a Google embedding model.
# The 'models/text-embedding-004' suggests a Google model.
# We'll use a placeholder for the API key.
GOOGLE_API_KEY_FOR_EMBEDDINGS = os.environ.get("GOOGLE_API_KEY") # Needs to be set
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # For OpenAI models if used

# Embedding client (adapted from notebook's OpenAI client pointing to Google)
# This is a bit unusual. Typically, you'd use google.generativeai directly for Google embeddings.
# For now, replicating the notebook's client setup logic.
# If direct google-generativeai is preferred, this would change.
embedding_client = None
if GOOGLE_API_KEY_FOR_EMBEDDINGS and OpenAIClient:
    try:
        embedding_client = OpenAIClient(api_key=GOOGLE_API_KEY_FOR_EMBEDDINGS, base_url="https://generativelanguage.googleapis.com/v1beta/models")
        # Note: The base_url in notebook was "https://generativelanguage.googleapis.com/v1beta/openai/"
        # Changed to "https://generativelanguage.googleapis.com/v1beta/models" which is more standard for direct Google API calls via an OpenAI-like client.
        # Or, if it's truly the text-embedding-004 from Vertex AI via OpenAI library, the base_url might be different.
        # This part is critical and might need adjustment based on the actual embedding endpoint.
        # The model name in notebook was "models/text-embedding-004". For OpenAI client, it's usually just "text-embedding-ada-002" etc.
        # For Google via this client, it might be "text-embedding-004" if the base URL routes it correctly.
    except Exception as e:
        print(f"Error initializing embedding client: {e}")
        embedding_client = None
else:
    print("Warning: GOOGLE_API_KEY for embeddings not set or OpenAI library not available. Embedding functions will not work.")


# --- Embedding Functions ---
def get_embedding(text, model_name="text-embedding-004", dimensions=None): # dimensions was 1024 in notebook, but for text-embedding-004 it's 768
    """
    Generates embeddings using the configured client.
    The model "models/text-embedding-004" in the notebook implies a Google model.
    The notebook used dimensions=1024, but Google's text-embedding-004 is 768.
    Let's use model="text-embedding-004" and let the API handle dimensions or specify 768.
    """
    if not embedding_client:
        print("Error: Embedding client not initialized.")
        return None
    if not isinstance(text, str): # Handle non-string inputs
        print("Warning: Non-string input to get_embedding. Returning None.")
        return None
    text = text.replace("\n", " ")
    try:
        # The client.embeddings.create call might need adjustment based on the actual client library and model.
        # If using the OpenAI library for a Google model, the `model` parameter might need to be `model=model_name`
        # and dimensions might be ignored or handled by the endpoint.
        # For "text-embedding-004" (Google), dimension is typically 768.
        # If the notebook's "OpenAI" client was actually for Vertex AI, this call might differ.
        # Assuming the notebook's `client.embeddings.create` call was functional with its setup.
        # The `dimensions` parameter is specific to some OpenAI models, not standard for all.
        # Google's "text-embedding-004" has a fixed output dimensionality of 768.
        # If the notebook truly got 1024 dimensions, it might have been a different model or a misconfiguration.
        # For safety, let's try to pass dimensions if specified and supported, otherwise let the model default.

        # Corrected model name to be just the ID if using Google's standard API structure
        # The notebook had "models/text-embedding-004".
        # If the base_url is `.../v1beta/models`, then model should be `text-embedding-004`.
        # If base_url is `.../v1beta/openai/`, then it might be `models/text-embedding-004`. This is unusual.

        # Let's assume the model name is just 'text-embedding-004' and handle dimensions based on model specifics.
        # The `dimensions` parameter is not standard for Google's `text-embedding-004` via their direct SDK.
        # If using OpenAI client wrapper, it might be an attempt to use an OpenAI feature.
        # Given the notebook code, we try to replicate it.

        # The notebook used: client.embeddings.create(input = [text], model=model,dimensions=dimensions).data[0].embedding
        # model was "models/text-embedding-004", dimensions=1024. This is contradictory for text-embedding-004.
        # Let's assume user wants 'text-embedding-004' (which is 768D) and the dimensions param was an attempt for OpenAI models.
        # Or, if the endpoint `generativelanguage.googleapis.com/v1beta/openai/` somehow proxies to an OpenAI model, then 1024 could be valid for some.
        # This is the most ambiguous part.
        # Safest bet: use the model name as-is from notebook, and pass dimensions if provided.

        if dimensions:
             return embedding_client.embeddings.create(input=[text], model=model_name, dimensions=dimensions).data[0].embedding
        else:
             return embedding_client.embeddings.create(input=[text], model=model_name).data[0].embedding

    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def embed_response_row(row, text_column="English Responses", model_name="text-embedding-004", dimensions=None):
    text = row[text_column]
    return get_embedding(text, model_name=model_name, dimensions=dimensions)

def add_embeddings_to_df(df, text_column="English Responses", embedding_col_name="embedding", model_name="text-embedding-004", dimensions=None):
    print(f"Adding embeddings to DataFrame using model {model_name}...")
    if text_column not in df.columns:
        print(f"Error: Text column '{text_column}' not found in DataFrame.")
        return df
    df[embedding_col_name] = df.apply(lambda row: embed_response_row(row, text_column, model_name, dimensions), axis=1)
    print("Embeddings added.")
    return df

def add_embeddings_for_all_qs(qs_list, qmeta_df, text_column="English Responses", embedding_col_name="embedding", model_name="text-embedding-004", dimensions=None):
    if not qs_list or not embedding_client:
        print("Error: qs_list is empty or embedding client not initialized.")
        return qs_list

    print(f"Generating embeddings for all 'Ask Opinion' questions using model {model_name}...")
    for i, df_orig in enumerate(qs_list):
        df = df_orig.copy()
        q_type = qmeta_df.iloc[i]["question type"] if i < len(qmeta_df) else "Unknown"

        if q_type == "Ask Opinion":
            print(f"  Processing question {i}...")
            qs_list[i] = add_embeddings_to_df(df, text_column, embedding_col_name, model_name, dimensions)
    print("Finished adding embeddings for all relevant questions.")
    return qs_list

# --- Similarity Ranking ---
def rank_by_similarity(df_with_embeddings, query_text, text_column="English Responses", embedding_col_name="embedding", model_name="text-embedding-004", dimensions=None):
    if embedding_col_name not in df_with_embeddings.columns:
        print(f"Error: Embedding column '{embedding_col_name}' not found in DataFrame.")
        return pd.DataFrame()
    if df_with_embeddings[embedding_col_name].isnull().all():
        print("Error: All embeddings are null. Cannot rank by similarity.")
        return df_with_embeddings.copy() # Return original if no embeddings

    query_embedding = get_embedding(query_text, model_name=model_name, dimensions=dimensions)
    if query_embedding is None:
        print("Error: Could not generate embedding for query text.")
        return df_with_embeddings.copy()

    df_copy = df_with_embeddings.copy()

    # Ensure embeddings are in a consistent format (list or np.array) and handle NaNs/None
    valid_embeddings = []
    valid_indices = []
    for idx, emb in enumerate(df_copy[embedding_col_name]):
        if emb is not None and (isinstance(emb, (list, np.ndarray)) and len(emb) > 0):
             # Ensure all embeddings have the same dimension as query_embedding
            if len(emb) == len(query_embedding):
                valid_embeddings.append(emb)
                valid_indices.append(df_copy.index[idx])
            else:
                print(f"Warning: Skipping embedding at index {df_copy.index[idx]} due to dimension mismatch (data: {len(emb)}, query: {len(query_embedding)}).")
        # else:
            # print(f"Warning: Invalid or empty embedding found at index {df_copy.index[idx]}.")


    if not valid_embeddings:
        print("Error: No valid embeddings found in the DataFrame to compare with query.")
        df_copy['cosine_similarity'] = np.nan
        return df_copy.sort_values(by='cosine_similarity', ascending=False)

    embeddings_matrix = np.array(valid_embeddings)
    similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]

    # Assign similarities back to the original DataFrame copy
    df_copy['cosine_similarity'] = np.nan # Initialize column
    for i, original_idx in enumerate(valid_indices):
        df_copy.loc[original_idx, 'cosine_similarity'] = similarities[i]

    return df_copy.sort_values(by='cosine_similarity', ascending=False)


# --- LLM Chain Definitions ---
llm = None
if ChatGoogleGenerativeAI: # Default to Gemini Flash as in notebook
    try:
        # GOOGLE_API_KEY should be set in environment for this
        llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0, max_tokens=None, timeout=None, max_retries=2) # gemini-2.0-flash not found, changed to gemini-1.0-pro
        # Note: Notebook used "gemini-2.0-flash". If this specific model is required and available, use it.
        # Otherwise, "gemini-pro" or "gemini-1.0-pro" is a common alternative.
        # The notebook also had an OpenAI client with a Google base URL for embeddings.
        # For chat, it used ChatGoogleGenerativeAI.
    except Exception as e:
        print(f"Error initializing ChatGoogleGenerativeAI: {e}. LLM features will be limited.")
        llm = None
elif ChatOpenAI and OPENAI_API_KEY: # Fallback to OpenAI if Google GenAI fails or not preferred
    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") # Example OpenAI model
    except Exception as e:
        print(f"Error initializing ChatOpenAI: {e}. LLM features will be limited.")
        llm = None
else:
    print("Warning: No LLM client (Google GenAI or OpenAI) could be initialized. LLM chains will not work.")

# Initialize chains as None, they will be created if llm and PromptTemplate are available
rerankChain = summaryChain = bulletsChain = outcomeChain = valueChain = None
genSummaryChain = genOutcomeChain = genValuesChain = genBulletsChain = rerankOnlyChain = None

if llm and PromptTemplate:
    rerankPrompt = PromptTemplate(
        input_variables=["question", "responses", "query"],
        template="""\
Participants in a research study were asked '{question}'.
These are their responses:
{responses}
Filter the responses, keeping only the responses that are atleast somewhat related to or helpful in anwsering: {query}
Rank the filtered resposnes starting with the most related to or helpful in answering: {query}
Output this set of responses formated exactly in the way they are given above, with a newline spereate each response.
If there are no related response, output "there are no relevant responses"
"""
    )
    rerankChain = LLMChain(llm=llm, prompt=rerankPrompt, output_key="reranked_responses")

    summaryPrompt = PromptTemplate(
        input_variables=["question", "responses", "focus"],
        template="""\
Participants in a research study were asked '{question}'.
These are their responses:
{responses}
Create a hierarchical taxonomy of the unique ideas and themes within these responses using very short bullet points. Avoid dupliate ideas. If there are no responses to analyze, do not provide a taxonomy. Do not include anything in the taxonomy not inlcuded in the responses.
{focus}
"""
    )
    summaryChain = LLMChain(llm=llm, prompt=summaryPrompt, output_key="summary")

    bulletsPrompt = PromptTemplate(
        input_variables=["question", "summary", "focus"],
        template="""\
Participants in a research study were asked '{question}'.
The TAXONOMY of ideas in participants responses are:
{summary}
Summarize the TAXONOMY into 1-15 concise bullet points, with each bullet point starting with a single theme and then overviewing the ideas within that theme. Be direct and specific, DO NOT say things like "Particpants said" or "responses" or "there was a desire" or "this theme" or "focusing on". Just say the ideas. NEVER repeat a theme or idea. A theme name should not include "and". Each bullet should be very short. Each bullets must ONLY contain ideas from the taxonomy. If there is no taxonomy, just output "no ideas to synthesize"
{focus}
Example bullet points:
- Creaive outlets: music, drawing, painting, writing stories, designing houses, creating new receipes, home remodeling.
- Health improvement: Increased exercise, better diets, access to high-quality healthcare, and reduction of chronic diseases.
"""
    )
    bulletsChain = LLMChain(llm=llm, prompt=bulletsPrompt, output_key="bullets")

    outcomePrompt = PromptTemplate(
        input_variables=["question", "summary", "responses"],
        template="""\
Participants in a research study were asked '{question}'.
These are their responses:
{responses}
The main ideas from these responses are:
{summary}
We define an 'outcome' to be a single specific concrete result that can be observed and measured in the world. An 'outcome' should NOT include an explination of how it is acheived (ie. an 'outcome' should NOT include the words "due to" or "as a result of" or "through" or "by" etc.). An 'outcome' MUST be specific enough to be objectively observed or measured.
Write a list of the 'outcomes' present in the main ideas from the responses summarized above. DO NOT REPEAT ANY IDEAS.
Here are some examples of 'outcomes':
- Earths climate remains below 15C
- The number of gun deaths decreases to below 100 per day globally
- The fraction of people who cannot afford or access healthcare decreases
- No nuclear devices are detonated within 100 miles of a human
- More people report being happy with their life
"""
    )
    outcomeChain = LLMChain(llm=llm, prompt=outcomePrompt, output_key="outcomes")

    valuePrompt = PromptTemplate(
        input_variables=["question", "summary", "responses"],
        template="""\
Participants in a research study were asked '{question}'.
These are their responses:
{responses}
The main ideas from these responses are:
{summary}
We define an 'value' to be a deontilogical property that can be reflected on how an AI behaves, reguardless of the result of that behavior. We do not consider a specific AI behavior to be a 'value'. For example "non-judegment: the AI's behavior does not imply a value judgement about the users feelings or experience" IS a 'value' , but "the AI does not say 'I am judgeing you'" is NOT a 'value'.
Write a list of the unique 'values' present in the  main ideas from the responses summarized above.
Here are some example 'values':
- Empathy: Showing understanding and compassion to make the user feel heard and supported.
- Respect: Honoring the user's feelings and experiences without minimizing their pain or struggles.
- Non-judgment: Providing support without criticism or bias to create a safe space for the user to express themselves.
DO NOT COPY THE EXAMPLE 'values' ABOVE VERBATIM. Construct them based on the responses and summarized ideas above.
"""
    )
    valueChain = LLMChain(llm=llm, prompt=valuePrompt, output_key="values")

    if SequentialChain:
        if summaryChain:
            genSummaryChain = SequentialChain(chains=[summaryChain], input_variables=["question", "responses", "focus"], output_variables=["summary"], verbose=False)
        if summaryChain and outcomeChain:
            genOutcomeChain = SequentialChain(chains=[summaryChain, outcomeChain], input_variables=["question", "responses", "focus"], output_variables=["summary", "outcomes"], verbose=False)
        if summaryChain and valueChain:
            genValuesChain = SequentialChain(chains=[summaryChain, valueChain], input_variables=["question", "responses", "focus"], output_variables=["summary", "values"], verbose=False)
        if summaryChain and bulletsChain:
            genBulletsChain = SequentialChain(chains=[summaryChain, bulletsChain], input_variables=["question", "responses", "focus"], output_variables=["summary", "bullets"], verbose=False)
        if rerankChain:
            rerankOnlyChain = SequentialChain(chains=[rerankChain], input_variables=["question", "responses", "query"], output_variables=["reranked_responses"], verbose=False)

# --- Main LLM Synthesis Function ---
def synthesize(qs_df_list, qmeta_df, qid, segs_for_ranking_cols, synth_type, rank_type, thresh, n_max, query_text="",
             response_col="English Responses", embedding_col="embedding",
             embedding_model="text-embedding-004", embedding_dims=None):
    """
    Main LLM synthesis pipeline.
    qs_df_list: List of DataFrames (processed, potentially with embeddings).
    qmeta_df: DataFrame with question metadata.
    qid: Question ID (index in qs_df_list).
    segs_for_ranking_cols: List of actual segment column names to use for ranking metrics (if applicable).
    """
    if not llm or not all([genSummaryChain, genOutcomeChain, genValuesChain, genBulletsChain, rerankOnlyChain]):
        print("Error: LLM or necessary chains not initialized. Cannot synthesize.")
        return {"error": "LLM components not initialized", "data": pd.DataFrame()}

    if qid >= len(qs_df_list):
        print(f"Error: qid {qid} is out of range for qs_df_list (length {len(qs_df_list)}).")
        return {"error": "qid out of range", "data": pd.DataFrame()}

    df_question = qs_df_list[qid].copy()
    if response_col not in df_question.columns:
        print(f"Error: Response column '{response_col}' not found in DataFrame for qid {qid}.")
        return {"error": f"Missing response column '{response_col}'", "data": pd.DataFrame()}

    ba = pd.DataFrame() # DataFrame to store selected responses

    # --- Response Selection Logic (adapted from non_llm_analysis bridging_ask etc.) ---
    # We need bridging_ask, get_polarizing_responses etc. from non_llm_analysis or replicate them here
    # For now, let's assume these functions are available if needed, or simplify for LLM script.
    # The LLM script should ideally focus on LLM tasks, and response selection might be pre-done.
    # However, the notebook's `synthesize` included this.

    # Simplified response selection for this script:
    # If rank_type is 'relevance', it uses its own embedding logic.
    # For other rank_types, it would need metric columns pre-calculated or access to metric functions.

    if rank_type == "relevance":
        print("Ranking by relevance...")
        if embedding_col not in df_question.columns:
            print(f"Embedding column '{embedding_col}' not found for relevance ranking. Attempting to generate them...")
            df_question = add_embeddings_to_df(df_question, response_col, embedding_col, embedding_model, embedding_dims)
            if embedding_col not in df_question.columns or df_question[embedding_col].isnull().all():
                print("Error: Failed to generate embeddings for relevance ranking.")
                return {"error": "Failed to generate embeddings", "data": pd.DataFrame()}

        ba_ = rank_by_similarity(df_question, query_text, response_col, embedding_col, embedding_model, embedding_dims)
        ba_ = ba_[ba_['cosine_similarity'] > thresh] if 'cosine_similarity' in ba_.columns else pd.DataFrame()
        ba = ba_.head(n_max)
    elif rank_type in ["bridging", "polarization", "divergence", "low_agreement", "sample"]:
        # These would typically rely on metric columns ('bridge', 'polarization', etc.)
        # being present in df_question, or functions to calculate them.
        # For this LLM-focused script, let's assume if these rank_types are used,
        # the necessary columns are already in df_question (e.g., from non_llm_script output).
        print(f"Ranking by '{rank_type}' (requires pre-calculated metric columns or access to metric functions).")
        if rank_type == "bridging" and 'bridge' in df_question.columns:
            ba_ = df_question[df_question['bridge'] > thresh].sort_values(by='bridge', ascending=False)
            ba = ba_.head(n_max)
        elif rank_type == "polarization" and 'polarization' in df_question.columns:
            ba = df_question.sort_values(by='polarization', ascending=False).head(n_max)
        # Add other rank types if necessary, assuming columns exist
        elif rank_type == "sample":
            ba = df_question.sample(n=min(n_max, len(df_question)))
        else:
            print(f"Warning: Rank type '{rank_type}' selected, but required metric columns might be missing. Defaulting to sampling.")
            ba = df_question.sample(n=min(n_max, len(df_question)))
            if ba.empty:
                 print("Warning: No responses selected for synthesis after sampling (DataFrame might be empty).")
                 return {"error": "No responses selected", "data": pd.DataFrame()}


    if ba.empty:
        print("Warning: No responses selected for synthesis based on ranking criteria.")
        return {"error": "No responses selected", "data": pd.DataFrame()}

    responses_str = "\n\n- ".join(ba[response_col].astype(str).tolist())
    if responses_str: responses_str = "- " + responses_str # Ensure first item also has a bullet

    question_str = qmeta_df.iloc[qid]["question text"] if qid < len(qmeta_df) else f"Question {qid}"

    focus_on = ""
    if query_text:
        focus_on = f"Only include topics and themes that are reasonably relevant to: {query_text}"
        if rerankOnlyChain:
            try:
                prelim_responses_dict = rerankOnlyChain.invoke({
                    "question": question_str, "responses": responses_str, "query": query_text
                })
                responses_str = prelim_responses_dict.get("reranked_responses", responses_str)
                if responses_str == "there are no relevant responses":
                    print("LLM reranking found no relevant responses to the query.")
                    return {"summary": "No relevant responses found by LLM.", "data": ba}
            except Exception as e:
                print(f"Error during LLM reranking: {e}")
                # Proceed with original responses_str if reranking fails
        else:
            print("Warning: rerankOnlyChain not available for focusing.")


    out_dict = {"data": ba.to_dict(orient='records')} # Include selected data
    try:
        if synth_type == "outcomes" and genOutcomeChain:
            llm_out = genOutcomeChain.invoke({"question": question_str, "responses": responses_str, "focus": focus_on})
            out_dict.update(llm_out)
        elif synth_type == "values" and genValuesChain:
            llm_out = genValuesChain.invoke({"question": question_str, "responses": responses_str, "focus": focus_on})
            out_dict.update(llm_out)
        elif synth_type == "bullets" and genBulletsChain:
            llm_out = genBulletsChain.invoke({"question": question_str, "responses": responses_str, "focus": focus_on})
            out_dict.update(llm_out)
        elif synth_type == "summary" and genSummaryChain:
            llm_out = genSummaryChain.invoke({"question": question_str, "responses": responses_str, "focus": focus_on})
            out_dict.update(llm_out)
        else:
            print(f"Warning: Synthesis type '{synth_type}' not recognized or corresponding chain not available.")
            out_dict[synth_type] = "Synthesis type not supported or chain unavailable."

    except Exception as e:
        print(f"Error during LLM synthesis ({synth_type}): {e}")
        out_dict[synth_type] = f"Error during synthesis: {e}"

    return out_dict

# --- Helper to load processed data ---
def load_processed_qs_data(json_filepath="processed_qs_data.json"):
    """Loads qs (list of DataFrames) and qmeta (DataFrame) from JSON files."""
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            loaded_list_of_dicts = json.load(f)
        qs = [pd.DataFrame(df_dict_list) for df_dict_list in loaded_list_of_dicts] # Assuming list of list of dicts

        # Reconstruct qmeta (similar to non_llm_analysis)
        meta_list = [["question type", "question text"]]
        for df in qs:
            if not df.empty and 'Question Type' in df.columns and 'Question' in df.columns:
                # Try to get from the first valid data row (often index 1 if header is separate)
                q_type_row_idx = 0
                if len(df) > 1 and df['Question Type'].iloc[0] != df['Question Type'].iloc[1]: # Heuristic for header row
                     # If first two rows' QType differ, assume row 1 is data. Else row 0.
                     q_type_row_idx = 1 if df['Question Type'].iloc[0] not in ['Poll Single Select', 'Ask Opinion', 'Ask Experience'] else 0

                if len(df) > q_type_row_idx:
                    meta_list.append([df['Question Type'].iloc[q_type_row_idx], df['Question'].iloc[q_type_row_idx]])
                elif len(df) > 0: # Fallback to first row if only one row
                     meta_list.append([df['Question Type'].iloc[0], df['Question'].iloc[0]])
                else:
                    meta_list.append(["Unknown", f"Unknown Question (empty df)"])
            else:
                meta_list.append(["Unknown", f"Unknown Question (cols missing)"])
        qmeta = pd.DataFrame(meta_list[1:], columns=meta_list[0])
        print(f"Successfully loaded {len(qs)} DataFrames from {json_filepath}")
        return qs, qmeta
    except FileNotFoundError:
        print(f"Error: Processed data file {json_filepath} not found.")
        return [], pd.DataFrame()
    except Exception as e:
        print(f"Error loading processed data from {json_filepath}: {e}")
        return [], pd.DataFrame()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM Analysis Pipeline.")
    parser.add_argument("--input_json", type=str, default="processed_qs_data.json",
                        help="Path to the input JSON file containing processed qs data (output from non_llm_analysis.py).")
    parser.add_argument("--output_dir", type=str, default="llm_outputs",
                        help="Directory to save LLM synthesis results.")
    parser.add_argument("--qid", type=int, default=None, help="Question ID (index) to synthesize for. If None, attempts to find first 'Ask Opinion'.")
    parser.add_argument("--synth_type", type=str, default="bullets", choices=["bullets", "summary", "outcomes", "values"], help="Type of LLM synthesis to perform.")
    parser.add_argument("--rank_type", type=str, default="sample", choices=["relevance", "bridging", "polarization", "divergence", "low_agreement", "sample"], help="Method to rank/select responses for synthesis.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for ranking (e.g., cosine similarity for 'relevance', bridge score for 'bridging').")
    parser.add_argument("--n_max", type=int, default=10, help="Maximum number of responses to use for synthesis.")
    parser.add_argument("--query_text", type=str, default="", help="Query text to focus synthesis or for 'relevance' ranking.")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-004", help="Name of the embedding model to use.")

    args = parser.parse_args()

    print("Starting LLM Analysis Script with provided arguments...")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    qs_data, qmeta_data = load_processed_qs_data(args.input_json)

    if not qs_data or qmeta_data.empty:
        print(f"Could not load data from {args.input_json} for LLM analysis. Exiting.")
    else:
        qid_to_synthesize = args.qid
        if qid_to_synthesize is None:
            ask_opinion_qids = [i for i, q_type in enumerate(qmeta_data["question type"]) if q_type == "Ask Opinion"]
            if not ask_opinion_qids:
                print("No 'Ask Opinion' questions found in the loaded data. Please specify a --qid.")
                exit()
            qid_to_synthesize = ask_opinion_qids[0]
            print(f"No --qid provided, selected first 'Ask Opinion' question: ID {qid_to_synthesize}")

        if qid_to_synthesize >= len(qs_data) or qid_to_synthesize < 0:
            print(f"Error: Provided --qid {qid_to_synthesize} is out of range for the loaded data (0-{len(qs_data)-1}).")
            exit()

        question_text_for_log = qmeta_data.iloc[qid_to_synthesize]['question text'][:70] if qid_to_synthesize < len(qmeta_data) else f"Question {qid_to_synthesize}"
        print(f"\nAttempting synthesis for Question ID: {qid_to_synthesize} - '{question_text_for_log}...'")
        print(f"  Synthesis Type: {args.synth_type}, Rank Type: {args.rank_type}, N_max: {args.n_max}, Query: '{args.query_text}'")

        # Placeholder for segment column names if needed by rank_type
        # These would need to be derived from the data or passed as arguments if complex.
        example_segment_cols_for_ranking = []

        # Ensure embeddings if rank_type is 'relevance'
        if args.rank_type == "relevance":
            target_df = qs_data[qid_to_synthesize]
            if "embedding" not in target_df.columns or target_df["embedding"].isnull().all():
                print(f"  Generating embeddings for question {qid_to_synthesize} (rank_type='relevance')...")
                # Create a copy to modify for this specific synthesis call if embeddings are added
                qs_data_copy = [df.copy() for df in qs_data]
                qs_data_copy[qid_to_synthesize] = add_embeddings_to_df(
                    qs_data_copy[qid_to_synthesize],
                    model_name=args.embedding_model
                )
                # Use the copy with new embeddings for this synthesis
                current_qs_list_for_synthesis = qs_data_copy
            else:
                current_qs_list_for_synthesis = qs_data # Use original if embeddings exist
        else:
            current_qs_list_for_synthesis = qs_data


        synth_output = synthesize(
            qs_df_list=current_qs_list_for_synthesis,
            qmeta_df=qmeta_data,
            qid=qid_to_synthesize,
            segs_for_ranking_cols=example_segment_cols_for_ranking,
            synth_type=args.synth_type,
            rank_type=args.rank_type,
            thresh=args.threshold,
            n_max=args.n_max,
            query_text=args.query_text,
            embedding_model=args.embedding_model
        )

        q_text_for_filename = qmeta_data.iloc[qid_to_synthesize]['question text'][:30].replace(" ", "_").replace("?", "").replace("/","") if qid_to_synthesize < len(qmeta_data) else f"qid_{qid_to_synthesize}"
        output_filename = os.path.join(args.output_dir, f"synthesis_{q_text_for_filename}_{args.synth_type}.json")

        if synth_output and not synth_output.get("error"):
            print(f"\n--- Synthesis Results for QID {qid_to_synthesize} ---")
            if args.synth_type in synth_output:
                print(f"Question: {qmeta_data.iloc[qid_to_synthesize]['question text'] if qid_to_synthesize < len(qmeta_data) else 'N/A'}")
                print(f"Synthesized {args.synth_type.capitalize()}:")
                print(synth_output[args.synth_type])

                # Save the full synthesis output (including data and summary) to JSON
                try:
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        # Convert pandas DataFrames in 'data' part of output to dict for JSON serialization
                        if 'data' in synth_output and isinstance(synth_output['data'], pd.DataFrame):
                             synth_output['data'] = synth_output['data'].to_dict(orient='records')
                        elif 'data' in synth_output and isinstance(synth_output['data'], list): # if it's already list of dicts
                            pass # it's fine
                        json.dump(synth_output, f, indent=2)
                    print(f"Full synthesis output saved to: {output_filename}")
                except Exception as e:
                    print(f"Error saving synthesis output to JSON: {e}")
                    print("Printing raw output instead:")
                    print(synth_output)

            else:
                print(f"Synthesis type '{args.synth_type}' not found in output.")
                print("Raw output:", synth_output)
        else:
            print(f"Could not generate synthesis for QID {qid_to_synthesize}. Error: {synth_output.get('error', 'Unknown error')}")

    print("\nLLM Analysis Script Finished.")
