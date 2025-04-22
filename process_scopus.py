# -*- coding: utf-8 -*-
# process_scopus_v1_1.py

import pandas as pd
import glob
import os
import re # Ensure re is imported

# --- Configuration ---
# Option 1: Relative path (if running script from BoE_Paper directory)
# FOLDER_PATH = './Sub-databases'
# Option 2: Absolute path (works from anywhere)
FOLDER_PATH = '/Users/simoneabbiati/Desktop/BoE_Paper/Sub-databases' # <--- *** VERIFY this path ***

OUTPUT_FILENAME = 'merged_cleaned_deduplicated_citations_filtered.csv' # Sensible output name

# --- Columns ---
CITATION_COLUMN = 'Citation count' # <--- *** VERIFY this column name ***

# List ALL potential abstract columns used during keyword filtering.
# These will be dropped AFTER filtering and deduplication.
# Include the original abstract column name from your input files AND
# the 'Scraped_Abstract' column if it was added by the scraper.
ABSTRACT_COLUMNS_TO_DROP_AND_CHECK = ['Abstract summary', 'Scraped_Abstract'] # <--- *** VERIFY/ADJUST these ***

# --- Deduplication Settings ---
DEDUPLICATION_COLUMNS = ['Title', 'Authors', 'Year'] # <--- *** ADJUST this list if needed ***
CONVERT_TO_LOWERCASE = True # Case-insensitive deduplication

# --- Medical Keyword Filtering Settings ---
FILTER_MEDICAL_KEYWORDS = True # Set to False to disable filtering
MEDICAL_KEYWORDS_REGEX = r'\b(heal[a-z]*|medic[a-z]*|clinic[a-z]*)\b'
# Columns to search within for medical keywords. MUST include 'Title' and relevant abstract columns.
COLUMNS_TO_CHECK_FOR_MEDICAL = ['Title'] + [col for col in ABSTRACT_COLUMNS_TO_DROP_AND_CHECK if col != 'Title'] # Automatically include Title + Abstract cols

# --- End Configuration ---

def clean_string(text):
    """Helper function to clean string data, handling potential NaNs."""
    if pd.isna(text):
        return "" # Return empty string for NaN/None
    if not isinstance(text, str):
        text = str(text) # Convert non-strings (like numbers) to string

    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    # Optional: Add more cleaning rules here if needed
    # text = text.replace('\n', ' ').replace('\r', '')
    return text

def process_and_deduplicate_cleaned_files(folder_path, citation_col, abstract_cols_to_drop, dedupe_cols, use_lowercase, output_file):
    """
    Reads, filters by citation, merges, cleans, filters medical terms,
    deduplicates, drops abstract columns, and saves CSV files.
    """
    all_filtered_data = []
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        print(f"Error: No CSV files found in the specified directory: {folder_path}")
        return
    print(f"Found {len(csv_files)} CSV files to process...")

    # 1. Read each file, filter by citation count
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print(f"  Processing: {filename}...")
        try:
            # Attempt to read with common encodings
            try:
                df = pd.read_csv(filepath)
            except UnicodeDecodeError:
                print(f"    Trying latin1 encoding for {filename}...")
                df = pd.read_csv(filepath, encoding='latin1')
            except Exception as read_e:
                 print(f"    Failed to read {filename} with UTF-8 and latin1: {read_e}. Skipping.")
                 continue

            # Check for essential columns
            if citation_col not in df.columns:
                print(f"    Warning: Citation column '{citation_col}' not found in {filename}. Skipping file.")
                continue

            # Check for columns needed for filtering/deduplication and issue warnings
            cols_needed = set(dedupe_cols)
            if FILTER_MEDICAL_KEYWORDS: cols_needed.update(COLUMNS_TO_CHECK_FOR_MEDICAL)
            missing_cols = [col for col in cols_needed if col not in df.columns]
            if missing_cols:
                print(f"    Warning: Columns needed for processing not found in {filename}: {missing_cols}. Results might be affected.")

            # Filter by citation count
            df[citation_col] = pd.to_numeric(df[citation_col], errors='coerce')
            filtered_df = df[df[citation_col].fillna(0) > 0].copy()

            if not filtered_df.empty:
                 all_filtered_data.append(filtered_df)
            else:
                 print(f"    No rows with > 0 citations found or kept in {filename}.")

        except Exception as e:
            print(f"    Error processing {filename}: {e}. Skipping.")

    if not all_filtered_data:
        print("\nError: No data collected from files after citation filtering.")
        return

    # 2. Merge datasets
    print(f"\nMerging data from {len(all_filtered_data)} files...")
    merged_df = pd.concat(all_filtered_data, ignore_index=True)
    print(f"Total rows after merging: {len(merged_df)}")

    # 3. *** Data Cleaning Step (BEFORE Filtering/Deduplication) ***
    print("\nCleaning data (stripping whitespace)...")
    string_columns = merged_df.select_dtypes(include=['object', 'string']).columns
    for col in string_columns:
        if col in merged_df.columns: # Ensure column still exists
             print(f"  Cleaning column: {col}")
             merged_df[col] = merged_df[col].apply(clean_string)
        else:
             print(f"  Skipping cleaning for column '{col}' (not found in merged data).")

    # --- 4. *** Medical Keyword Filtering Step *** ---
    initial_rows_before_med_filter = len(merged_df)
    rows_removed_medical = 0
    if FILTER_MEDICAL_KEYWORDS:
        print(f"\nFiltering out rows containing medical keywords ({MEDICAL_KEYWORDS_REGEX})...")
        # Identify columns to check that actually exist in the merged dataframe
        check_cols_exist = [col for col in COLUMNS_TO_CHECK_FOR_MEDICAL if col in merged_df.columns]
        if not check_cols_exist:
            print("  Warning: None of the specified columns for medical keyword checking exist. Skipping filter.")
        else:
            print(f"  Checking columns: {check_cols_exist}")
            mask = pd.Series(False, index=merged_df.index) # Start with False
            for col in check_cols_exist:
                # Ensure column is treated as string for `.str` methods, handle NaN safely
                mask |= merged_df[col].fillna('').astype(str).str.contains(MEDICAL_KEYWORDS_REGEX, case=False, regex=True)

            merged_df = merged_df[~mask].copy() # Keep rows where mask is False, use .copy()
            rows_removed_medical = initial_rows_before_med_filter - len(merged_df)
            print(f"  Removed {rows_removed_medical} rows containing medical keywords.")
    else:
        print("\nMedical keyword filtering is disabled.")
    # --- End Medical Filtering ---


    # 5. *** Deduplication Step (using cleaned, filtered data) ***
    initial_rows_before_dedupe = len(merged_df) # Rows after potential medical filter
    print(f"\nPreparing for deduplication using columns: {dedupe_cols}")
    temp_lower_cols = []; dedupe_subset = dedupe_cols[:]
    if use_lowercase:
        print("Converting specified deduplication columns to lowercase for comparison...")
        for col in dedupe_cols:
            if col in merged_df.columns and merged_df[col].dtype in ['object', 'string']:
                temp_col_name = f"__{col}_lower__"
                # Use fillna('') before lower to handle potential NaN values gracefully
                merged_df[temp_col_name] = merged_df[col].fillna('').str.lower()
                temp_lower_cols.append(temp_col_name)
                # Replace original col name with temp lower name in the subset list
                if col in dedupe_subset: # Ensure it's actually in the list before replacing
                     dedupe_subset[dedupe_subset.index(col)] = temp_col_name
            else: print(f"  Warning: Column '{col}' for lowercase conversion is not text/found.")

    # Check if dedupe_subset contains valid columns before proceeding
    valid_dedupe_subset = [col for col in dedupe_subset if col in merged_df.columns]
    if not valid_dedupe_subset:
         print("Error: No valid columns found for deduplication after processing. Cannot deduplicate.")
         deduplicated_df = merged_df # Keep the dataframe as is
    else:
        print(f"Deduplicating based on effective columns: {valid_dedupe_subset}...")
        deduplicated_df = merged_df.drop_duplicates(subset=valid_dedupe_subset, keep='first').copy()

    final_rows = len(deduplicated_df)
    duplicates_removed = initial_rows_before_dedupe - final_rows
    if temp_lower_cols:
        print("Removing temporary lowercase columns...")
        cols_to_drop_temp = [col for col in temp_lower_cols if col in deduplicated_df.columns] # Ensure cols exist
        if cols_to_drop_temp:
             deduplicated_df.drop(columns=cols_to_drop_temp, inplace=True)

    print(f"\nDeduplication complete.")
    print(f"  - Rows before deduplication (after merge & filters): {initial_rows_before_dedupe}")
    print(f"  - Final rows after deduplication: {final_rows}")
    print(f"  - Number of duplicate articles removed: {duplicates_removed}")


    # 6. Drop specified Abstract Columns (do this LAST)
    actual_abstract_cols_to_drop = [col for col in abstract_cols_to_drop if col in deduplicated_df.columns]
    if actual_abstract_cols_to_drop:
        print(f"\nDropping specified abstract columns: {actual_abstract_cols_to_drop}...")
        deduplicated_df.drop(columns=actual_abstract_cols_to_drop, inplace=True)
    else:
        print("\nNo specified abstract columns found in the final data to drop.")


    # 7. Save the final dataset
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        deduplicated_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nSuccessfully saved the final filtered & deduplicated dataset to '{output_file}'")
    except Exception as e: print(f"\nError saving the final dataset: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    print("--- Starting CSV Processing Script (v1.1 - Added Medical Filter) ---")

    # Check if the explicitly defined folder path exists
    if not os.path.isdir(FOLDER_PATH):
        print(f"Error: The specified directory does not exist or is not accessible: {FOLDER_PATH}")
        print("Please check the 'FOLDER_PATH' variable in the script and ensure the directory exists.")
        exit() # Stop the script if the correct folder isn't found
    else:
        print(f"Processing files from folder: {FOLDER_PATH}")

    # Check deduplication columns
    if not DEDUPLICATION_COLUMNS:
         print("Error: The 'DEDUPLICATION_COLUMNS' list is empty. Please specify columns to identify unique rows.")
         exit()
    print(f"Using columns for deduplication: {DEDUPLICATION_COLUMNS}")
    print(f"Case-insensitive deduplication: {CONVERT_TO_LOWERCASE}")

    # Check medical filtering settings
    if FILTER_MEDICAL_KEYWORDS:
        print(f"Medical keyword filtering enabled. Regex: '{MEDICAL_KEYWORDS_REGEX}'")
        print(f"Checking columns: {COLUMNS_TO_CHECK_FOR_MEDICAL}")
    else:
        print("Medical keyword filtering disabled.")


    process_and_deduplicate_cleaned_files(
        FOLDER_PATH,
        CITATION_COLUMN,
        ABSTRACT_COLUMNS_TO_DROP_AND_CHECK, # Pass the list of columns to drop
        DEDUPLICATION_COLUMNS,
        CONVERT_TO_LOWERCASE,
        OUTPUT_FILENAME
    )

    print("\n--- Script Finished ---")