import pandas as pd
import torch # Make sure PyTorch is installed
from transformers import pipeline
import warnings
import time
from math import ceil

# Suppress specific future warnings from Huggingface/PyTorch if they appear
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
# !!! User-provided paths and settings !!!
CSV_FILE_PATH = "/Users/simoneabbiati/Desktop/BoE_Paper/bibliography_processedFALSE.csv" # !!! CHANGE THIS !!!
ABSTRACT_COLUMN = "Scraped_Abstract" # Exact column name for abstracts
TITLE_COLUMN = None # Set to None as requested, will use generic IDs (Change if you have titles)

# --- Zero-Shot Classification Setup ---
# Define the candidate labels. These should reflect the categories you're interested in.
# Phrasing matters! Try to phrase them as statements the text might entail.
candidate_labels = [
    "AI systems in finance lack sufficient explainability",
    "Transparency is a challenge for AI adoption in financial services",
    "Difficulty in understanding AI decision-making is a barrier in finance",
    "Poor data quality hinders AI adoption in finance and regulation",
    "Data privacy concerns limit the use of AI in financial applications",
    "Regulatory restrictions on data use are barriers to AI in finance",
    "Regulatory compliance is a major barrier to AI in finance",
    "Ethical considerations impede AI adoption in financial regulation",
    "AI-specific regulations pose challenges for financial institutions",
    "Integrating AI with existing systems is a barrier in finance",
    "High implementation costs are a barrier to AI in financial services",
    "Lack of standardization in AI tools slows adoption in finance",
    "Organizational resistance to change hinders AI adoption in finance",
    "Lack of internal expertise is a barrier to AI in financial institutions",
    "Security risks associated with AI are a concern in financial regulation",
    "Bias in AI algorithms is a barrier to ethical AI in finance",
    "Developing clear business cases to promote AI in finance",
    "Demonstrating the value of AI for compliance and risk management",
    "Highlighting the benefits of AI for customer insights in finance",
    "Improving data quality is crucial for successful AI in finance",
    "Strategies to enhance data availability for AI in financial regulation",
    "Investing in data cleaning and curation for AI in finance",
    "Building internal AI expertise is key for financial institutions",
    "Training programs are needed to develop AI skills in the finance sector",
    "Hiring AI specialists to strengthen financial organizations",
    "Proactively addressing regulatory concerns to enable AI in finance",
    "Collaborating with regulators to create AI compliance frameworks",
    "Aligning AI implementation with financial regulatory requirements",
    "Planning for seamless integration of AI into financial systems",
    "Adopting modular AI solutions for legacy financial infrastructure",
    "Promoting standardization of AI tools in the financial industry",
    "Investing in education and awareness to foster AI adoption in finance",
    "Establishing clear policies for ethical and secure AI usage in finance",
    "Developing privacy and bias mitigation policies for AI in financial regulation",
    "Building organizational support for AI through education and communication",
    "Technical aspects or methods of AI or machine learning",
    "Regulatory compliance or financial risk management using AI",
    "General discussion of AI in finance (neutral focus)",
    "General discussion of financial or regulatory topics (non-AI focus)",
    "Other AI topic (not related to finance/regulation)"
]

# Define mapping for higher-level categories (using structured definition)
category_to_labels_mapping = {  # Renamed for clarity to reflect structure
    "Explainability and Transparency Barriers": [
        "AI systems in finance lack sufficient explainability",
        "Transparency is a challenge for AI adoption in financial services",
        "Difficulty in understanding AI decision-making is a barrier in finance"
    ],
    "Data Quality and Privacy Barriers": [
        "Poor data quality hinders AI adoption in finance and regulation",
        "Data privacy concerns limit the use of AI in financial applications",
        "Regulatory restrictions on data use are barriers to AI in finance"
    ],
    "Regulatory and Ethical Barriers": [
        "Regulatory compliance is a major barrier to AI in finance",
        "Ethical considerations impede AI adoption in financial regulation",
        "AI-specific regulations pose challenges for financial institutions"
    ],
    "Integration and Cost Barriers": [
        "Integrating AI with existing systems is a barrier in finance",
        "High implementation costs are a barrier to AI in financial services"
    ],
    "Organizational and Human Barriers": [
        "Organizational resistance to change hinders AI adoption in finance",
        "Lack of internal expertise is a barrier to AI in financial institutions",
        "Security risks associated with AI are a concern in financial regulation",
        "Bias in AI algorithms is a barrier to ethical AI in finance"
    ],
    "Business Case and Value Demonstration Strategies": [
        "Developing clear business cases to promote AI in finance",
        "Demonstrating the value of AI for compliance and risk management",
        "Highlighting the benefits of AI for customer insights in finance"
    ],
    "Data Improvement and Availability Strategies": [
        "Improving data quality is crucial for successful AI in finance",
        "Strategies to enhance data availability for AI in financial regulation",
        "Investing in data cleaning and curation for AI in finance"
    ],
    "Expertise and Training Strategies": [
        "Building internal AI expertise is key for financial institutions",
        "Training programs are needed to develop AI skills in the finance sector",
        "Hiring AI specialists to strengthen financial organizations"
    ],
    "Regulatory Engagement and Proactive Compliance Strategies": [
        "Proactively addressing regulatory concerns to enable AI in finance",
        "Collaborating with regulators to create AI compliance frameworks",
        "Aligning AI implementation with financial regulatory requirements"
    ],
    "Integration and Standardization Strategies": [
        "Planning for seamless integration of AI into financial systems",
        "Adopting modular AI solutions for legacy financial infrastructure",
        "Promoting standardization of AI tools in the financial industry",
        "Lack of standardization in AI tools slows adoption in finance" # Moved this here for consistency in category name
    ],
    "Education, Awareness, and Policy Strategies": [
        "Investing in education and awareness to foster AI adoption in finance",
        "Establishing clear policies for ethical and secure AI usage in finance",
        "Developing privacy and bias mitigation policies for AI in financial regulation",
        "Building organizational support for AI through education and communication"
    ],
    "Other Categories": [
        "Technical aspects or methods of AI or machine learning",
        "Regulatory compliance or financial risk management using AI",
        "General discussion of AI in finance (neutral focus)",
        "General discussion of financial or regulatory topics (non-AI focus)",
        "Other AI topic (not related to finance/regulation)"
    ]
}

# Now, create a reversed mapping for efficient lookup of category from label
label_to_category_mapping = {}
for category, labels in category_to_labels_mapping.items():
    for label in labels:
        label_to_category_mapping[label] = category

# Use label_to_category_mapping in your result processing
category_mapping = label_to_category_mapping # Rename to original variable name for rest of script to work without changes


MODEL_NAME = "facebook/bart-large-mnli"

# Processing parameters
BATCH_SIZE = 8 # Process abstracts in batches for efficiency. Adjust based on RAM/VRAM.
DEVICE = 0 if torch.cuda.is_available() else -1 # Use GPU (device 0) if available, else CPU (device -1)

# --- 1. Load Data ---
print(f"Loading data from: {CSV_FILE_PATH}")
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"CSV loaded successfully. Found {len(df)} rows.")
except FileNotFoundError:
    print(f"ERROR: File not found at {CSV_FILE_PATH}")
    exit()
except Exception as e:
    print(f"ERROR: Could not read CSV file. Error: {e}")
    exit()

print("Columns found in CSV:", list(df.columns))

if ABSTRACT_COLUMN not in df.columns:
    print(f"ERROR: Abstract column '{ABSTRACT_COLUMN}' not found.")
    print(f"Available columns are: {list(df.columns)}")
    exit()

# Ensure abstracts are strings and handle missing values
df['abstract_text'] = df[ABSTRACT_COLUMN].fillna('').astype(str)
abstracts_to_classify = df['abstract_text'].tolist()

# Get titles or create IDs
if TITLE_COLUMN and TITLE_COLUMN in df.columns:
    df['title_or_id'] = df[TITLE_COLUMN].fillna("No Title").astype(str)
else:
    if TITLE_COLUMN:
         print(f"Warning: Title column '{TITLE_COLUMN}' specified but not found. Using generic IDs.")
    df['title_or_id'] = [f"Document_{i}" for i in range(len(df))]

# Filter out potentially empty abstracts if needed (though classification might handle them)
original_count = len(abstracts_to_classify)
abstracts_to_classify = [a for a in abstracts_to_classify if len(a.strip()) > 10] # Keep only abstracts with some content
print(f"Filtered out {original_count - len(abstracts_to_classify)} abstracts shorter than 10 chars.")

if not abstracts_to_classify:
    print("ERROR: No abstracts left to classify after filtering.")
    exit()

# --- 2. Initialize Zero-Shot Pipeline ---
print(f"Initializing Zero-Shot classification pipeline with model: {MODEL_NAME}")
print(f"Using device: {'GPU (CUDA)' if DEVICE == 0 else 'CPU'}")
if DEVICE == 0:
    print("Ensure PyTorch is installed with CUDA support.")

try:
    # multi_label=False means scores sum to 1 (softmax); True means independent scores (sigmoid)
    # For picking the single best category, multi_label=False is often preferred.
    classifier = pipeline("zero-shot-classification",
                          model=MODEL_NAME,
                          device=DEVICE)
    print("Pipeline initialized successfully.")
except Exception as e:
    print(f"ERROR: Failed to initialize the pipeline. Error: {e}")
    print("Make sure the model name is correct and libraries are installed.")
    exit()

# --- 3. Classify Abstracts in Batches ---
print(f"Starting classification of {len(abstracts_to_classify)} abstracts...")
print(f"Using batch size: {BATCH_SIZE}")

results = []
start_time = time.time()
num_batches = ceil(len(abstracts_to_classify) / BATCH_SIZE)

for i in range(0, len(abstracts_to_classify), BATCH_SIZE):
    batch_texts = abstracts_to_classify[i : i + BATCH_SIZE]
    batch_start_time = time.time()

    try:
        # Pass the batch and candidate labels to the classifier
        batch_results = classifier(batch_texts, candidate_labels, multi_label=False)
        results.extend(batch_results)

    except Exception as e:
        print(f"\nERROR processing batch starting at index {i}. Error: {e}")
        print("Skipping this batch. Results might be incomplete.")
        # Add placeholders or handle error as needed
        results.extend([None] * len(batch_texts)) # Add None placeholders for failed batch

    batch_end_time = time.time()
    batch_num = (i // BATCH_SIZE) + 1
    print(f"  Processed Batch {batch_num}/{num_batches} "
          f"({len(batch_texts)} abstracts) in {batch_end_time - batch_start_time:.2f} seconds.")

end_time = time.time()
print(f"\nClassification finished in {end_time - start_time:.2f} seconds.")

# --- 4. Process and Analyze Results ---
print("Processing classification results...")

# Check if results list length matches the number of classified abstracts
if len(results) != len(abstracts_to_classify):
     print(f"Warning: Number of results ({len(results)}) does not match number of abstracts classified ({len(abstracts_to_classify)}). There might have been errors.")
     # Pad results with Nones if necessary, though this indicates a problem
     results.extend([None] * (len(abstracts_to_classify) - len(results)))

predicted_labels = []
predicted_scores = []
predicted_categories = [] # New list for higher-level categories

for res in results:
    if res is not None and 'labels' in res and 'scores' in res:
        top_label = res['labels'][0]
        predicted_labels.append(top_label) # Get the top label
        predicted_scores.append(res['scores'][0]) # Get the score for the top label
        predicted_categories.append(category_mapping.get(top_label, "Uncategorized")) # Get category from mapping, default "Uncategorized"
    else:
        predicted_labels.append("Classification Failed")
        predicted_scores.append(0.0)
        predicted_categories.append("Classification Failed") # Or handle uncategorized differently

# Add results back to the *original* DataFrame
if len(predicted_labels) == len(df): # Check if counts match after potential filtering
     df['zsl_predicted_label'] = predicted_labels
     df['zsl_score'] = predicted_scores
     df['zsl_category'] = predicted_categories # Add the new category column
else:
     print("Warning: Length mismatch after filtering/classification. Results might not align perfectly with original DataFrame.")
     # Attempt to add to a potentially filtered df (less ideal)
     temp_df = pd.DataFrame({'abstract_text': abstracts_to_classify,
                              'zsl_predicted_label': predicted_labels,
                              'zsl_score': predicted_scores,
                              'zsl_category': predicted_categories}) # Include new category
     # Merge back based on abstract text (can be slow and fail on duplicates)
     df = pd.merge(df, temp_df, on='abstract_text', how='left')


# Analyze the distribution of labels and categories
print("\n--- Distribution of Predicted Labels ---")
if 'zsl_predicted_label' in df.columns:
     print(df['zsl_predicted_label'].value_counts())

print("\n--- Distribution of Predicted Categories ---")
if 'zsl_category' in df.columns:
    print(df['zsl_category'].value_counts())


     # Filter for abstracts classified under your target labels (example adjusted for categories)
    target_category_1 = "Explainability and Transparency Barriers" # Example category
    target_category_2 = "Education, Awareness, and Policy Strategies" # Example category

    df_category_1 = df[df['zsl_category'] == target_category_1].sort_values('zsl_score', ascending=False)
    df_category_2 = df[df['zsl_category'] == target_category_2].sort_values('zsl_score', ascending=False)

    print(f"\n--- Top Abstracts Classified as Category '{target_category_1}' ---")
    if not df_category_1.empty:
        print(df_category_1[['title_or_id', 'zsl_predicted_label', 'zsl_score']].head(10))
    else:
        print("No abstracts classified in this category.")


    print(f"\n--- Top Abstracts Classified as Category '{target_category_2}' ---")
    if not df_category_2.empty:
        print(df_category_2[['title_or_id', 'zsl_predicted_label', 'zsl_score']].head(10))
    else:
        print("No abstracts classified in this category.")


     # Select the top ~20 papers combining both example categories (or focus on one)
    top_relevant_papers_zsl = pd.concat([df_category_1, df_category_2]).sort_values('zsl_score', ascending=False).head(20)

    print(f"\n--- Top 20 Papers Related to Categories '{target_category_1}' and '{target_category_2}' (Zero-Shot Classification) ---")
    if not top_relevant_papers_zsl.empty:
        print(top_relevant_papers_zsl[['title_or_id', 'zsl_category', 'zsl_predicted_label', 'zsl_score']])
    else:
        print("No abstracts in the top combined categories.")


     # --- 5. Save Results ---
    output_filename = "abstracts_with_zero_shot_classification.csv"
    try:
         df.to_csv(output_filename, index=False)
         print(f"\nResults saved to '{output_filename}'")
    except Exception as e:
         print(f"Error saving results to CSV: {e}")

else:
    print("\nClassification columns not found in DataFrame, skipping analysis and saving.")

print("\nScript finished.")