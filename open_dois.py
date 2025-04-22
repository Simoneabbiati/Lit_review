# Filename: doi_resolver.py

import requests
import csv
import os
import time
from requests.exceptions import RequestException, HTTPError, Timeout, ConnectionError

# --- Configuration ---
DOI_FILENAME = "dois.txt"        # Input file with one DOI per line
OUTPUT_FILENAME = "resolved_dois.csv" # Output file for results
REQUEST_TIMEOUT = 15             # Seconds to wait for a server response
USER_AGENT = "DOIResolverScript/1.0 (Python Requests; +https://github.com/)" # Be polite to servers
DELAY_BETWEEN_REQUESTS = 0.2     # Seconds to wait between requests (optional, be kind to doi.org)
# -------------------

def resolve_dois(input_filename, output_filename):
    """
    Reads DOIs from input_filename, resolves them to their final URLs,
    and saves the results to output_filename (CSV format).
    """
    dois_to_resolve = []
    results = [] # List to store tuples (doi, final_url, status)

    # --- 1. Read DOIs from the input file ---
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        print(f"Please create '{input_filename}' with one DOI per line.")
        return

    try:
        with open(input_filename, 'r') as f:
            for line in f:
                doi = line.strip()
                if doi: # Only process non-empty lines
                    dois_to_resolve.append(doi)
    except Exception as e:
        print(f"Error reading file '{input_filename}': {e}")
        return

    if not dois_to_resolve:
        print(f"No valid DOIs found in '{input_filename}'.")
        return

    total_dois = len(dois_to_resolve)
    print(f"Found {total_dois} DOIs in '{input_filename}'. Starting resolution...")

    # --- 2. Resolve each DOI ---
    headers = {'User-Agent': USER_AGENT} # Identify our script

    for i, doi in enumerate(dois_to_resolve):
        doi_url = f"https://doi.org/{doi}"
        print(f"Resolving ({i+1}/{total_dois}): {doi} -> ", end="", flush=True) # Print progress without newline yet

        final_url = None
        status = "Unknown Error" # Default status

        try:
            # Make the request, following redirects automatically
            response = requests.get(
                doi_url,
                allow_redirects=True, # Follow redirects (this is default but explicit)
                timeout=REQUEST_TIMEOUT,
                headers=headers
            )

            # Check for HTTP errors (like 404 Not Found, 500 Server Error)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            # If successful, the final URL is in response.url
            final_url = response.url
            status = "Success"
            print(f"Success -> {final_url}")

        except Timeout:
            status = "Error: Timeout"
            print(f"Error: Request timed out after {REQUEST_TIMEOUT} seconds.")
        except ConnectionError:
            status = "Error: Connection Failed"
            print("Error: Could not connect to the server.")
        except HTTPError as e:
            status = f"Error: HTTP {e.response.status_code}"
            print(f"Error: HTTP {e.response.status_code} for {doi_url}")
        except RequestException as e:
            # Catch other potential requests errors
            status = f"Error: Request Failed ({type(e).__name__})"
            print(f"Error: Could not resolve {doi_url} - {e}")
        except Exception as e:
            # Catch any other unexpected errors
             status = f"Error: Unexpected ({type(e).__name__})"
             print(f"An unexpected error occurred: {e}")


        results.append({'doi': doi, 'final_url': final_url, 'status': status})

        # Optional delay between requests
        if DELAY_BETWEEN_REQUESTS > 0 and i < total_dois - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # --- 3. Write results to CSV file ---
    print(f"\nWriting results to '{output_filename}'...")
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Define the columns - using the keys from the first result dictionary if available
            # or defining them explicitly if results list is empty
            if results:
                fieldnames = results[0].keys()
            else:
                fieldnames = ['doi', 'final_url', 'status'] # Default headers if no results

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader() # Write the header row (DOI, Final_URL, Status)
            writer.writerows(results) # Write all the data rows

        print("Finished writing results.")

    except Exception as e:
        print(f"Error writing to output file '{output_filename}': {e}")


# --- Main execution ---
if __name__ == "__main__":
    resolve_dois(DOI_FILENAME, OUTPUT_FILENAME)
    print("\nScript finished.")
    # Optional: Keep the console window open until user presses Enter
    # input("Press Enter to exit...")