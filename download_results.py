import os
import glob

def download_all_results():
    """
    Helper function for Google Colab to download all benchmark results and plots.
    """
    try:
        from google.colab import files
    except ImportError:
        print("Error: This script is intended to be run inside a Google Colab environment.")
        return

    # Files to download
    patterns = [
        "benchmark_results.json",
        "*.png"
    ]
    
    files_to_download = []
    for pattern in patterns:
        files_to_download.extend(glob.glob(pattern))
    
    if not files_to_download:
        print("No result files found (json or png) in the current directory.")
        return

    print(f"Found {len(files_to_download)} files to download...")
    for f in sorted(files_to_download):
        print(f"Downloading: {f}")
        files.download(f)

if __name__ == "__main__":
    download_all_results()
