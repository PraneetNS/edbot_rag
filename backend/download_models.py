import os
import urllib.request
import sys

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    temp_dest = dest_path + ".tmp"
    try:
        def report_hook(block_num, block_size, total_size):
            read_so_far = block_num * block_size
            if total_size > 0:
                percent = min(100.0, read_so_far * 100.0 / total_size)
                sys.stdout.write(f"\rProgress: {percent:.1f}% ({read_so_far / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)")
            else:
                sys.stdout.write(f"\rProgress: {read_so_far / (1024*1024):.1f}MB")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, temp_dest, reporthook=report_hook)
        print() # Newline
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_dest, dest_path)
        print(f"Successfully downloaded and saved {dest_path}")
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        if os.path.exists(temp_dest):
            try:
                os.remove(temp_dest)
            except Exception:
                pass
        raise e

def main():
    # Make sure we target the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    files = {
        "kokoro-v1_0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "voices-v1_0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
    }

    os.makedirs(backend_dir, exist_ok=True)
    
    for filename, url in files.items():
        dest = os.path.join(backend_dir, filename)
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"[{filename}] already exists ({size_mb:.1f} MB), skipping.")
        else:
            print(f"[{filename}] is missing. Starting download...")
            try:
                download_file(url, dest)
            except Exception:
                print(f"Failed to download {filename}. Please try again later.")
                sys.exit(1)

if __name__ == "__main__":
    main()
