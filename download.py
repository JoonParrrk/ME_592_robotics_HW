import urllib.request
import sys

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
filename = "sam_vit_b_01ec64.pth"

def progress_bar(count, block_size, total_size):
    # Calculate and print the percentage
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write(f"\rDownloading {filename}... {percent}%")
    sys.stdout.flush()

print("Connecting to Meta's servers...")
urllib.request.urlretrieve(url, filename, reporthook=progress_bar)
print("\nDownload completely successfully!")