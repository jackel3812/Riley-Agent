import os
import shutil

def backup_file(file_path):
    if os.path.exists(file_path):
        shutil.copy(file_path, file_path + ".bak")
        print(f"Backup created: {file_path}.bak")

def replace_code(file_path, new_code):
    backup_file(file_path)
    with open(file_path, "w") as f:
        f.write(new_code)
    print(f"Updated: {file_path}")

def update_from_github(repo_url, file_path):
    import requests
    response = requests.get(repo_url)
    if response.status_code == 200:
        replace_code(file_path, response.text)
    else:
        print("Failed to fetch update.")

# Example usage:
# update_from_github("https://raw.githubusercontent.com/username/repo/main/app.py", "app.py")
