import kagglehub

# Download latest version
path = kagglehub.dataset_download("anairamcosta/titanic-test-csv")

print("Path to dataset files:", path)