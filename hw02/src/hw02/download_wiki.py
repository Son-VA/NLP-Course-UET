from huggingface_hub import hf_hub_download

for n in range(4):
    name = f"20231101.vi/train-0000{n}-of-00004.parquet"
    path = hf_hub_download("wikimedia/wikipedia", name, repo_type="dataset")
    print("done with: ", path)