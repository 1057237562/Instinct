from datasets import load_dataset

def format_prompt(x):
    return {"prompt": x["instruction"].strip() + "\n" + x["input"].strip(), "completion": x["output"].strip()}

def main():
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    ds = ds.map(format_prompt, num_proc=8, remove_columns=ds.column_names)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    ds.push_to_hub("HuggingFaceH4/CodeAlpaca_20K")

if __name__ == "__main__":
    main()