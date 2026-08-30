from mlcroissant import Dataset

ds = Dataset(jsonld="https://huggingface.co/api/datasets/roskoN/dailydialog/croissant")
records = ds.records("full")