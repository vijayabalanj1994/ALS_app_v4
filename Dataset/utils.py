import os
import pandas as pd

def get_metadata(image_path):

    filename = os.path.basename(image_path)
    image_no = int(os.path.splitext(filename)[0])

    df = pd.read_excel(r"C:\Users\vijay\Neuro_BioMark\ALS_app_v2\Dataset\image_keys.xlsx", header=1)

    result = df[df["Image No"] == image_no][["Case ID", "Region"]]

    if not result.empty:
        case_id = result.iloc[0]["Case ID"]
        region = result.iloc[0]["Region"]
        return str(image_no), case_id, region
    else:
        return str(image_no), "UNKNOWN", "UNKNOWN"
