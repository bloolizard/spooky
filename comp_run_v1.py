import numpy
import pandas as pd

# Sample Submission
# id, EAP, HPL, MWS
def create_output_csv():
    columns = ["id", "EAP", "HPL", "MWS"]
    df = pd.DataFrame(columns=columns)
    df.loc[1] = [1,95,2,3]
    df.to_csv("output.csv", index=False)

create_output_csv()