import pandas as pd
import os


for name in os.listdir("test"):
    df = pd.read_csv(f"test/{name}")

    for root in ["valid", "train"]:
        if os.path.exists(f"{root}/{name}"):
            new_df =  pd.read_csv(f"{root}/{name}")
            df = pd.concat([df, new_df], ignore_index=True)

    n = len(df)
    n_test = int(n * 0.2)

    if name == "Irf.csv":
        if n_test % 2 != 0:
            n_test += 1

    if name == "MultiWD.csv":
        while n_test %  6 != 0:
            n_test+=1

    df_test =  df.iloc[:n_test]
    df_val = df.iloc[n_test:n_test*2]
    df_train = df.iloc[n_test*2:]


    df_test.to_csv(f"test/{name}", index=False)
    df_val.to_csv(f"valid/{name}", index=False)
    df_train.to_csv(f"train/{name}", index=False)
