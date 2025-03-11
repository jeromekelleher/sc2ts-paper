import sc2ts
import pandas as pd


def run():

    # ds = sc2ts.Dataset("data/viridian_mafft_2024-10-14_v1.vcz.zip", date_field="Date_tree")
    recomb_df = pd.read_csv("data/recombinants.csv")
    print(recomb_df)




if __name__ == "__main__":
    run()

