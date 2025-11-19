import numpy as np
import os
import pandas as pd
import gc

LIST_OF_PLAYERS = ["Carlsen, Magnus",
                   "Cramling Bellon, Anna",
                   "Caruana, Fabiano",
                   "Nepomniachtchi, Ian",
                   "Firouzja, Alireza",
                   "Giri, Anish",
                   "Niemann, Hans",
                   "Cramling, Pia",
                   "Nakamura, Hikaru",
                   "Botez, Alexandra",
                   "Botez, Andrea",
                   "Belenkaya, Dina",
                   "So, Wesley",]



def filter_by_players(csvs_dir,list_of_players)->pd.DataFrame:
    new_df = pd.DataFrame()
    for filename in os.listdir(csvs_dir):
        print(f"processing {filename}")
        df = pd.read_csv(os.path.join(csvs_dir, filename))
        df = df[(df['white_name'].isin(list_of_players)) | (df['black_name'].isin(list_of_players))]  # filter by player
        new_df = pd.concat([new_df, df], ignore_index=True)
        print(new_df.shape)
    return(new_df)



csvs_dir = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\all_games_csvs\\"

new_df = filter_by_players(csvs_dir,LIST_OF_PLAYERS)

#test_df = pd.read_csv("C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\filtered_games.csv")

print(new_df.head())

print("Saving new dataframe")
outfile_path = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\filtered_games.csv"
new_df.to_csv(outfile_path,index=False)
print("Saved")
print("DONE")
