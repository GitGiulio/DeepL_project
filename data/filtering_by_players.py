import numpy as np
import os
import pandas as pd
import gc

old_LIST_OF_PLAYERS = ["Carlsen, Magnus",
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

LIST_OF_PLAYERS = ["Carlsen, Magnus",
                   "Flamerare",
                   "Caruana, Fabiano",
                   "Nepomniachtchi, Ian",
                   "RaspFish",
                   "Giri, Anish",
                   "YoBot_v2",
                   "Nadigraj",
                   "Nakamura, Hikaru",
                   "ToromBot",
                   "ArasanX",
                   "Nikitosikbot",
                   "ResoluteBot",
                   "MassterofMayhem",
                   "JelenaZ",
                   "caissa-x",
                   "lestri",
                   "Nikitosik-ai",
                   "doreality",
                   "therealYardbird",]

def find_most_common_players(csvs_dir)->pd.DataFrame:
    global_counts = pd.DataFrame()
    list = []
    for filename in os.listdir(csvs_dir):
        print(f"processing {filename}")
        df = pd.read_csv(os.path.join(csvs_dir, filename))
        X = 2400
        if df.iloc[0]['game_type'] == 'OTB':
            X = 2000
        else:
            X = 2400

        df['white_elo'] = pd.to_numeric(df['white_elo'], errors='coerce')
        df['black_elo'] = pd.to_numeric(df['black_elo'], errors='coerce')

        white_name_filtered = df.loc[df['white_elo'] > X, 'white_name']
        black_name_filtered = df.loc[df['black_elo'] > X, 'black_name']

        combined_filtered = pd.concat([white_name_filtered, black_name_filtered])

        value_counts = combined_filtered.value_counts()

        top_100 = value_counts.head(100)
        list.append(top_100)

    global_counts = pd.concat(list, axis=1).fillna(0).sum(axis=1).astype(int)
    global_counts = global_counts.sort_values(ascending=False).head(20)

    return global_counts

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

#most_common_players = find_most_common_players(csvs_dir)

#print(most_common_players)

new_df = filter_by_players(csvs_dir,LIST_OF_PLAYERS)

#test_df = pd.read_csv("C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\filtered_games.csv")

print(new_df.head())

print("Saving new dataframe")
outfile_path = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\filtered_games_new.csv"
new_df.to_csv(outfile_path,index=False)
print("Saved")
print("DONE")
