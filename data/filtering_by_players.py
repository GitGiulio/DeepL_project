import numpy as np
import os
import pandas as pd
import gc

#csvs_dir = "C:\\Users\\mathi\\Documents\\University\\Aarhus University\\MSc Computer Engineering\\Semester 1\\Deep Learning\\DATA_PROJECT"
csvs_dir = r"C:\Users\giuli\PycharmProjects\DeepL_project_test\data\all_games_csvs"
outfile_path = "C:\\Users\\mathi\\Documents\\University\\Aarhus University\\MSc Computer Engineering\\Semester 1\\Deep Learning\\project\\DeepL_project\\filtered_games_nobots.csv"

FIDE_RECOGNIZED_TITLES = ['IM', 'FM', 'CM', 'GM', 'NM', 'FIDE']
BOT_TITLE = 'bot'
OTHER_REAL_PLAYER_TITLE = 'unknown_real'
RELEVANT_LINES = 80  # change accordingly to what you manually added in file

INCLUDE_NON_FIDE_PLAYERS = True

PLAYER_CAP = 25

MINIMUM_OTB_ELO = 2000
MINIMUM_ONLINE_ELO = 2400

# Dont use anymore, unless new data is downloaded!
def find_most_common_players(csvs_dir)->pd.DataFrame:
    global_counts = pd.DataFrame()
    list = []
    for filename in os.listdir(csvs_dir):
        print(f"Processing {filename}")
        df = pd.read_csv(os.path.join(csvs_dir, filename))
        X = MINIMUM_ONLINE_ELO
        if df.iloc[0]['game_type'] == 'OTB':
            X = MINIMUM_OTB_ELO

        df['white_elo'] = pd.to_numeric(df['white_elo'], errors='coerce')
        df['black_elo'] = pd.to_numeric(df['black_elo'], errors='coerce')

        white_name_filtered = df.loc[df['white_elo'] > X, 'white_name']
        black_name_filtered = df.loc[df['black_elo'] > X, 'black_name']

        combined_filtered = pd.concat([white_name_filtered, black_name_filtered])

        value_counts = combined_filtered.value_counts()

        list.append(value_counts)

    global_counts = pd.concat(list, axis=1).fillna(0).sum(axis=1).astype(int)
    global_counts = global_counts.sort_values(ascending=False)

    return global_counts.to_string()  # purely used to print txt file for manual sorting.

# Filter players based on manually changed 'filtered_player_counts_output.txt'
def filter_by_players(csvs_dir,list_of_players)->pd.DataFrame:
    new_df = pd.DataFrame()
    for filename in os.listdir(csvs_dir):
        print(f"Processing {filename}")
        df = pd.read_csv(os.path.join(csvs_dir, filename))
        df = df[(df['white_name'].isin(list_of_players)) | (df['black_name'].isin(list_of_players))]  # filter by player
        new_df = pd.concat([new_df, df], ignore_index=True)
        print(new_df.shape)
    return(new_df)

def create_list_of_players(txtFile) -> list:
    with open(txtFile) as f:
        lines = f.readlines()
        
        list_of_players = []
        
        for i, f in enumerate(lines):
            if i < RELEVANT_LINES:
                splitted = f.strip().split()
                player_title = splitted[2]
                
                if player_title != BOT_TITLE:
                    if INCLUDE_NON_FIDE_PLAYERS:
                        list_of_players.append(splitted[0])
                    else:
                        if player_title != OTHER_REAL_PLAYER_TITLE:
                            list_of_players.append(splitted[0])
    
    return list_of_players[:PLAYER_CAP]

#most_common_players = find_most_common_players(csvs_dir)
#print(most_common_players)

#players = create_list_of_players('C:\\Users\\mathi\\Documents\\University\\Aarhus University\\MSc Computer Engineering\\Semester 1\\Deep Learning\\project\\DeepL_project\\data\\filtered_player_counts_output.txt')
#print(players)

'''
['ArasanX', 'MassterofMayhem', 'JelenaZ', 'lestri', 'doreality', 'therealYardbird', 'Chesssknock', 
'No_signs_of_V', 'Recobachess', 'drawingchest', 'kasparik_garik', 'ChainsOfFantasia', 
'Alexandr_KhleBovich', 'unknown-maestro_2450', 'gefuehlter_FM', 'gmmitkov', 'positionaloldman', 
'Consent_to_treatment', 'Gyalog75', 'chargemax23', 'Boreminator', 'sotirakis', 'cn_ua', 'anhao', 
'manuel-abarca', 'Chess_diviner', 'Toro123', 'Odirovski', 'manneredmonkey', 'Viktor_Solovyov', 
'Stas-2444', 'Zhigalko,', 'vistagausta', 'Romsta', 'Aborigen100500', 'JoeAssaad', 'bodoque50', 
'doreality1991', 'Niper13', 'Violet_Pride', 'Ivanoblitz', 'Atalik,', 'iakov98', 'AlexD64', 
'satlan', 'Bakayoyo', 'athena-pallada', 'Pblu35', 'okriak', 'morus22', 'Corre_por_tu_vida', 
'Attila76', 'Karlos_ulsk', 'www68', 'Podrebo', 'papasi', 'crackcubano', 'Chessibague']
'''

NEW_LIST_OF_PLAYERS_MANUAL = ['ArasanX', 'MassterofMayhem', 'JelenaZ', 'lestri', 'doreality', 'therealYardbird', 'Chesssknock',
'No_signs_of_V', 'Recobachess', 'drawingchest', 'kasparik_garik', 'ChainsOfFantasia','Consent_to_treatment',
'Alexandr_KhleBovich', 'unknown-maestro_2450', 'gefuehlter_FM', 'gmmitkov', 'positionaloldman',"Carlsen, Magnus","Nakamura, Hikaru"]


new_df = filter_by_players(csvs_dir,NEW_LIST_OF_PLAYERS_MANUAL)

print(new_df.head())

print("Saving new dataframe")
new_df.to_csv(r"C:\Users\giuli\PycharmProjects\DeepL_project_test\data\filtered_games_new.csv",index=False)
print("Saved")
