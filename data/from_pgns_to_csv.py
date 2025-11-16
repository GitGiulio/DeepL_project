import numpy as np
import os
import pandas as pd

outfile_path = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\all_games.csv"



def filter_lines_by_matches(filepaths):
    """
    Col 1: game_number,
    Col 2: game_type (OTB/online),
    Col 3: white_name,
    Col 4: black_name,
    Col 5: white_elo,
    Col 6: black_elo,
    Col 7: time_control,
    Col 8: moves,
    Col 9: number_of_moves,
    Col 10: list_of_moves,
    """
    game_number = 0
    dataframe = pd.DataFrame(columns=['game_number', 'game_type', 'white_name', 'black_name', 'white_elo',
                                      'black_elo', 'time_control', 'moves', 'number_of_moves', 'game_result'])

    for file_number,filepath in enumerate(filepaths):
        print("Processing", filepath)
        with open(filepath, 'r', encoding="utf-8") as infile:
            lines = infile.readlines()
        if "OTB" in filepath:
            game_type = "OTB"
        elif "Online" in filepath:
            game_type = "ONLINE"
        else:
            raise Exception(f"Invalid game type for {filepath}")

        for index,line in enumerate(lines):

            # TODO process the file and create the following variables
            white_name = ""
            black_name = ""
            white_elo = 0
            black_elo = 0
            game_result = ""
            time_control = ""
            list_of_moves = []
            number_of_moves = 0
            moves = ""

            if "[White " in line:
                game_number += 1
                if game_number%1 == 0:
                    print(f"{game_number} games processed, file {file_number}")
                dataframe.loc[game_number, "game_number"] = game_number
                dataframe[game_number,"game_type"] = game_type
                white_name = lines[index][11:-3]  # TODO check index
                if "[WhiteElo " in lines[index+3]:
                    white_elo = int(lines[index+3][11:-3])  # TODO check index
                else:
                    pass
                    #TODO implement for all a seach from index to index+10 or somthing (with a fun where you pass the str)
                if "[Black " in lines[index+1]:
                    black_name = lines[index][11:-3]  # TODO check index
                else:
                    pass
                if "[BlackElo " in lines[index+4 + 1]:
                    black_elo = lines[index+4][11:-3]  # TODO check index
                else:
                    pass
                if "[Result " in lines[index+4 + 1]:
                    game_result = lines[index+1][11:-3]  # TODO check index
                else:
                    pass
                if "[Time " in lines[index+4 + 1]:
                    time_control = lines[index+1][11:-3]  # TODO check index
                else:
                    pass
                #dataframe[game_number,"time_control"] = lines[index+4][x:-3]  # TODO fix (also if is there always)

                not_next_game = True
                start_moves = False
                this_line_index = index + 5
                next_line = lines[this_line_index]
                while not_next_game:
                    if "[White " in next_line:
                        not_next_game = False
                    else:
                        if not start_moves:
                            if "\n" == next_line:
                                start_moves = True
                        else:
                            if "\n" != next_line:
                                moves = moves + next_line
                        this_line_index += 1
                        next_line = lines[this_line_index]
                dataframe[game_number, "white_name"] = white_name
                dataframe[game_number, "white_elo"] = white_elo
                dataframe[game_number,"black_name"] = black_name
                dataframe[game_number,"black_elo"] = black_elo
                dataframe[game_number, "game_result"] = game_result
                dataframe[game_number, "time_control"] = time_control
                dataframe[game_number,"moves"] = moves  # TODO check

            # todo count the moves and do
            # dataframe[game_number,"number_of_moves"] = number_of_moves
            else:
                continue

    return dataframe


path_online_95_09 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_Online_1995_2009.pgn"
path_online_10_14 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_Online_2010_2014.pgn"
path_online_15_19 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_Online_2015_2019.pgn"
path_online_2020 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_Online_2020.pgn"
path_online_2021 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_Online_2021.pgn"
path_online_2022 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_Online_2022.pgn"
path_online_2023 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_Online_2023.pgn"
path_online_2024 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_Online_2024.pgn"
path_online_2025 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_Online_2025.pgn"
path_otb_90_99 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_OTB_1990_1999.pgn"
path_otb_00_04 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_OTB_2000_2004.pgn"
path_otb_05_09 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_OTB_2005_2009.pgn"
path_otb_10_14 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_OTB_2010_2014.pgn"
path_otb_15_19 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_OTB_2015_2019.pgn"
path_otb_20_24 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_OTB_2020_2024.pgn"
path_otb_2025 = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\original_PGNs\\LumbrasGigaBase_OTB_2025.pgn"

filepaths = [path_online_95_09, path_online_10_14, path_online_15_19, path_online_2020, path_online_2021,
             path_online_2022, path_online_2023, path_online_2024, path_online_2025, path_otb_90_99, path_otb_00_04,
             path_otb_05_09, path_otb_10_14, path_otb_15_19, path_otb_20_24, path_otb_2025,]


df = filter_lines_by_matches(filepaths)

df.to_csv("C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\all_games.csv")