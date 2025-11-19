import numpy as np
import os
import pandas as pd
import gc

outfile_path = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\all_games.csv"


def moves_manipulation(pgn_string):
    """
    This function uses regular expressions to count the number of moves an at the same time create a list of all the moves, in order
    To create the regex expressions I used Microsoft Copilot, since it's a tedious job that is easy to mess up

    Returns:
        tuple (int, list): num_moves, list_of_moves
    """
    import re

    # Remove newlines
    pgn_string = pgn_string.replace("\n", " ")

    # Remove comments in curly braces { ... }
    pgn_string = re.sub(r"\{[^}]*\}", "", pgn_string)

    # Remove variations in parentheses ( ... )
    pgn_string = re.sub(r"\([^)]*\)", "", pgn_string)

    # Remove result at the end (e.g., 1-0, 0-1, 1/2-1/2)
    pgn_string = re.sub(r"\s*(1-0|0-1|1/2-1/2)\s*$", "", pgn_string)

    # Remove move numbers like "1.", "2.", etc.
    pgn_string = re.sub(r"\d+\.", "", pgn_string)

    # Remove annotations like ! or ?
    pgn_string = re.sub(r"[!?]+", "", pgn_string)

    # Split by spaces and filter empty strings
    list_of_moves = [move for move in pgn_string.split() if move]

    # Count moves
    num_moves = len(list_of_moves)

    return num_moves, list_of_moves

def find_a_metadata(lines,index,metadata_str):
    for i in range(30):
        if len(lines) <= index+i:
            return "NOT_FOUND"
        if "[White " in lines[index+i]:
            return "NOT_FOUND"
        elif metadata_str in lines[index+i]:
            return lines[index+i][len(metadata_str):-3]
    return "NOT_FOUND"

def filter_file_concurrent(filepath,file_number):
    saved = False
    game_number = 0
    chunk_number = 0
    rows = []
    print("Processing", filepath)
    with open(filepath, 'r', encoding="latin-1") as infile:
        lines = infile.readlines()
    if "OTB" in filepath:
        game_type = "OTB"
    elif "Online" in filepath:
        game_type = "ONLINE"
    else:
        raise Exception(f"Invalid game type for {filepath}")

    for index, line in enumerate(lines):

        if "[White " in line:
            saved = False
            game_number += 1
            if game_number % 10000 == 0:
                print(f"{game_number} games processed, file {file_number}")
            white_name = lines[index][len("[White \""):-3]
            if "[WhiteElo " in lines[index + 3]:
                white_elo = int(lines[index + 3][len("[WhiteElo \""):-3])
            else:
                white_elo = find_a_metadata(lines, index + 2, "[WhiteElo \"")
            if "[Black " in lines[index + 1]:
                black_name = lines[index + 1][len("[Black \""):-3]
            else:
                black_name = find_a_metadata(lines, index + 1, "[Black \"")
            if "[BlackElo " in lines[index + 4]:
                black_elo = lines[index + 4][len("[BlackElo \""):-3]
            else:
                black_elo = find_a_metadata(lines, index + 3, "[BlackElo \"")
            if "[Result " in lines[index + 2]:
                game_result = lines[index + 2][len("[Result \""):-3]
            else:
                game_result = find_a_metadata(lines, index + 1, "[Result \"")
            time_control = find_a_metadata(lines, index + 2, "[TimeControl \"")

            not_next_game = True
            start_moves = False
            this_line_index = index + 5
            next_line = lines[this_line_index]
            moves = ""
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
                        else:
                            not_next_game = False
                            continue
                    this_line_index += 1
                    next_line = lines[this_line_index]

            number_of_moves, list_of_moves = moves_manipulation(moves)

            new_row = {
                "game_number": f"{file_number}_{game_number}",
                "game_type": game_type,
                "white_name": white_name,
                "white_elo": white_elo,
                "black_name": black_name,
                "black_elo": black_elo,
                "game_result": game_result,
                "time_control": time_control,
                "moves": moves,
                "number_of_moves": number_of_moves,
                "list_of_moves": list_of_moves
            }
            rows.append(new_row)
            if game_number % 1000000 == 0 and game_number != 0 and not saved:
                saved = True
                chunk_number += 1
                print("SAVING")
                dataframe = pd.DataFrame(rows)
                dataframe.to_csv(f"C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\file_{file_number}_part{chunk_number}.csv",
                                 index=False)
                print("SAVED")
                del dataframe
                rows = []
                gc.collect()  # clear memory
        else:
            continue

    dataframe = pd.DataFrame(rows)
    if chunk_number != 0:
        chunk_number += 1
        print(f"SAVING file_{file_number}_part{chunk_number}")
        dataframe.to_csv(f"C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\file_{file_number}_part{chunk_number}.csv",
                         index=False)
    else:
        dataframe.to_csv(f"C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\file_{file_number}.csv",
                     index=False)
    print("SAVED")

    del dataframe
    rows.clear()
    gc.collect()  # clear memory
    return True
    #return dataframe

def filter_list_of_files(filepaths):
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
    dataframe = pd.DataFrame(columns=['game_number', 'game_type', 'white_name', 'black_name',
                                      'white_elo', 'black_elo', 'time_control', 'moves',
                                      'number_of_moves', 'list_of_moves', 'game_result'])

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

            if "[White " in line:
                game_number += 1
                if game_number%10000 == 0:
                    print(f"{game_number} games processed, file {file_number}")
                white_name = lines[index][len("[White \""):-3]
                if "[WhiteElo " in lines[index+3]:
                    white_elo = int(lines[index+3][len("[WhiteElo \""):-3])
                else:
                    white_elo = find_a_metadata(lines,index+2,"[WhiteElo \"")
                if "[Black " in lines[index+1]:
                    black_name = lines[index+1][len("[Black \""):-3]
                else:
                    black_name = find_a_metadata(lines, index, "[Black \"")
                if "[BlackElo " in lines[index+4]:
                    black_elo = lines[index+4][len("[BlackElo \""):-3]
                else:
                    black_elo = find_a_metadata(lines, index+3, "[BlackElo \"")
                if "[Result " in lines[index+2]:
                    game_result = lines[index+2][len("[Result \""):-3]
                else:
                    game_result = find_a_metadata(lines, index+1, "[Result \"")
                time_control = find_a_metadata(lines, index, "[Result \"")

                not_next_game = True
                start_moves = False
                this_line_index = index + 5
                next_line = lines[this_line_index]
                moves = ""
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
                            else:
                                not_next_game = False
                                continue
                        this_line_index += 1
                        next_line = lines[this_line_index]

                number_of_moves, list_of_moves = moves_manipulation(moves)

                new_row = {
                    "game_number": f"{file_number}_{game_number}",
                    "game_type": game_type,
                    "white_name": white_name,
                    "white_elo": white_elo,
                    "black_name": black_name,
                    "black_elo": black_elo,
                    "game_result": game_result,
                    "time_control": time_control,
                    "moves": moves,
                    "number_of_moves": number_of_moves,
                    "list_of_moves": list_of_moves
                }
                dataframe = pd.concat([dataframe, pd.DataFrame([new_row])], ignore_index=True)

            else:
                continue
        dataframe.to_csv(f"C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\file_{file_number}.csv", index=False)

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

allfilepaths = [path_online_95_09, path_online_10_14,path_online_15_19, path_online_2020,path_online_2021, path_online_2022,
                path_online_2023, path_online_2024,path_online_2025,path_otb_90_99,path_otb_00_04,path_otb_05_09,
                path_otb_10_14,path_otb_15_19,path_otb_20_24,path_otb_2025]

filter_file_concurrent(allfilepaths[15],15)
print("DONE")
#from concurrent.futures import ThreadPoolExecutor
#
#with ThreadPoolExecutor(max_workers=8) as executor:
#    results = [executor.submit(filter_file_concurrent, file,file_number) for file,file_number in zip(filepaths6,file_nums6)]
#
#for r in results:
#    if not r:
#        print("ERROR")
#    else:
#        print(r)
#final_df = pd.concat(dataframes, ignore_index=True)

#final_df.to_csv("C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\all_games.csv", index=False)