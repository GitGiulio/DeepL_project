import numpy as np
#import matplotlib.pyplot as plt
import sklearn
import os
import pandas as pd

# outfile_path = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\output_otb.txt"
outfile_path = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\model_train_input.csv"



def filter_lines_by_matches(filepath, list_of_players):
    # TODO parsing file e trasforma in un csv della forma:
    """
    Col 1(input): "(W/B) n*(n*(x. move|(move move)) n*($y)) (result)",
    Col 2(true_value): "Player name (class_number)"

    --I can also add additional columns with other data like--
    Col 3(PlayerElo): "elo" <- giving those to the model would be kinda cheating
    Col 4(Opponent): "Player name"
    Col 5(OpponentElo): "elo" <- giving those to the model would be kinda cheatingù

    Col 7(NumberOfMoves): "number of moves" <-- this may be useful for feeding the model in the proper way

    Col 6(TimeControl): "time control" <--- could be a good idea to also give this to the model as input
    """
    with open(filepath, 'r', encoding="utf-8") as infile:
        lines = infile.readlines()

    dataframe = pd.DataFrame()

    dataframe["input"] = ""
    dataframe["TrueValue"] = ""
    dataframe["PlayerElo"] = 0
    dataframe["Opponent"] = ""
    dataframe["OpponentElo"] = 0
    dataframe["NumberOfMoves"] = 0
    dataframe["TimeControl"] = ""

    game_number = 0
    for index,line in enumerate(lines):
        for player_name in list_of_players:
            if player_name in line:
                input_string = ""
                if line[0:7] == "[Black ":
                    input_string = input_string + "B"
                    if player_name[1:-1] == lines[index][8:-3]:
                        print("CheckBlack")
                    dataframe.loc[game_number, "TrueValue"] = player_name[1:-1]
                    dataframe[game_number,"Opponent"] = lines[index-1][8:-3]
                    dataframe[game_number,"PlayerElo"] = int( lines[index+3][11:-3] )
                    dataframe[game_number,"OpponentElo"] = int( lines[index+2][11:-3] )
                    #dataframe[game_number,"TimeControl"] = lines[index+4][x:-3]

                elif line[0:7] == "[White ":
                    input_string = input_string + "W"
                    if player_name[1:-1] == lines[index][8:-3]:
                        print("CheckWhite")

                    dataframe.loc[game_number, "TrueValue"] = player_name[1:-1]
                    dataframe[game_number,"Opponent"] = lines[index + 1][8:-3]
                    dataframe[game_number,"PlayerElo"] = int(lines[index + 3][11:-3])
                    dataframe[game_number,"OpponentElo"] = int(lines[index + 4][11:-3])
                    # dataframe[game_number,"TimeControl"] = lines[index+5][x:-3]
                else:
                    print("ERROR")
                while False:
                    # TODO parsare le mosse per scriverle tutte unite senza cose aggiuntive
                    # TODO forse posso eliminare i commenti qui se non li voglio
                    pass
                dataframe.loc[game_number, "input"] = input_string
                game_number = game_number + 1
            else:
                continue

    return dataframe

outfile_path = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\output.txt"

LIST_OF_PLAYERS = ["\"Carlsen, Magnus\"",
                   "\"Cramling Bellon, Anna\"",
                   "\"Caruana, Fabiano\"",
                   "\"Nepomniachtchi, Ian\"",
                   "\"Firouzja, Alireza\"",
                   "\"Giri, Anish\"",
                   "\"Niemann, Hans\"",
                   "\"Cramling, Pia\"",
                   "\"Nakamura, Hikaru\"",
                   "\"Botez, Alexandra\"",
                   "\"Botez, Andrea\"",
                   "\"Belenkaya, Dina\"",
                   "\"So, Wesley\"",]

crate_pandas_dataframe(outfile_path, LIST_OF_PLAYERS)


