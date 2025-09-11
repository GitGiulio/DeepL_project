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
    """
    with open(filepath, 'r', encoding="utf-8") as infile:
        lines = infile.readlines()

    keep = False
    for index,line in enumerate(lines):
        if keep:
            if line[0:7] == "[White ":
                keep = False
            else:
                if "[TimeControl" in line:  # I have literally no fukking clue of why this does not work
                    outfile.write(line)
                elif (line[0:7] == "[Black " or line[0:7] == "[White " or line[0:8] == "[Result " or
                        line[0:10] == "[WhiteElo " or line[0:10] == "[BlackElo " or line[0:7] == "[TimeC " or
                        line[0] != "["): # TODO actually non va bene perche le line delle mosse possono anche iniziare per [ se ti va di sfiga
                    outfile.write(line)
                else:
                    continue
        if not keep:  # not else because I am changing keep in the if
            for player_name in list_of_players:
                if player_name in line:
                    if line[0:7] == "[Black ":
                        outfile.write(lines[index-1])
                    outfile.write(line)
                    keep = True
                else:
                    continue

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


