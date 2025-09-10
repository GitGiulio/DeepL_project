import numpy as np
#import matplotlib.pyplot as plt
import sklearn
import os

def filter_lines_by_matches(filename, list_of_players):
    with open(filename, 'r') as infile:
        lines = infile.readlines()

    outfile_path = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\output.txt"
    outfile = open(outfile_path, 'w')

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
                        line[0] != "["):
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

    outfile.close()

path = "C:\\Users\\giuli\\PycharmProjects\\DeepL_project_test\\data\\LumbrasGigaBase_Online_2024.pgn"

LIST_OF_PLAYERS = ["\"Carlsen, Magnus\"",
                   "\"Cramling Bellon, Anna\"",
                   "\"Caruana, Fabiano\"",
                   "\"Nepomniachtchi, Ian\"",
                   "\"Firouzja, Alireza\"",
                   "\"Giri, Anish\"",
                   "\"Niemann, Hans\""]

filter_lines_by_matches(path, LIST_OF_PLAYERS)
