"""

Our data: string of the type:

"B 1. d4 g6 2. Bf4 Bg7 3. e3 c5 4. Nf3 Qb6 5. b3 $6 Nc6 6. c3 d6 7. Be2 cxd4 8. exd4
Nf6 9. O-O O-O 10. Nbd2 $6 Nd5 11. Nc4 $1 Qd8 12. Qd2 $2 b5 13. Nb2 Nxf4 14. Qxf4
b4 $6 15. Rac1 $6 bxc3 16. Rxc3 $9 Nb4 $9 17. a3 $4 Nd5 $1 18. Qc1 Nxc3 19. Qxc3 Bb7
20. Qd2 e5 21. Nc4 e4 22. Ne1 d5 23. Na5 Qb6 24. Nxb7 Qxb7 25. b4 a5 26. Nc2
axb4 27. axb4 Ra2 28. Rc1 Rc8 29. Bd1 Rc4 30. Ra1 Rxa1 31. Nxa1 Qxb4 32. Qxb4
Rxb4 33. Nb3 Bxd4 34. Kf1 Bb6 35. Ke2 d4 36. Nd2 f5 37. f3 d3+ 38. Ke1 e3 39.
Nf1 Kg7 40. g3 $6 Rb2 41. g4 Ba5+ 42. Nd2 Rxd2 43. Kf1 Rxd1+ 0-1"

Lenght not defined

Notice I added B as first char cuz we need to insert also the player we are classifing (in this case Hikaru that plays as Black)

So the general stucture is "(W/B) n*(n*(x. move|(move move)) n*($y)) (result)" where:
- (W/B) defines which player we are interested in classifing
- X. indicates the move number (increments after both players move)
- move is the move in Standard Algebraic Notation (read rules here https://en.wikipedia.org/wiki/Portable_Game_Notation) can be up to 7 char as far as i can tell
- result is the game result and can be 1-0 | 0-1 | 1/2-1/2 | * if the game is till going (does not matter for us)

*IMPORTANT* in theory there could be comments in the form of {This opening is called the Ruy Lopez.} or ;This opening is called the Ruy Lopez.

------------------------------------------------------------------------------------------------------------------------

How to use the data?? Maby just transformers + attention + maby somthing else

I think a RNN (recurrent neural net) is the best way since we have a undefinetly long series of moves (BUT I'm not sure)

then -> convolutional layers / linear layers / transformers + attention maby??

output layer -> softmax into N+1 output nodes where N is the number of players we want to be able to classify (number of classes)
(N+1 beause I think is a good idea to also have data of some other random players to teach the NN to recognise when a game is of NONE of those players)
for each player we want a similar amount ~M of games and then either ~M or more (but not to many more) form other random players (mixed of mixed levelse)
"""