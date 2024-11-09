# NNUE_engine

### Gonna update this later. 
I was bored and decided to extend a previous project to using NNUE instead of base generic algorithms like MiniMax, alpha-beta pruning, Monte-Carlo and Quiescence search. 

However I intend for this NNUE to use Quiescence search, book-moving and a couple of other retrieval methods. 

We'll see how this goes... 

### Update on this: 
- Instead of using Stockfish engine to generate the datasets, I decided to switch to scrapped FEN datasets from Lichess and Chess.com. The dataset contains over 19 million chess positions. Link to the dataset will be added later.
- But for now, I'm only using one of the three datasets, focusing on tactics alone. 
#### In-Progress
  - Worked on the first important part: I implemented a position evaluator already, so all it does is assign points to a position (+ve if white is winning and -ve if black is winning). Aha, seems like I made it sound so simple

#### Todo: 
  - At a later time, build an AI agent capable of playing a full game.  
