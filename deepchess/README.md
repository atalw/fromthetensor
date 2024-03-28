### Deepchess

Implementation of [https://arxiv.org/pdf/1711.09667.pdf](https://arxiv.org/pdf/1711.09667.pdf).

The basic idea is to create a pos2vec autoencoder whose encoded output feeds into a deepchess neural network for supervised training. 2 (win,lose) or (lose, win) positions are given as input and the model has to predict the which positions wins and which one loses. This model is then distilled down to a simpler network for speed. The distilled network is used in a novel alpha-beta minimax algorithm that compares positions instead of giving scores to positions.

Training on a smaller dataset for 500k positions works and I achieved a 90% validation accuracy. This however did not translate to good gameplay. Higher than 1 million positions doesn't train and causes a GPU hang on my system.

#### How to run

- Run `data.py` to generate datasets.
- `train_pos2vec.py` trains the first autoencoder network.
- `train_deepchess.py` trains the deepchess network.
- `train_distiller.py` trains the final distilled network.
- Play a game by running `game.py`.
