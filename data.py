import sys
import random
import numpy as np
np.set_printoptions(threshold=np.nan)

class Data(object):
  def __init__(self, batch_size=32, max_length=40, vocab_size=20,
               rep_dim=32, mode="seq"):
    self.batch_size = batch_size
    self.max_length = max_length
    self.vocab_size = vocab_size
    self.mode = mode

    # For Tic-Tac-Toe
    self.wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]

  def get_n_batchs(self, n=1):
    """
    For use with Multi-GPU training
    """
    if n == 1:
      return self.get_batch()
    else:
      inps = []
      outs = []
      lens = []
      for i in range(n):
        i, o, l = self.get_batch()
        inps.append(i)
        outs.append(o)
        lens.append(l)
    return inps, outs, lens

  def get_batch(self):
    if self.mode == "seq":
      return self.get_seq_batch()
    elif self.mode == "img":
      return self.get_img_batch()
    else:
      print "Houston, we have a problem"
      sys.exit()

  def who_won(self, board):
    """
      0 - Nobody
      1 - X
      2 - O
      3 - Both
    """
    winners = set()
    for x,y,z in self.wins:
      if board[x] == board[y] and board[y] == board[z]:
        winners.add(board[x])
    if 1 in winners and 2 in winners:
      return 3
    if 1 in winners:
      return 1
    if 2 in winners:
      return 2
    return 0

  def generate_random_board(self):
    while True:
      turns = np.random.randint(4, 9)  
      places = np.zeros((9))
      actions = random.sample(range(9), turns)
      for i in range(len(actions)):
        places[actions[i]] = i % 2 + 1
      who = self.who_won(places)
      if who == 3:
        continue
      label = who
      places = np.array([[int(i == v) for i in range(3)] for v in places])
      places = np.reshape(places, (3,3,3))
      places = places.repeat(10, axis=0).repeat(10, axis=1)    # Now a 30/30/3
      return places, label

  def get_img_batch(self):
    inps = []
    outs = []
    for i in range(self.batch_size):
      board, label = self.generate_random_board()
      inps.append(board)
      outs.append(label)
    return np.array(inps), np.array(outs), None

  def print_boards(self, boards):
    for board in boards:
      b = ""
      for row in board:
        for val in np.argmax(row, axis=1):
          b += "_" if val == 0 else "X" if val == 1 else "o"
        b += "\n"
      print b

  def get_seq_batch(self):
    """
    Generate a batch for training with language data
    """
    inps = []
    outs = []
    lens = []
    for i in range(self.batch_size):
      leng = np.random.randint(1, self.max_length)
      sent = np.random.choice(range(self.vocab_size), leng)
      labl = sent % 2

      lens.append(leng)
      inps.append(np.pad(sent, (0, self.max_length - leng), 'constant', constant_values=0))
      outs.append(np.pad(labl, (0, self.max_length - leng), 'constant', constant_values=0))
    return np.array(inps), np.array(outs), np.array(lens)
