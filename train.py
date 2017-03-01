import data

training = data.Data(mode="img", batch_size=2)
inp, out, _  = training.get_batch()
training.print_boards(inp)
print out

training = data.Data(mode="seq", batch_size=2)
inp, out, lens  = training.get_batch()
print inp
print out
print lens
