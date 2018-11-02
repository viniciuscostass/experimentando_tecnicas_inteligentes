from csv import reader
import numpy
import scipy.stats
import random

def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset



dataset = load_csv('normalized_DCL.csv')

list_with_bugs = []
list_without_bugs = []
header = dataset[0]
dataset.remove(dataset[0])
for developer in dataset:
  if developer[-1] == 'FALSE':
    developer[-1] = 0
    list_without_bugs.append(developer)
  else:
    developer[-1] = 1
    list_with_bugs.append(developer)

diff = len(list_without_bugs) - len(list_with_bugs)
if diff > 0:
    for i in range(diff):
        new_array = []
        for j in range(len(dataset[0])):
            new_array.append(str(random.uniform(0,1)))
        list_with_bugs.append(new_array)
else:
    for i in range(-diff):
        new_array = []
        for j in range(len(dataset[0])):
            new_array.append(str(random.uniform(0, 1)))
        list_without_bugs.append(new_array)
for i in range(len(dataset[0])-1):
  array_true = numpy.array(list_with_bugs, dtype=float)[:, i]
  array_false = numpy.array(list_without_bugs, dtype=float)[:, i]

  wilcoxon = scipy.stats.wilcoxon(array_true, array_false)
  print(header[i]+" -> "+str(wilcoxon))
