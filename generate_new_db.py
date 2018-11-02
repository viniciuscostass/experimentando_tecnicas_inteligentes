from csv import reader

def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset

def normalizate(dataSet):
    for col in range(len(dataSet[0]) - 2):
        max_col = find_max(dataSet, col)
        min_col = find_min(dataSet, col)
        for instance in dataSet:
            instance[col] = float(instance[col])
            if (max_col - min_col) == 0:
                instance[col] = 0
            else:
                instance[col] = (instance[col] - min_col) / (max_col - min_col)
            print(instance[col])
    return dataSet

# This function finds the maximum value for a given attribute
def find_max(dataSet, col):
    max_col = -100000
    for instance in dataSet:
        instance[col] = float(instance[col])
        if instance[col] > max_col:
            max_col = instance[col]
    return max_col

# This function finds the minimum value for a given attribute
def find_min(dataSet, col):
    min_col = 100000
    for instance in dataSet:
        instance[col] = float(instance[col])
        if instance[col] < min_col:
            min_col = instance[col]
    return min_col




data = load_csv("DCL.csv")

header = data[0]
data.remove(data[0])
normalizate(data)
print(data)

fh = open("normalized_DCL.csv","w")
fh.writelines(",".join(header))
fh.writelines("\n")
for i in range(len(data)):
    linhas_de_texto = [ str(x) for x in data[i] ]
    fh.writelines(",".join(linhas_de_texto))
    fh.writelines("\n")

fh.close()
