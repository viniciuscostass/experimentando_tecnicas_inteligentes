from random import seed
from random import randrange
from csv import reader
from statistics import median

def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def precision_metric(actual, predicted):
    correct_positives = 0
    positives = 0
    for i in range(len(actual)):
        if predicted[i] == 1:
            positives += 1
            if actual[i] == predicted[i]:
                correct_positives += 1
    if positives != 0:
        return correct_positives/positives
    return 0

def recall_metric(actual, predicted):
    correct_positives = 0
    real_positives = 0
    for i in range(len(actual)):
        if actual[i] == 1:
            real_positives += 1
            if actual[i] == predicted[i]:
                correct_positives += 1
    if real_positives != 0:
        return correct_positives/real_positives
    return 1.0

def f_measure_metric(precision, recall):
    if (precision + recall) != 0:
        return 2*(precision * recall) / (precision + recall)
    return 0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    global tree_fold, best_tree_fold, best_scores_fold
    max_f_measure = 0
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        (predicted, tree) = algorithm(train_set, test_set, *args)
        print_tree(tree, 0)
        # print("--------")
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        precision = precision_metric(actual, predicted)
        recall = recall_metric(actual, predicted)
        f_measure = f_measure_metric(precision, recall)
        if f_measure > max_f_measure:
            max_f_measure = f_measure
            best_tree_fold = tree_fold
            best_scores_fold = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f-measure': f_measure}
        tree_fold = ""
        scores.append({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f-measure': f_measure})
    return scores

def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))

    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p

        gini += (1.0 - score) * (size / n_instances)
    return gini

def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    if len(right)<= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

def print_tree(node, depth=0):
    global tree_fold, header
    if isinstance(node, dict):
        tree_fold += depth*' '+"["+header[node['index']]+" < " + "{0:.3f}".format(node['value'])+ "]\n"
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        answer = "TRUE" if node == 1 else "FALSE"
        tree_fold += depth*' '+"["+str(answer)+"]\n"

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions, tree)

files = ["normalized_DCL.csv"]

for f in files:
    print(f)
    filename = f
    max_mean_accuracy = 0
    max_median_accuracy = 0
    max_median_fmeasure = 0
    max_median_fmeasure = 0
    better_fold = None
    better_scores = []
    better_tree = []
    tree_fold = ""
    best_tree_fold = ""
    best_scores_fold = None

    dataset = load_csv(filename)

    for j in range(2,15):

        dataset = load_csv(filename)
        header = dataset[0]
        dataset = dataset[1:]
        for d in dataset:
            d[-1] = '0' if d[-1] == 'FALSE' else '1'
        for i in range(len(dataset[0])):
            str_column_to_float(dataset, i)

        seed(1)
        n_folds = j
        max_depth = 4
        min_size = 10
        sum_accuracy = 0
        sum_fmeasure = 0
        scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
        accuracies = []
        fmeasures = []
        for i in range(n_folds):
            accuracies.append(scores[i]['accuracy'])
            sum_accuracy += scores[i]['accuracy']
            fmeasures.append(scores[i]['f-measure'])
            sum_fmeasure += scores[i]['f-measure']
        mean_accuracy = sum_accuracy/float(len(scores))
        mean_fmeasure = sum_fmeasure/float(len(scores))
        if median(accuracies) > max_median_accuracy:
            max_mean_accuracy = mean_accuracy
            max_median_accuracy = median(accuracies)
            max_mean_fmeasure = mean_fmeasure
            max_median_fmeasure = median(fmeasures)
            better_fold = j
            better_scores = best_scores_fold
            better_tree = best_tree_fold
        dataset = None
    print("Folds: %d" % (better_fold))
    print('Mean Accuracy: %.3f%%' % (max_mean_accuracy))
    print('Median Accuracy: %.3f%%' % (max_median_accuracy))
    print('Mean fmeasure: %.3f%' % (max_mean_fmeasure))
    print('Median fmeasure: %.3f%' % (max_median_fmeasure))

    print(better_scores)
    print(better_tree)
    print()
