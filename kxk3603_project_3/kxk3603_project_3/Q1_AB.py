import math
import numpy as np


#Clear data from '(' ')'  characters
def clear_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


# Read data fram a given file
def get_data(filename):
    with open(filename, 'r') as f:
        input_data = f.readlines()
        clear_input = list(map(clear_data, input_data))
        f.close()
    return clear_input

# Read data from a given file and convert it into numpy array
def readFile(dataset_address):
    input_data = get_data(dataset_address)
    input_np = np.array(input_data)
    return input_np


def get_accuracy(estimated, actual):
    dataset_size = len(estimated)
    bool_list = [1 if estimated[i] == actual[i] else 0 for i in range(0, dataset_size)]
    acc = sum(bool_list) / dataset_size
    return acc * 100


class DecisionTreeNode:
    """A decision tree node."""

    def __init__(self, entropy, num_samples, num_samples_per_class, predicted_class):
        self.entropy = entropy
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.information_gain = 0
        self.split_condition = None
        self.threshold = None
        self.is_leaf = False
        self.left = None
        self.right = None

    def __repr__(self):
        return self.to_string()

    def _add_leaf(self, report, class_name, indent):
        value_fmt = "{}{} value: {}\n"
        val = ""
        val += " class: " + str(class_name)
        report += value_fmt.format(indent, "", val)
        return report

    def print_tree_recurse(self, report, node, depth, max_depth, spacing = 3):
        indent = ("|" + (" " * spacing)) * depth
        indent = indent[:-spacing] + "-" * spacing

        #value = None
        class_name = node.predicted_class
        per_class_cnt = node.num_samples_per_class
        right_child_fmt = "{} {} < {}\n"
        left_child_fmt = "{} {} >=  {}\n"
        truncation_fmt = "{} {}\n"

        if depth <= max_depth + 1:
            info_fmt = ""
            info_fmt_left = info_fmt
            info_fmt_right = info_fmt

            if not node.is_leaf:
                name = "X {}".format(node.feature_index)
                #threshold = tree_.threshold[node]
                threshold = "{1:.{0}f}".format(2, node.threshold)
                report += right_child_fmt.format(indent, name, threshold)
                report += info_fmt_left
                report = self.print_tree_recurse(report, node.left, depth + 1, max_depth)

                report += left_child_fmt.format(indent, name, threshold)
                report += info_fmt_right
                report = self.print_tree_recurse(report, node.right, depth + 1, max_depth)
            else:  # leaf
                report = self._add_leaf(report, class_name, indent)
        else:
            subtree_depth = self.compute_depth(node)
            if subtree_depth == 1:
                report = self._add_leaf(report, class_name, indent)
            else:
                trunc_report = "truncated branch of depth %d" % subtree_depth
                report += truncation_fmt.format(indent, trunc_report)
        return report

    def to_string(self):
        # """String representation of the tree."""
        report = ""
        report = self.print_tree_recurse(report, self, 0, math.inf)
        return report

    def compute_width(self):
        if not self.left and not self.right:
            return 1
        else:
            return self.left.compute_width() + self.right.compute_width()

    def compute_depth(self):
        if not self.left and not self.right:
            return 0
        else:
            return max(self.left.compute_depth(),
                       self.right.compute_depth()) + 1


class DecisionTreeClassifier():

    def __init__(self, max_depth=math.inf):
        self.max_depth = max_depth

    def find_thresholds(self, feature_column, Y):
        #Sorting in Descending Order
        sorted_indices = np.argsort(-1* feature_column)
        sorted_Y = [Y[x] for x in sorted_indices]
        sorted_features = [feature_column[x] for x in sorted_indices]
        prev_y = sorted_Y[0]
        prev_value = sorted_features[0]
        thresholds = []
        for i in range(1, len(sorted_indices)):
            if sorted_Y[i] != prev_y:
                thresholds.append((prev_value + sorted_features[i])/2)
            prev_y = sorted_Y[i]
            prev_value = sorted_features[i]
        return thresholds

    
    def conditional_entropy(self, prior_probs, Y_list):
        cond_entropy = 0
        for i in range(len(prior_probs)):
            cond_entropy += (prior_probs[i] * self.calc_entropy(Y_list[i]))
        return cond_entropy

    
    def calc_entropy(self, Y):
        label_space = list(set(Y))
        entropy = 0
        for label in label_space:
            no_wrt_label = len(np.where(Y == label)[0])
            prob = no_wrt_label / len(Y)
            entropy = entropy - np.log2(prob ** prob)
        return entropy

    #Assuming all the features are continuous right from the start
    def best_split(self, X, Y):
        cond_entropy = dict()
        split_conditon = dict()
        for i in range(X.shape[1]):
            tmp_column = X[:, i]
            thresholds = self.find_thresholds(tmp_column, Y)
            tmp_cond_entropy = math.inf
            for thresh in thresholds:
                Y_set1 = Y[np.where(tmp_column >= thresh)]
                Y_set2 = Y[np.where(tmp_column < thresh)]
                prior_probs = [len(Y_set1)/len(Y), len(Y_set2)/len(Y)]
                if self.conditional_entropy(prior_probs, [Y_set1, Y_set2]) < tmp_cond_entropy:
                    tmp_cond_entropy = self.conditional_entropy(prior_probs, [Y_set1, Y_set2])
                    tmp_thres = thresh
            cond_entropy[i] = tmp_cond_entropy
            split_conditon[i] = tmp_thres
        idx = min(cond_entropy, key=lambda k: cond_entropy[k])
        split_thresh = split_conditon[idx]
        min_entropy = min(cond_entropy.values())
        return idx, split_thresh, min_entropy

    def tree_growth(self, X, Y, depth):    
        samples_per_class = {i: sum(Y == i) for i in set(Y)}
        entropy = self.calc_entropy(Y)
        max_class = max(samples_per_class, key=lambda k: samples_per_class[k])
        node = DecisionTreeNode(entropy=entropy, num_samples=len(Y), num_samples_per_class=samples_per_class,
                                    predicted_class=max_class)        
        if (entropy != 0) & (len(Y) != 0):
            if depth+1 > self.max_depth:
                node.left = None
                node.right = None
                node.is_leaf = True
            else:
                idx, split_thresh, entropy_min = self.best_split(X, Y)
                node.information_gain = node.entropy - entropy_min
                node.feature_index = idx
                node.split_condition = "X[{idx}] >= {split_thresh}".format(idx=idx, split_thresh=split_thresh)
                node.threshold = split_thresh
                node.is_leaf = False
                X_left = X[np.where(X[:, idx] >= split_thresh)]
                Y_left = Y[np.where(X[:, idx] >= split_thresh)]
                X_right = X[np.where(X[:, idx] < split_thresh)]
                Y_right = Y[np.where(X[:, idx] < split_thresh)]
                node.left = self.tree_growth(X_left, Y_left, depth+1)
                node.right = self.tree_growth(X_right, Y_right, depth+1)
        else:
            node.is_leaf = True
        return node

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.root = self.tree_growth(X, Y, 0)
    
    #if len(Y) != 0:
    def predict(self, X):
        dataset_size = X.shape[0]
        predictions = list()
        for k in range(0, dataset_size):
            temp_node = self.root
            while not temp_node.is_leaf:
                feature_idx = temp_node.feature_index
                split_threshold = temp_node.threshold
                if X[k][feature_idx] >= split_threshold:
                    temp_node = temp_node.left
                else:
                    temp_node = temp_node.right
            predictions.append(temp_node.predicted_class)
        return predictions


def main():
    print('START Q1_AB\n')
    '''
    Start writing your code here
    '''
    for d in range(1, 6):
        # Load Data for Q1_AB
        train_data = readFile('datasets/Q1_train.txt')
        X_train_np = train_data[:, :-1].astype('float')
        Y_train_np = train_data[:, -1].astype('str')

        test_data = readFile('datasets/Q1_test.txt')
        X_test_np = test_data[:, :-1].astype('float')
        Y_test_np = test_data[:, -1].astype('str')

        dt = DecisionTreeClassifier(max_depth=d)
        dt.fit(X_train_np, Y_train_np)
        # print("Decision Tree of depth {0} is: ".format(d))
        # print(dt.root)
        y_train_preds = dt.predict(X_train_np)
        y_test_preds = dt.predict(X_test_np)
        print("DEPTH = {}".format(d))
        print("ACCURACY | Train {0:.2f} | Test {1:.2f}".format(get_accuracy(np.array(y_train_preds), Y_train_np), get_accuracy(np.array(y_test_preds), Y_test_np)))
    print('END Q1_AB\n')


if __name__ == "__main__":
    main()
