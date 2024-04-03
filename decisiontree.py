import numpy as np 
from collections import Counter
class Node: 
    def __init__(self, feature=None, treshold=None, left=None, right=None,*,value=None): 
        self.feature = feature
        self.treshold = treshold
        self.left = left
        self.right = right
        self.value=None
    

    def is_leaf_node(self): 
        return self.value is not None 
    

class DecisionTree: 
    def __init__(self, maximum_depth=100, min_sample_split=1, n_features=None): 
        self.maximum_depth = maximum_depth
        self.min_sample_split = min_sample_split 
        self.n_features=n_features
        self.root = None 

    def fit(self, X, y): 
        self.n_features = X.shape[1] if not  self.n_features else min(X.shape(1), self.n_features)
        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0): 

        #checking for stopping criteria 
        n_samples, n_features = X.shape 
        n_labels = len(np.unique(y))
        #find the best split 
        if (depth >= self.maximum_depth or n_labels==1 or n_samples<self.min_sample_split):  
            return Node(value = self.most_common_label(y)) 

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)     

        best_feature, best_tresh = self.best_split(X, y, feat_idxs)
        left_idxs, right_idxs = self.split(X[:, best_feature, best_tresh]) 

        left = self.grow_tree(X[left_idxs, : ], y[left_idxs], depth+1)
        right = self.grow_tree(X[right_idxs, : ], y[right_idxs], depth+1)

        return Node(best_feature, best_feature, left, right)
    
    def best_split(self, X, y, feat_idxs):
        best_gain = -1 
        split_idx, split_treshold = None

        for feat_idx, in feat_idxs:
            X_column = X[:, feat_idxs] 
            tresholds = np.unique(X_column)
            for t in tresholds: 
                #information gain calculation 
                gain = self.information_gain(y, X_column, t)    
                if gain > best_gain:
                    best_gain = gain    
                    split_idx = feat_idx 
                    split_treshold = t   
        return split_idx, split_treshold

    def information_gain(self, y, X_column, t):
        #parent entropy
        parent_entropy = self.entropy(self, y) 
        #create children 
        left_idxs, right_idxs = self.split(X_column, t)
        
        if len(left_idxs) == 0 or  len(right_idxs) == 0:
            return 0 
        #calculate the weighted average 
        n, n_l, n_r = len(y), len(left_idxs), len(right_idxs)
        e_l, e_r = self.entropy(y[left_idxs]), self.entropy(y[right_idxs]) 
        child_entropy = (n_l/n) * e_l +  (n_r/n)*e_r
        #information gain 

        information_gain = parent_entropy - child_entropy

        return information_gain

    def split(self, X_column,splitting_tresh): 
        left_idxs = np.argwhere(X_column <= splitting_tresh).flatten()
        right_idxs = np.argwhere(X_column > splitting_tresh).flatten()
        return left_idxs, right_idxs

    def entropy(self, y): 
        #entropy = sum([val.log(1/val)for val in occurences divided by the number of unique labels in that particular row ])
        occurences = {}
        for label in y: 
            if label in y: 
                 occurences[label] += 1 
            else: 
                occurences[label] = 1 
        probabilities = []
        for _, val in occurences.iterrows(): 
            probabilities.append(val/len(y))
        entropy = sum([prob*np.log(1/prob) for prob in probabilities if prob>0])
        return entropy 

        pass
    def most_common_label(self, y): 
        counter = Counter(y)

        return counter.most_common[1][0][0]
    
    def predict(self, X):
        return np.array([self.traverse_tree(x) for x in X]) 
    
    def traverse_tree(self, x, node):
        if node.is_leaf_node(): 
            return self.value  
        if x[node.feature] <= node.treshold: 
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)    
