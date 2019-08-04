from sklearn import tree

# |Weight | Texture | Label
# |150    | Bumpy   | Orange
# |170    | Bumpy   | Orange
# |140    | Smooth  | Apple
# |130    | Smooth  | Apple
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1] # 0 - Apple / 1 - Orange

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[150, 0]]))