# recipe_classifier
A machine learning project for classifying whether dishes are from Asian cuisines or not using the ingredients present in those dishes.

# Overview
This dataset contains data from approximately 40,000 recipes. Each recipe is labeled with a cuisine of origin, and the list of ingredients used in that recipe are included as well. There are 6713 unique ingredients included in the entire dataset, and 20 different cuisines of origin are represented. The exact number of recipes from each cuisine is visualized below.

The first task is to predict which recipes are from asian cuisines. For the purpose of this task, the following cuisines will be labeled as asian: 'indian', 'japanese', 'chinese', 'filipino', 'thai', 'korean', and 'vietnamese'. In other words, a classification model will be trained on the data and evaluated on a holdout test set, and the classification accuracy will be reported. The 1000 most frequently occurring ingredients will be used as features for this problem, and the values in the feature matrix will be indicators as to whether an ingredient is present in a given recipe.

The second task is to identify which ingredients are most important when determining whether a recipe is asian or not. Many cuisines include similar ingredients. For example, onions may be used in Chinese stir frys, French sauces, and Mexican tacos. However, other ingredients are more unique to certain cuisines. For example, one would not expect Mexican masa to appear in a Thai dish. Thus, the goal of this task is to discover the most defining ingredients in asian cuisine.

A detailed description of the methods and models used are included in both markdown and PDF form. 

# Results
The MLP with 2 hidden layers achieved the highest classification accuracy for task 1 (~96% correct on the test set). The logistic regression, random forest, and gradient boosting algorithms also achieved over 95% accuracy on the test data. 

Based on the feature importance scores from the tree-based models and the coefficients from the logistic regression model, the soy sauce and fish sauce appear to be the ingredients most strongly associated with Asian cuisine.
