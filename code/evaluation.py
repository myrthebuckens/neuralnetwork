# SCRIPT FOR EVALUATING THE FINAL OUTPUT AND GETTING INSIGHTS FOR THE ERROR ANALYSIS

import pandas as pd
from sklearn.metrics import classification_report

with open ('../data/final_output.conll') as infile:
    data = pd.read_csv(infile, sep ='\t')

gold = data['NER']
predicted = data['PRED']
tokens = data['token']

# gaining evaluation metrics
# classification report
print(classification_report(gold, predicted))

# confusion matrix
matrix = {'Gold': gold, 'Predicted': predicted}
df = pd.DataFrame(matrix, columns=['Gold', 'Predicted'])
confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
print(confusion_matrix)


# deeper insights in errors made
# empty lists for saving
ORG = []
PER = []
MISC = []
LOC = []

# printing the first 30 misclassifications per class that are not the O class
for ner, pred, token in zip (gold, predicted, tokens):
    if ner == 'ORG' and ner != pred and pred != 'O':
        ORG.append((token, pred))

    if ner == 'PER' and ner != pred and pred != 'O':
        PER.append((token, pred))

    if ner == 'MISC' and ner != pred and pred != 'O':
        MISC.append((token, pred))

    if ner == 'LOC' and ner != pred and pred != 'O':
        LOC.append((token, pred))

print('30 wrong predictions for ORG:', ORG[0:30])
print('\n')
print('30 wrong predictions for PER:', PER[0:30])
print('\n')
print('30 wrong predictions for MISC:', MISC[0:30])
print('\n')
print('30 wrong predictions for LOC:', LOC[0:30])
print('\n')

