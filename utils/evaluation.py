import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score


def evaluate(model, texts, true_values):
    y_pred = model.predict(texts)
    y_true = true_values
    
    #confusion matrix
    cfm = confusion_matrix(y_true, y_pred)
    df_cm1 = pd.DataFrame(cfm, index = ["clean","offensive","hate"],
                  columns = ["clean","offensive","hate"])
    plt.clf()
    sns.heatmap(df_cm1, annot=True, cmap="Greys",fmt='g', cbar=True, annot_kws={"size": 30})
    
    
    #f1 evaluation
    evaluation = f1_score(y_true, y_pred, average='micro')

    print("F1 - micro: " + str(evaluation))

    evaluation = f1_score(y_true, y_pred, average='macro')
    print("F1 - macro: " + str(evaluation))