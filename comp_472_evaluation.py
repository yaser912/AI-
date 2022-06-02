from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
import seaborn as sns
import pandas as pd

# Source code credit for this function: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Truth')
    plt.xlabel('Prediction')

truth =      ["mask","Not a mask","mask","mask", "mask","Not a mask", "Not a mask", "mask", "mask", "Not a mask"]
prediction = ["mask","mask","mask","Not a mask","mask", "Not a mask", "mask", "Not a mask", "mask", "mask"]

cm = confusion_matrix(truth,prediction)
print_confusion_matrix(cm,["mask","Not a mask"])

print(classification_report(truth, prediction))


#F1 Score for mask class (apply data to formula):
print(2*(0.57*0.67/(0.57+0.67)))
#F1 Score for not a mask class:
print(2*(0.33*0.25/(0.33+0.25)))



#True positive: How many masks did the ANN predict correctly
#False positive: How many masks did the ANN say is not a mask even though it is
#True negative: how many non-mask images did the ANN predict correctly
#False negative: how many mask images did ANN say are non-mask when they are mask

#ACCURACY -> Out of all predictions, how many did the ANN get right?
#       prediction # / total number of images
#PRECISION -> How many mask predictions did the ANN get right? (out of all mask predictions)
#       prediction # / total number of images with mask (TP/TP+FP) (for mask class)
#RECALL -> Based on the actual number of mask images in data set, how many did 
#the ann predict correctly? 
#       Recall = TP/(TP+FN)  (for mask class)
#must do F1 score for mask and not mask class using formula





