import matplotlib.pyplot as plt
import seaborn as sns

def WriteConfusionSeaborn(m, labels, outpath):
    '''
    INPUT:
        m: confusion matrix (numpy array)
        labels: List of string, the category name of each entry in m
        name: Name for the output png plot
    '''
    fig, ax = plt.subplots()
    inn = m / m.sum(1, keepdims=True)
    ax = sns.heatmap(inn, cmap='Blues', fmt='.2%', xticklabels=labels, yticklabels=labels, annot=True, annot_kws={"size": 12})
    for t in ax.texts:
        t.set_text(t.get_text()[:-1])

    fig.savefig(outpath)
    print(m)
    print(f"Saved figure to {outpath}.")
    plt.close(fig)
