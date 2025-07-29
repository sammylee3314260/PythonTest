#import numpy as np
from scipy.stats import shapiro,levene,kruskal
import os
import pandas as pd
#import pickle
import scikit_posthocs as sp
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = '/mnt/SammyRis/Sammy/2025072021_exp_recov_3D'
    file = '2025072021_nuclear_Ki67'
    df_ori = pd.read_pickle(os.path.join(path,file+'.pkl'))
    value = 'intensity_mean'
    print(df_ori.head())
    f,axes = plt.subplots(2,1,figsize=(5,6))
    # f.subplots_adjust(top=0.85, hspace=0.1)
    # f.set_figheight(4)
    data_date = ['2025-07-20','2025-07-21']
    title = ['Exposure 24 hr','Recover 24 hr']
    order = ['Hyper969','Hyper484','Ctrl','Hypo66','Hypo50','Hypo33']
    colors = sns.color_palette("dark:lightgray",n_colors=len(order)+2)[0:len(order)]
    palette = dict(zip(order,colors))
    for i in range(2):
        df = df_ori[df_ori['Date']==data_date[0]]
        grouped = df.groupby('Group')[value].apply(list)
        # normal test and equal variation test
        '''
        for g, values in df.groupby('Group')[value]:
            stat, p = shapiro(values)
            print(f"Group {g}: p={p:.4f} {'normal' if p>0.05 else 'not normal'}")
        Levens_stat, Levens_p = levene(*grouped)
        print(f"Levenes test p = {Levens_p:.4f} {'eq' if Levens_p>0.05 else 'not eq'}")
        '''
        # non-parametric test
        kruskal_stat, kruskal_p = kruskal(*grouped)
        print(f"Kruskal Wallis test p-value: {kruskal_p}")
        # pairwise test
        dunn = sp.posthoc_dunn(df,val_col=value,group_col='Group',p_adjust='bonferroni')
        print(dunn)

        sns.barplot(data=df,x='Group',y=value,errorbar='se',capsize=0.2,color='lightgray',order=order,palette=palette,ax=axes[i])
        # sns.violinplot(data=df,x='Group',y=value,color='lightgray',errorbar='se',capsize=0.1,order=order,ax=axes[i])
        sns.stripplot(data=df,x='Group',color='darkorange',y=value,jitter=True,dodge=True,ax=axes[i],order=order) #  color='black'
        pairs=[
            ('Hyper969','Ctrl'),
            ('Hyper484','Ctrl'),
            ('Ctrl','Hypo66'),
            ('Ctrl','Hypo50'),
            ('Ctrl','Hypo33'),
        ]
        custom_p = [dunn.loc[pair[0],pair[1]] for pair in pairs]

        annotator = Annotator(axes[i],pairs,data=df,x='Group',y=value)
        annotator.configure(test=None,text_format='star',loc='outside',verbose=2)
        print(custom_p)
        annotator.set_pvalues_and_annotate(custom_p)
        # annotator.apply_and_annotate()
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        ymin, ymax = axes[i].get_ylim()
        axes[i].set_ylim(ymin,ymax*1.7)
        axes[i].set_title(title[i],weight='bold',pad=20)
        axes[i].set_xlabel(None)
        axes[i].set_ylabel('Intensity',weight='bold')
    f.suptitle("Ki67 intensity",weight='bold')
    f.tight_layout(rect=[0,0,1,0.97])
    plt.show()