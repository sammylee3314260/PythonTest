#import numpy as np
from scipy.stats import shapiro,levene,kruskal
import os
import pandas as pd
#import pickle
import scikit_posthocs as sp
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

def kruskal_analy(df, group_col, value_col):
    groups = [df[df[group_col]==g][value_col] for g in df[group_col].unique()]
    stat, p = kruskal(*groups)
    print(f'Kruskal Wallis test: H = {stat:.4f},p={p:.4f}')
    if p < 0.05:
        return sp.posthoc_dunn(df, val_col = value_col, group_col = group_col, p_adjust='bonferroni')
    return None

def auto_anova(df, group_col, value_col, group_order = None, n_min = 5):
    if group_order:
        df[group_col] = pd.Categorical(df[group_col], categories=group_order,ordered=True)
    else:
        df[group_col] = pd.Categorical(df[group_col],ordered = True)
    df = df.sort_values(group_col)

    # calculate n
    group_count = df.groupby(group_col)[value_col].count()
    print(f'Sample sizes:\n{group_count}')

    # if n < n_min (5) non_parametric
    if (group_count < n_min).any():
        print(f"Some groups have n < {n_min}, non_parametric (KW test)")
        return kruskal_analy(df,group_col,value_col)
    
    # test normality
    normal = True
    for g in df[group_col].unique():
        stat, p = shapiro(df[df[group_col] == g][value_col])
        print(f'Shapiro test for {g}: p={p:.4f}')
        if p < 0.05: normal = False
    if not normal:
        print('Non-normal distribution: non-parametric (Kruskal wallis test)')
        return kruskal_analy(df, group_col, value_col)

    # check even variance
    grouped_data = [df[df[group_col]==g][value_col] for g in df[group_col].unique()]
    stat, p = levene(*grouped_data)
    print(f'Levene test: p = {p:.4f}')
    if p < 0.05:
        print('Variance not equal: Welch\'s anova')
        welch = pg.welch_anova(dv = value_col, between = group_col, data = df)
        print(welch)
        return pg.pairwise_gameshowell(dv=value_col, between=group_col, data=df)
    else:
        print("Variance equal: Standard Anova")
        anova = pg.anova(dv=value_col, between=group_col, data=df)
        print(anova)
        return pg.pairwise_tukey(dv=value_col, between=group_col,data=df)
    
def get_group_name(df, group_col):
    return df[group_col].unique()
# get column name = df.column

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