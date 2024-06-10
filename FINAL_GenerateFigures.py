import numpy as np
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from helperFunctions import pointcloud
from helperFunctions import pairCorrelationFunction, topographicalCorrelationMap, plotTopographicalCorrelationMap
from helperFunctions import get_colors
import matplotlib.patches as patches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
sns.set_theme(style='white',font_scale=3)

# Update path below to point to "ForRelease_SummaryStatisticsData.csv", available via the link in github
pathToSummaryDataframe = './ForRelease_SummaryStatisticsData.csv'
stats_df = pd.read_csv(pathToSummaryDataframe)
    
# For convenience further down
info_df = stats_df[stats_df.columns[0:12]]
data_df = stats_df[stats_df.columns[12:]]
data_df = data_df.astype(float)

#%% Setup
# Define things like colours for different celltypes that can be used throughout plotting for consistency
cm = plt.cm.tab10
celltypes = {'CD146':cm(0),
             'CD34':cm(1),
             'Cytotoxic T Cell':cm(2),
             'Macrophage':cm(3),
             'Neutrophil':cm(4),
             'Periostin':cm(5),
             'Podoplanin':cm(6),
             'SMA':cm(8),
             'T Helper Cell':cm(9),
             'Treg Cell':plt.cm.tab20b(0)
              ,
              'Epithelium (imm)':[1,0.9,0.9,1],
              'Epithelium (str)':[0.9,1,0.9,1]}

figure2 = False
figure3 = False
figure4 = False
figure5 = False
supplementary_S2 = False
#%% FIGURE TWO
if figure2:
    np.random.seed(2)
    names = ['5398_cancer_ID-220','5398_adenoma_ID-129']
    pcfs = {}
    for name in names:
        pathToData = f'./DataForFigures/{name}.csv'
        stats = stats_df[stats_df['Name'] == name]
        df = pd.read_csv(pathToData)
        
        celltypeToPlot = [['Neutrophil','T Helper Cell']
                          ]
        
        # Plot cell counts
        counts = {}
        for celltype in celltypes:
            if 'Epithelium' not in celltype:
                counts[celltype] = np.sum(df['Celltype'] == celltype)
        plt.figure(figsize=(20,20))
        plt.bar(counts.keys(),counts.values(),color=celltypes.values())
        plt.xticks(rotation=90)
        plt.ylim([0,750])
        plt.xlabel('Cell type')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'./Figure2/{name}_CellCounts.png')
        plt.savefig(f'./Figure2/{name}_CellCounts.svg')
        
        plt.figure(figsize=(20,20))
        s = 50
        g = 0.9
        plt.scatter(df['x'],df['y'],color=[g,g,g,1],s=s)
        
        for ct in np.unique(celltypeToPlot):
            mask = df['Celltype'] == ct
            plt.scatter(df['x'][mask],df['y'][mask],color=celltypes[ct],s=s*2)
        plt.gca().axis('equal')
        x, y = np.min(df['x']), np.min(df['y'])
        for i in [150,1150]: # Corresponding to ROI boundaries
            plt.gca().axhline(y+i,linestyle=':',lw=4,c='k')
            plt.gca().axvline(x+i,linestyle=':',lw=4,c='k')
        plt.savefig(f'./Figure2/{name}_Neut-THC.png')
        plt.savefig(f'./Figure2/{name}_Neut-THC.svg')
        
        for pair in celltypeToPlot:
            # Plot statistics
            pc = pointcloud('Fig2',np.asarray([df['x'],df['y']]).T)
            pointcloud.addLabels(pc,'Celltype', 'categorical', df['Celltype'])
            
            
            r,g,cont = pairCorrelationFunction(pc, 'Celltype', pair,maxR=150,annulusStep=1,annulusWidth=10)
            pcfs[name]=g
            plt.figure(figsize=(20,20))
            plt.plot(r,g,lw=5)
            plt.gca().axhline(1,c='k',linestyle=':',lw=4)
            plt.ylim([0,3.75])
            plt.xlim([0,150])
            plt.xlabel('Radius ($r$)')
            plt.ylabel('$g_{N Th}(r)$')
            # plt.title(f'{pair[0]} to {pair[1]}')
            plt.savefig(f'./Figure2/{name}_{pair[0]}-{pair[1]}_PCF.png')
            plt.savefig(f'./Figure2/{name}_{pair[0]}-{pair[1]}_PCF.svg')

            
            tcm = topographicalCorrelationMap(pc, 'Celltype', pair[0], 'Celltype', pair[1])
            plt.figure(figsize=(20,20))
            ax = plt.gca()
            plotTopographicalCorrelationMap(pc,tcm,ax=ax,cmap='RdBu_r',colorbarLimit=20)
            ax.set_xlim([x+150,x+1150])
            ax.set_ylim([y+150,y+1150])
            plt.savefig(f'./Figure2/{name}_{pair[0]}-{pair[1]}_TCM.png')
            plt.savefig(f'./Figure2/{name}_{pair[0]}-{pair[1]}_TCM.svg')
        
        # Plot Wasserstein distances
        # Get columns
        cols = [v for v in stats.columns if 'WassersteinDistance' in v]
        # Convert to dictionary
        wds = {}
        for v in cols:
            [b,c] = v.split('_')[1].split('-')
            wds[(b,c)] = stats[v].iloc[0]
        ser = pd.Series(list(wds.values()),
                      index=pd.MultiIndex.from_tuples(wds.keys()))
        dfw = ser.unstack()
        
        #% Plot QCM distances
        # Get columns
        cols = [v for v in stats.columns if 'QCM' in v]
        # Convert to dictionary
        qcms = {}
        for v in cols:
            [b,c] = v.split('_')[1].split('-')
            qcms[(b,c)] = float(stats[v].iloc[0])
        ser = pd.Series(list(qcms.values()),
                      index=pd.MultiIndex.from_tuples(qcms.keys()))
        dfq = ser.unstack().fillna(0)
        
        #% Plot QCM and Wasserstein on the same heatmap
        mask1 = np.triu(np.ones_like(dfw, dtype=bool))
        mask2 = np.tril(np.ones_like(dfq, dtype=bool))
        plt.figure(figsize=(30,20))
        ax = plt.gca()
        sns.heatmap(dfw,mask=mask1,ax=ax,cmap='Greens',linewidths=.5,square=True,cbar_kws={'label':'Wasserstein Distance', "pad":-.01},vmax=350,vmin=0)
        sns.heatmap(dfq,mask=mask2,ax=ax,cmap='RdBu_r',linewidths=.5,square=True,cbar_kws={'label':'QCM', "pad":-.01},center=0,vmax=5,vmin=-3)
        plt.gca().axis('equal')
        plt.tight_layout()
        plt.savefig(f'./Figure2/{name}_Wasserstein-QCM-Heatmap.png')
        plt.savefig(f'./Figure2/{name}_Wasserstein-QCM-Heatmap.svg')
    
#%% FIGURE THREE
if figure3:
    np.random.seed(3)
    import umap
    reducer = umap.UMAP(n_components=2)
    
    from sklearn.preprocessing import StandardScaler
    scaling = StandardScaler()
    Y = scaling.fit_transform(data_df)
            
    embedding = reducer.fit_transform(Y)
    
    diseasecolsdict = {'cancer' : plt.cm.Set1(0),
                       'adenoma': plt.cm.Set1(1)}
    disease = np.asarray([diseasecolsdict[v] for v in info_df['disease']])
    
    plt.figure(figsize=(18,18))
    plt.scatter(embedding[:,0],embedding[:,1],c=disease)
    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig('./Figure3/Global_UMAP-by-disease.png')
    plt.savefig('./Figure3/Global_UMAP-by-disease.svg')
    def build_legend(data):
        # Thanks to https://stackoverflow.com/questions/58718764/how-to-create-a-color-bar-using-a-dictionary-in-python
        """
        Build a legend for matplotlib plt from dict
        """
        from matplotlib.lines import Line2D
        legend_elements = []
        for key in data:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=key,
                                            markerfacecolor=data[key], markersize=40))
        return legend_elements
    diseasecolsdict_renamedForFigure = {'carcinoma' : plt.cm.Set1(0),
                                        'adenoma': plt.cm.Set1(1)}
    legend_elements = build_legend(diseasecolsdict_renamedForFigure)
    plt.figure(figsize=(6.5,3))
    plt.gca().legend(handles=legend_elements,loc='center')
    plt.gca().set_axis_off()
    plt.show()
    plt.savefig('./Figure3/disease_legend.png')
    plt.savefig('./Figure3/disease_legend.svg')

    IDs = np.unique(info_df['sampleID'])
    cols = get_colors(len(IDs))
    IDcolsdict = {IDs[v]:cols[v] for v in range(len(IDs))}
    IDcols = np.asarray([IDcolsdict[v] for v in info_df['sampleID']])
    plt.figure(figsize=(18,18))
    plt.scatter(embedding[:,0],embedding[:,1],c=IDcols)
    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig('./Figure3/Global_UMAP-by-ID.png')
    plt.savefig('./Figure3/Global_UMAP-by-ID.svg')
    
    disease = np.asarray([diseasecolsdict[v] for v in info_df['disease']])
    #%
    examples = [41632, 25914, 24505, 23559, 17903, 12575]
    for example in examples:
        np.random.seed(3)
        mask = info_df['sampleID'] == example
        cols = np.asarray([[0.7,0.7,0.7,1.0] if ~mask[v] else disease[v] for v in range(len(disease))])
        plt.figure(figsize=(18,18))
        plt.scatter(embedding[~mask,0],embedding[~mask,1],c=cols[~mask])
        plt.scatter(embedding[mask,0],embedding[mask,1],c=cols[mask])
        plt.gca().set_aspect('equal', 'datalim')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.savefig(f'./Figure3/Global_UMAP_example-{example}.png')
        plt.savefig(f'./Figure3/Global_UMAP_example-{example}.svg')

    
    # Do some quantification
    from sklearn import metrics
    for cluster in ['sampleID','disease']:
        silhouette_avg = metrics.silhouette_score(Y, info_df[cluster])    
        print(cluster, silhouette_avg)
        
    # Set flag to True to regenerate the RF data
    generateRandomForestDatafiles = False
    if generateRandomForestDatafiles:
        np.random.seed(3)
        for sampleID in np.unique(info_df['sampleID']):
            mask = info_df['sampleID'] == sampleID
            scaling = StandardScaler()
            Y = scaling.fit_transform(data_df)
            Y = Y[mask,:]
            
            diseasecolsdict = {'cancer' : plt.cm.Set1(0),
                               'adenoma': plt.cm.Set1(1)}
            disease = np.asarray([diseasecolsdict[v] for v in info_df[mask]['disease']])
            subsets = {'Counts':['Count'],
                       'Counts-PCF':['Count','PCF'],
                       'Counts-PCF-QCM-Wasserstein':['Count','PCF','QCM','WassersteinDistance'],
                       'All':['Count','PCF','QCM','WassersteinDistance','TDA']}
            for subset in subsets:
                prefixes = subsets[subset]
                mask = [v.split('_')[0] in prefixes for v in data_df.columns]
                Z = Y[:,mask]
                
                reducer = umap.UMAP(n_components=2)
                
                embedding = reducer.fit_transform(Z)
                
                plt.figure(figsize=(18,18))
                plt.scatter(embedding[:,0],embedding[:,1],c=disease,s=200)
                plt.gca().set_aspect('equal', 'datalim')
                plt.xlabel('UMAP 1')
                plt.ylabel('UMAP 2')
                plt.savefig(f'./Figure3/{sampleID}_UMAP-by-disease_{subset}.png')
                plt.close()
                
            
        # Now do all the random forests
        subsets = {'Counts':['Count'],
                   'PCF':['PCF'],
                   'QCM':['QCM'],
                   'WassersteinDistance':['WassersteinDistance'],
                   'TDA':['TDA'],
                   'Counts-PCF':['Count','PCF'],
                   'Counts-QCM':['Count','QCM'],
                   'Counts-WassersteinDistance':['Count','WassersteinDistance'],
                   'Counts-TDA':['Count','TDA'],
                   'PCF-QCM-Wasserstein':['PCF','QCM','WassersteinDistance'],
                   'Counts-PCF-QCM-Wasserstein':['Count','PCF','QCM','WassersteinDistance'],
                   'All':['Count','PCF','QCM','WassersteinDistance','TDA']}
        
        results = {}
        for sampleID in np.unique(info_df['sampleID']):
            print(sampleID)
            mask = info_df['sampleID'] == sampleID
            diseasecolsdict = {'cancer' : plt.cm.Set1(0),
                               'adenoma': plt.cm.Set1(1)}
            disease = np.asarray([diseasecolsdict[v] for v in info_df[mask]['disease']])
            
            acc_means = {}
            acc_sds = {}
            silhouettes = {}
            for subset in subsets:
                prefixes = subsets[subset]
                mask2 = [v.split('_')[0] in prefixes for v in data_df.columns]
            
                X = data_df[mask]
                X = X[data_df.columns[mask2]]
                y = [0 if v == 'adenoma' else 1 for v in info_df[mask]['disease']]
                features = data_df.columns
                
                scores = []
                confs = []
                for state in range(10):
                    state = state
                    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=state)
                    
                    state = state
                    rf = RandomForestClassifier(n_estimators=1000,random_state=state)
                    
                    rf.fit(X_train,y_train)
                    
                    score = rf.score(X_test, y_test)
                    y_pred = rf.predict(X_test)
                    conf = confusion_matrix(y_test, y_pred)
                    scores.append(score)
                    confs.append(conf)
                toSave = {'scores':scores,'confusion_matrices':confs}
                with open(f'./Figure3/Data/RFmetrics_{sampleID}_{subset}.pkl','wb') as fid:
                    pickle.dump(toSave, fid)
                acc_means[subset] = np.mean(scores)
                acc_sds[subset] = np.std(scores)
            results[sampleID] = {'acc_means':acc_means, 'acc_sds':acc_sds}
    else:
        print('Using pre-calculated Random Forest data for plotting')
        #%
    subsets = {'Counts':['Count'],
               'PCF':['PCF'],
               'QCM':['QCM'],
               'WassersteinDistance':['WassersteinDistance'],
               'TDA':['TDA'],
               'Counts-PCF':['Count','PCF'],
               'Counts-QCM':['Count','QCM'],
               'Counts-WassersteinDistance':['Count','WassersteinDistance'],
               'Counts-TDA':['Count','TDA'],
               'PCF-QCM-Wasserstein':['PCF','QCM','WassersteinDistance'],
               'Counts-PCF-QCM-Wasserstein':['Count','PCF','QCM','WassersteinDistance'],
               'All':['Count','PCF','QCM','WassersteinDistance','TDA']}
    results = {}
    for sampleID in np.unique(info_df['sampleID']):
        acc_means = {}
        acc_sds = {}
        for subset in subsets:
            with open(f'./Figure3/Data/RFmetrics_{sampleID}_{subset}.pkl','rb') as fid:
                resultsDict = pickle.load(fid)
            scores = resultsDict['scores']
            acc_means[subset] = np.mean(scores)
            acc_sds[subset] = np.std(scores)
        results[sampleID] = {'acc_means':acc_means, 'acc_sds':acc_sds}
    # Remove sampleIDs 37315 and 40814 as all ROIs come from the same class (cancer/adenoma) so MDIs are all 0
    exclude = [37315, 40814]
    d = {v:results[v]['acc_means'] for v in results if v not in exclude}
    df_accuracies = pd.DataFrame.from_dict(d).T
    # Sort columns by mean
    order = list(df_accuracies.mean().sort_values().index)
    
    IDs = np.unique(info_df['sampleID'])
    cols = get_colors(len(IDs))
    col_dict = {IDs[v]:cols[v] for v in range(len(IDs))}
    
    plt.figure(figsize=(30,20))
    sns.boxplot(data=df_accuracies, order=order, color='w')#, s=10)
    df_accuracies_manip = df_accuracies.T.reset_index(names=['Subset'])
    df_accuracies_manip = df_accuracies_manip.melt(id_vars='Subset', var_name='ID', value_name='Accuracy')
    sns.swarmplot(data=df_accuracies_manip, x='Subset',y='Accuracy', order=order,s=12, hue='ID',palette=col_dict)
    plt.gca().set_xticklabels(plt.gca().get_xticklabels(),rotation=90)
    sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(1, 1),ncols=2,markerscale=4)
    plt.ylabel('Accuracy')
    plt.tight_layout()
    
    
    examples = [41632, 25914, 24505, 23559, 17903, 12575]
    for toHighlight in examples:
        col_dict = {IDs[v]:[0,0,0,1.0] if IDs[v] != toHighlight else [1.0,0,0,1.0] for v in range(len(IDs))}
        
        plt.figure(figsize=(30,20))
        toPlot = ['QCM','WassersteinDistance','PCF','Counts','TDA','All']
        sns.boxplot(data=df_accuracies, order=toPlot, color='w')#, s=10)
        sns.swarmplot(data=df_accuracies_manip, x='Subset',y='Accuracy', order=toPlot,s=20, hue='ID',palette=col_dict)
        plt.gca().set_xticklabels(plt.gca().get_xticklabels(),rotation=90)
        sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(1, 1),ncols=2,markerscale=4)
        plt.title(toHighlight)
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig(f'./Figure3/Boxplots_subset_highlight-{toHighlight}.png')
        plt.savefig(f'./Figure3/Boxplots_subset_highlight-{toHighlight}.svg')
        plt.close()
        
        mask = info_df['sampleID'] == toHighlight
        scaling = StandardScaler()
        Y = scaling.fit_transform(data_df)
        Y = Y[mask,:]
        disease = np.asarray([diseasecolsdict[v] for v in info_df[mask]['disease']])
        for subset in ['Counts','PCF','QCM','WassersteinDistance','TDA','All']:
            np.random.seed(3)
            prefixes = subsets[subset]
            mask = [v.split('_')[0] in prefixes for v in data_df.columns]
            Z = Y[:,mask]
            
            reducer = umap.UMAP(n_components=2)
            
            embedding = reducer.fit_transform(Z)
            
            plt.figure(figsize=(18,18))
            plt.scatter(embedding[:,0],embedding[:,1],c=disease,s=200)
            plt.gca().set_aspect('equal', 'datalim')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.savefig(f'./Figure3/{toHighlight}_UMAP-by-disease_{subset}.png')
            plt.savefig(f'./Figure3/{toHighlight}_UMAP-by-disease_{subset}.svg')
            plt.close()
#%%  
if figure4:
    np.random.seed(4)    
    sampleIDs = [25914,5531]
    # For supplementary figure 2
    boxplotdata = {}
    for sampleID in sampleIDs:
        mask = info_df['sampleID'] == sampleID
        
        scaling = StandardScaler()
        Y = scaling.fit_transform(data_df)
        Y = Y[mask,:]
        
        diseasecolsdict = {'cancer' : plt.cm.Set1(0),
                           'adenoma': plt.cm.Set1(1)}
        disease = np.asarray([diseasecolsdict[v] for v in info_df[mask]['disease']])
        subsets = {'All':['Count','PCF','QCM','WassersteinDistance','TDA']}
        
        
        #% Do random forest to separate adenoma and carcinoma using these metrics
        X = data_df[mask] # Unscaled Y
        y = [0 if v == 'adenoma' else 1 for v in info_df[mask]['disease']]
        features = data_df.columns
        
        scores = []
        fis = []
        for state in range(10):
            state = state
            X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=state)
            
            state = state
            rf = RandomForestClassifier(n_estimators=1000,random_state=state)
            
            rf.fit(X_train,y_train)
            
            score = rf.score(X_test, y_test)
            print(f'Classifier score: {score}')
            scores.append(score)
            
            f_i = list(zip(features,rf.feature_importances_))
            f_i.sort(key = lambda x : x[1])
            fis.append(f_i)
        
        # Get average score and average feature importances
        meanscore = np.mean(scores)
        MDIs = {v[0]:[] for v in fis[0]}
        
        for repeat in fis:
            for v in range(len(repeat)):
                MDIs[repeat[v][0]].append(repeat[v][1])
        MDI_mean = {}
        MDI_sd = {}
        for key in MDIs:
            MDI_mean[key] = np.mean(MDIs[key])
            MDI_sd[key] = np.std(MDIs[key])
        sorted_MDI_mean = sorted(MDI_mean.items(), key=lambda x:x[1])
        
        
        nToPlot = 15        
        f_trunc = sorted_MDI_mean[-nToPlot:]
        cols = {'Count':'r',
         'H1-MeanDeath':plt.cm.Blues(1.0),
         'H1-MeanPersistence':plt.cm.Blues(0.9),
         'H1-MeanBirth':plt.cm.Blues(0.8),
         'H0-MeanDeath':plt.cm.Blues(0.7),
         'H0-MeanPersistence':plt.cm.Blues(0.6),
         'H0-MeanBirth':plt.cm.Blues(0.5),
         'H1-nFeatures':plt.cm.Blues(0.4),
         'H0-nFeatures':plt.cm.Blues(0.3),
         'QCM':plt.cm.Oranges(0.5),
         'WassersteinDistance':plt.cm.Purples(0.5),
         'gmin':plt.cm.Greens(0.8),
         'gmax':plt.cm.Greens(0.6),
         'rtrough':plt.cm.Greens(0.4),
         'rpeak':plt.cm.Greens(0.2)}
        def build_legend(data):
            # Thanks to https://stackoverflow.com/questions/58718764/how-to-create-a-color-bar-using-a-dictionary-in-python
            """
            Build a legend for matplotlib plt from dict
            """
            from matplotlib.lines import Line2D
            legend_elements = []
            for key in data:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=key,
                                                markerfacecolor=data[key], markersize=40))
            return legend_elements
        cols_prettierForCaption = {'Count':'r',
         'PH: $H^1$ mean death value':plt.cm.Blues(1.0),
         'PH: $H^1$ mean persistence':plt.cm.Blues(0.9),
         'PH: $H^1$ mean birth value':plt.cm.Blues(0.8),
         'PH: $H^0$ mean death value':plt.cm.Blues(0.7),
         'PH: $H^0$ mean persistence':plt.cm.Blues(0.6),
         'PH: $H^0$ mean birth value':plt.cm.Blues(0.5),
         'PH: $H^1$ number of features':plt.cm.Blues(0.4),
         'PH: $H^0$ number of features':plt.cm.Blues(0.3),
         'QCM':plt.cm.Oranges(0.5),
         'Wasserstein Distance':plt.cm.Purples(0.5),
         'PCF: minimum value':plt.cm.Greens(0.8),
         'PCF: maximum value':plt.cm.Greens(0.6),
         'PCF: $r$ value at minimum':plt.cm.Greens(0.4),
         'PCF: $r$ value at maximum':plt.cm.Greens(0.2)}
        legend_elements = build_legend(cols_prettierForCaption)
        plt.figure(figsize=(7.5,11))
        plt.gca().legend(handles=legend_elements,loc='center')
        plt.gca().set_axis_off()
        plt.show()
        plt.savefig('./Figure4/MDI_colorbar.png')
        plt.savefig('./Figure4/MDI_colorbar.svg')


        colors = []
        for v in f_trunc:
            for key in cols:
                if key in v[0]:
                    colors.append(cols[key])
        plt.figure(figsize=(18,18))
        plt.barh([x[0] for x in f_trunc],[x[1] for x in f_trunc],color=colors,xerr=[MDI_sd[x[0]] for x in f_trunc])
        plt.tight_layout()
        plt.xlabel('MDI')
        plt.title(f'Mean classifier score: {meanscore:.3f}')
        plt.savefig(f'./Figure4/MDI_{sampleID}_top{nToPlot}.png')
        plt.savefig(f'./Figure4/MDI_{sampleID}_top{nToPlot}.svg')
        
        #%
        features = [v[0] for v in sorted_MDI_mean]
    
        celltypeA, celltypeB = [], []
        pairs = []
        methods = []
        MDIs = []
        for v in features:
            parts = v.split('_')
            if parts[0] == 'PCF':
                pair = parts[2]
                methods.append(parts[1])
            elif parts[0] == 'Count':
                pair = '-'.join([parts[1],parts[1]])
                methods.append(parts[0])
            else:
                pair = parts[1]
                if parts[0] == 'TDA':
                    methods.append(parts[-1])
                else:
                    methods.append(parts[0])
                
            celltypeA.append(pair.split('-')[0])
            celltypeB.append(pair.split('-')[1])
            MDIs.append(MDI_mean[v])
            pairs.append(pair)
            
        methodArray = np.array(methods)
        celltypeA = np.array(celltypeA)
        celltypeB = np.array(celltypeB)
        MDIs = np.array(MDIs)
        pairs = np.array(pairs)
            
        # Find top nToPlot cell-cell pairs, and then plot the contribution to each pair of each method
        bigList = {}
        totalMDIs = {}
        for pair in np.unique(pairs):
            mask = pairs == pair
            methods = methodArray[mask]
            MDI = MDIs[mask]
            lookup = {}
            for i in range(len(MDI)):
                lookup[methods[i]] = MDI[i]
            bigList[pair] = lookup
            totalMDIs[pair] = np.sum(MDI)
        sorted_totalMDIs = sorted(totalMDIs.items(), key=lambda x:x[1])
    
        nToPlot = 10
        pairsToPlot = sorted_totalMDIs[-nToPlot:]
        pairsToPlot = np.flipud(pairsToPlot )

        plt.figure(figsize=(18,nToPlot+2))
        for thing in pairsToPlot:
            pair = thing[0]
            vals = bigList[pair]
            # Sort vals alphabetically by keys
            methods = sorted(vals.keys(), key=lambda x:x[0])
            
            bottom = 0
            for method in methods:
                MDI = vals[method]
                plt.gca().barh(pair, MDI, label=pair, left=bottom,facecolor=cols[method])
                bottom += MDI
        plt.tight_layout()
        plt.xlabel('Cumulative MDI')
        plt.title(f'Mean classifier score: {meanscore:.3f}')
        plt.savefig(f'./Figure4/MDI_{sampleID}_byMethod.png')
        plt.savefig(f'./Figure4/MDI_{sampleID}_byMethod.svg')
    
        toSave = {'scores':scores,'feature_importances':fis}
        with open(f'./Figure4/Data/___FeatureImportance-data_{sampleID}.pkl','wb') as fid:
            pickle.dump(toSave, fid)
        plt.close('all')

        
        
        #% Boxplot
        featuresToPlot = ['Count_Periostin','TDA_Periostin-Periostin_H0-nFeatures','PCF_gmax_Periostin-Periostin',
                          'PCF_gmin_Neutrophil-SMA','TDA_Neutrophil-SMA_H1-MeanDeath','TDA_Neutrophil-SMA_H1-MeanBirth']
        toPlot = []
        for i in range(np.shape(X)[0]):
            newDict = {}
            if y[i] == 0:
                newDict['Stage'] = 'adenoma'
            else:
                newDict['Stage'] = 'cancer'
            for feature in featuresToPlot:
                newDict[feature] = X[feature].iloc[i]
            toPlot.append(newDict)
        df_plot = pd.DataFrame(toPlot)
        boxplotdata[sampleID] = df_plot
        #%
        from scipy.stats import mannwhitneyu
        from statannotations.Annotator import Annotator
        
        for feature in featuresToPlot:
            pairs = [(('adenoma','cancer'))]#,('DiseaseState','cancer'))
            plt.figure(figsize=(9,9))
            ax=sns.boxplot(data=df_plot, x='Stage',y=feature,palette=diseasecolsdict)
            
            # plt.ylim(featureLims[feature])
            annot = Annotator(ax, pairs, data=df_plot,x='Stage',y=feature)#,hue='Cluster')
            annot.configure(test='Mann-Whitney',text_format='star', loc='outside')
            annot.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.00)
            sns.despine()
            plt.tight_layout()
            plt.savefig(f'./Figure4/Statistic_{sampleID}_{feature}.png')
            plt.savefig(f'./Figure4/Statistic_{sampleID}_{feature}.svg')
    # Combined boxplots
    df_boxes = boxplotdata[25914]
    df_boxes['Patient ID'] = 'A'
    df_temp = boxplotdata[5531]
    df_temp['Patient ID'] = 'B'
    df_boxes = pd.concat([df_boxes,df_temp])
    featuresToPlot_words = {'Count_Periostin':'Count - Periostin',
                            'TDA_Periostin-Periostin_H0-nFeatures':'Periostin-Periostin - Number of $H_0$ features',
                            'PCF_gmax_Periostin-Periostin':'Periostin-Periostin - maximum value of PCF',
                            'PCF_gmin_Neutrophil-SMA':'Neutrophil-SMA - minimum value of PCF',
                            'TDA_Neutrophil-SMA_H1-MeanDeath':'Neutrophil-SMA - average $H_1$ feature death value',
                            'TDA_Neutrophil-SMA_H1-MeanBirth':'Neutrophil-SMA - average $H_1$ feature birth value'}
    featuresToPlot_symbols = {'Count_Periostin':'$N_P$',
                            'TDA_Periostin-Periostin_H0-nFeatures':'$N^{0}_{PP}$',
                            'PCF_gmax_Periostin-Periostin':'$g^{Hi}_{PP}$',
                            'PCF_gmin_Neutrophil-SMA':'$g^{Lo}_{NS}$',
                            'TDA_Neutrophil-SMA_H1-MeanDeath':'$\overline{{d^{1}_{NS}}}$',
                            'TDA_Neutrophil-SMA_H1-MeanBirth':'$\overline{{b^{1}_{NS}}}$'}
    for feature in featuresToPlot:
        # pairs = [(('adenoma','cancer'))]#,('DiseaseState','cancer'))
        pairs = [(("A","adenoma"), ("A","cancer")),
                 (("B","adenoma"), ("B","cancer"))]
        plt.figure(figsize=(8,12))
        ax=sns.boxplot(data=df_boxes, x='Patient ID',hue='Stage',y=feature,palette=diseasecolsdict)
        
        # plt.ylim(featureLims[feature])
        annot = Annotator(ax, pairs, data=df_boxes,x='Patient ID',y=feature,hue='Stage')
        annot.configure(test='Mann-Whitney',text_format='star', loc='outside')
        annot.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.00)
        sns.despine(offset=10,trim=True)
        # plt.legend(loc='upper left',bbox_to_anchor=(1.03,1))
        plt.gca().get_legend().remove()
        plt.ylabel(featuresToPlot_words[feature])
        plt.tight_layout()
        plt.savefig(f'./Figure4/Statistic_CombinedSamples_{feature}.png')
        plt.savefig(f'./Figure4/Statistic_CombinedSamples_{feature}.svg')
        
    for feature in featuresToPlot:
        # pairs = [(('adenoma','cancer'))]#,('DiseaseState','cancer'))
        pairs = [(("A","adenoma"), ("A","cancer")),
                 (("B","adenoma"), ("B","cancer"))]
        plt.figure(figsize=(8,12))
        ax=sns.boxplot(data=df_boxes, x='Patient ID',hue='Stage',y=feature,palette=diseasecolsdict)
        
        # plt.ylim(featureLims[feature])
        annot = Annotator(ax, pairs, data=df_boxes,x='Patient ID',y=feature,hue='Stage')
        annot.configure(test='Mann-Whitney',text_format='star', loc='outside')
        annot.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.00)
        sns.despine(offset=10,trim=True)
        # plt.legend(loc='upper left',bbox_to_anchor=(1.03,1))
        plt.gca().get_legend().remove()
        plt.ylabel(featuresToPlot_symbols[feature],rotation=0)
        plt.tight_layout()
        plt.savefig(f'./Figure4/Statistic_CombinedSamples_{feature}_symbol.png')
        plt.savefig(f'./Figure4/Statistic_CombinedSamples_{feature}_symbol.svg')
        
    plt.close('all')
#%% FIGURE FIVE

if figure5:
    np.random.seed(5)    
    # Load MDI data used in Figure 4
    sampleIDs = np.unique(info_df['sampleID'])
    # Remove sampleIDs 37315 and 40814 as all ROIs come from the same class (cancer/adenoma) so MDIs are all 0
    for ID in [37315, 40814]:
        sampleIDs = np.delete(sampleIDs,np.where(sampleIDs == ID))
    MDI_dict = {}
    for sampleID in sampleIDs:
        print(sampleID)
        with open(f'./Figure4/Data/___FeatureImportance-data_{sampleID}.pkl','rb') as fid:
            data = pickle.load(fid)
            fis = data['feature_importances']
            MDIs = {v[0]:[] for v in fis[0]}
            
            for repeat in fis:
                for v in range(len(repeat)):
                    MDIs[repeat[v][0]].append(repeat[v][1])
            MDI_mean = {}
            for key in MDIs:
                MDI_mean[key] = np.mean(MDIs[key])
            MDI_dict[sampleID] = MDI_mean
    MDI_df = pd.DataFrame.from_dict(MDI_dict).transpose()
    
    # Each row of MDI_df is a point in 1410 dimensional MDI space
    # Visualise
    import umap
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(MDI_df)
    
    # Do PCA
    nComp = 10
    from sklearn.decomposition import PCA
    Y = MDI_df.values
    pca = PCA(n_components=nComp,svd_solver="randomized")
    pca.fit(Y)
        
    p = pca.transform(Y)

    bigList = {}
    for sampleID in sampleIDs:
        vals = {}
        for celltype in celltypes:
            if 'Epithelium' not in celltype:
                summedMDIs = np.sum(MDI_df[[v for v in MDI_df.columns if celltype in v]],axis=1)
                vals[celltype] = summedMDIs[sampleID]
        # Normalise to sum to 1
        norm = np.sum(list(vals.values()))
        bigList[sampleID] = {v:vals[v]/norm for v in vals}
    #%
    # Sort bigList by periostin value
    bigList = {k: v for k, v in sorted(bigList.items(), key=lambda item: item[1]['Periostin'])}
    
    acceptMask = p[:,0] > 0
    
    # Do PCAs for supplemental Figure S3
    MGs = [{1:[24922],
            2:[41251],
            3:[17989]}
           ,
           {1:[25914],
            2:[43179],
            3:[48639]}
           ]
           
    
    for mgind, manualGrouping in enumerate(MGs):
        # Plot cluster features as boxplot
        ID2group = {}
        for group in manualGrouping:
            for ID in manualGrouping[group]:
                ID2group[ID] = group
    
        cm = plt.cm.tab10
        cols = [cm(ID2group[v]-1) if v in ID2group else [0.7,0.7,0.7,1] for v in MDI_df.index]
        plt.figure(figsize=(24, 24))
        plt.scatter(p[:,0], p[:,1],s=1500,c=cols)
        plt.gca().set_xlabel('PC1')
        plt.gca().set_ylabel('PC2')
        plt.savefig(f'./Figure5/For SI/PCA {mgind}.png')
        plt.savefig(f'./Figure5/For SI/PCA {mgind}.svg')
    
    
    # PCA for main text
    manualGrouping = {1:[23559],#,24922,25914],
                      2:[5531],#,43179,41251],
                      3:[28717]}#,17989,48639]}
    # Plot cluster features as boxplot
    ID2group = {}
    for group in manualGrouping:
        for ID in manualGrouping[group]:
            ID2group[ID] = group

    cm = plt.cm.tab10
    cols = [cm(ID2group[v]-1) if v in ID2group else [0.7,0.7,0.7,1] for v in MDI_df.index]
    plt.figure(figsize=(24, 24))
    plt.scatter(p[:,0], p[:,1],s=1500,c=cols)
    plt.gca().set_xlabel('PC1')
    plt.gca().set_ylabel('PC2')
    plt.savefig('./Figure5/PCA.png')
    plt.savefig('./Figure5/PCA.svg')
    
    
    #% Visualise top n contributing cell-cell pairs to each PC
    n = 25
    for pc_to_plot in [0,1]:#,2]:
        names = MDI_df.columns
        pcimportances = {pc_to_plot:{}}
        for x in celltypes:
            for y in celltypes:
                if ('Epithelium' not in x) and ('Epithelium' not in y):
                    pair = '-'.join([x,y])
                    mask = np.asarray([pair in v for v in names])
                    if x == y:
                        # Include cell counts under x-x interactions
                        ind = np.where(names == f'Count_{x}')[0][0]
                        mask[ind] = True
                    
                    pc = pca.components_[pc_to_plot,:]
                    importances = pc[mask]
                    pairTotalImp = np.sum(importances)
                    pcimportances[pc_to_plot][pair] = pairTotalImp
        
        imps = pd.DataFrame.from_dict(pcimportances)
        # Plot top n PCs
        namesToPlot = []
        vals = imps
        nlarg = vals.abs().nlargest(n,keep='all',columns=pc_to_plot)
        namesToPlot.extend(list(nlarg.index))
        namesToPlot.reverse()
        #%
        # sns.set_theme(style='white',font_scale=2)
        fig, ax = plt.subplots(figsize=(15,20),nrows=1,ncols=1,sharey=True)
        name_plot = []
        y_plot = []
        col_plot = []
        for y in range(len(namesToPlot)):
            name_plot.append(namesToPlot[y])
            y_plot.append(imps.loc[namesToPlot[y]][pc_to_plot])
            col_plot.append(celltypes[namesToPlot[y].split('-')[0]])
        ax.barh(y=name_plot,width=y_plot,color=col_plot)
        ax.axvline(0,c='k',linestyle=':',lw=2)
        ax.set_xlim([-1,1])
        ax.set_xlabel(f'PC{pc_to_plot+1}')
        plt.tight_layout()
        # sns.set_theme(style='white',font_scale=3)
        plt.savefig(f'./Figure5/PC{pc_to_plot+1}.png')
        plt.savefig(f'./Figure5/PC{pc_to_plot+1}.svg')
        
    
    IDs = {'Cluster One':[23559,24922,25914],
           'Cluster Two':[5531,43179,41251],
           'Cluster Three':[28717,17989,48639]}
    
                 
    statsToPlot = {'Count_Periostin':{'maxval':250,'cmap':plt.cm.copper_r(np.linspace(0,1,100)),'label':'$N_P$'},
                    'PCF_gmax_Neutrophil-Treg Cell':{'maxval':4,'cmap':plt.cm.Purples(np.linspace(0.1,1,100)),'label':'$g^{Hi}_{NT}$'},
                    'TDA_SMA-Macrophage_H1-MeanDeath':{'maxval':7,'cmap':plt.cm.Greens(np.linspace(0.1,1,100)),'label':'$\overline{D^1_{SM}}$'}
                   }
    for cluster in IDs:
        sampleIDs = IDs[cluster]
        for sampleID in sampleIDs:
            mask = info_df['sampleID'] == sampleID
            xs = list(info_df[mask]['xMin'])
            ys = list(info_df[mask]['yMin'])
                    
            edgecolors = {'adenoma':'k','cancer':'w'}
            
            for stat in statsToPlot:    
                arr = list(data_df[stat][mask])
                maxval = statsToPlot[stat]['maxval']
                cm = statsToPlot[stat]['cmap']
                label = statsToPlot[stat]['label']
                
                from matplotlib.colors import LinearSegmentedColormap
                cm = LinearSegmentedColormap.from_list('temp',cm)
    
                plt.figure(figsize=(24,24))
                ax = plt.gca()
                for i in range(len(info_df[mask])):
                    x = xs[i]
                    y = ys[i]
                    ec = edgecolors[info_df[mask].iloc[i]['disease']]
                    val = arr[i]
                    col = cm(val/maxval)
                    if not np.isnan(val):
                        rect = patches.Rectangle((x, y), 1000, 1000, linewidth=5, edgecolor=ec, facecolor=col)
                        ax.add_patch(rect)
                ax.axis('equal')
                plt.xlim([min(xs),max(xs)+1000])
                plt.ylim([min(ys),max(ys)+1000])
                
                # (use the colormap you want to have on the colorbar)
                img = plt.imshow(np.array([[0,maxval]]), cmap=cm)
                img.set_visible(False)
                
                plt.colorbar(orientation="vertical",label=label)
                plt.savefig(f'./Figure5/{cluster}/{sampleID}_{stat}.png')
                plt.savefig(f'./Figure5/{cluster}/{sampleID}_{stat}.svg')
                plt.close()
                

    
#%% SUPPLEMENTARY - S2 Extensions to Fig 4
if supplementary_S2:
    np.random.seed(4)    
    sampleIDs = [12575,17903,23559,41632]
    # For supplementary figure 2
    boxplotdata = {}
    for sampleID in sampleIDs:
        mask = info_df['sampleID'] == sampleID
        
        scaling = StandardScaler()
        Y = scaling.fit_transform(data_df)
        Y = Y[mask,:]
        
        diseasecolsdict = {'cancer' : plt.cm.Set1(0),
                           'adenoma': plt.cm.Set1(1)}
        disease = np.asarray([diseasecolsdict[v] for v in info_df[mask]['disease']])
        subsets = {'All':['Count','PCF','QCM','WassersteinDistance','TDA']}
        
        
        #% Do random forest to separate adenoma and carcinoma using these metrics
        X = data_df[mask] # Unscaled Y
        y = [0 if v == 'adenoma' else 1 for v in info_df[mask]['disease']]
        features = data_df.columns
        
        scores = []
        fis = []
        for state in range(10):
            state = state
            X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=state)
            
            state = state
            rf = RandomForestClassifier(n_estimators=1000,random_state=state)
            
            rf.fit(X_train,y_train)
            
            score = rf.score(X_test, y_test)
            print(f'Classifier score: {score}')
            scores.append(score)
            
            f_i = list(zip(features,rf.feature_importances_))
            f_i.sort(key = lambda x : x[1])
            fis.append(f_i)
        
        # Get average score and average feature importances
        meanscore = np.mean(scores)
        MDIs = {v[0]:[] for v in fis[0]}
        
        for repeat in fis:
            for v in range(len(repeat)):
                MDIs[repeat[v][0]].append(repeat[v][1])
        MDI_mean = {}
        MDI_sd = {}
        for key in MDIs:
            MDI_mean[key] = np.mean(MDIs[key])
            MDI_sd[key] = np.std(MDIs[key])
        sorted_MDI_mean = sorted(MDI_mean.items(), key=lambda x:x[1])
        
        
        nToPlot = 15        
        f_trunc = sorted_MDI_mean[-nToPlot:]
        cols = {'Count':'r',
         'H1-MeanDeath':plt.cm.Blues(1.0),
         'H1-MeanPersistence':plt.cm.Blues(0.9),
         'H1-MeanBirth':plt.cm.Blues(0.8),
         'H0-MeanDeath':plt.cm.Blues(0.7),
         'H0-MeanPersistence':plt.cm.Blues(0.6),
         'H0-MeanBirth':plt.cm.Blues(0.5),
         'H1-nFeatures':plt.cm.Blues(0.4),
         'H0-nFeatures':plt.cm.Blues(0.3),
         'QCM':plt.cm.Oranges(0.5),
         'WassersteinDistance':plt.cm.Purples(0.5),
         'gmin':plt.cm.Greens(0.8),
         'gmax':plt.cm.Greens(0.6),
         'rtrough':plt.cm.Greens(0.4),
         'rpeak':plt.cm.Greens(0.2)}
        def build_legend(data):
            # Thanks to https://stackoverflow.com/questions/58718764/how-to-create-a-color-bar-using-a-dictionary-in-python
            """
            Build a legend for matplotlib plt from dict
            """
            from matplotlib.lines import Line2D
            legend_elements = []
            for key in data:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=key,
                                                markerfacecolor=data[key], markersize=40))
            return legend_elements
        cols_prettierForCaption = {'Count':'r',
         'PH: $H^1$ mean death value':plt.cm.Blues(1.0),
         'PH: $H^1$ mean persistence':plt.cm.Blues(0.9),
         'PH: $H^1$ mean birth value':plt.cm.Blues(0.8),
         'PH: $H^0$ mean death value':plt.cm.Blues(0.7),
         'PH: $H^0$ mean persistence':plt.cm.Blues(0.6),
         'PH: $H^0$ mean birth value':plt.cm.Blues(0.5),
         'PH: $H^1$ number of features':plt.cm.Blues(0.4),
         'PH: $H^0$ number of features':plt.cm.Blues(0.3),
         'QCM':plt.cm.Oranges(0.5),
         'Wasserstein Distance':plt.cm.Purples(0.5),
         'PCF: minimum value':plt.cm.Greens(0.8),
         'PCF: maximum value':plt.cm.Greens(0.6),
         'PCF: $r$ value at minimum':plt.cm.Greens(0.4),
         'PCF: $r$ value at maximum':plt.cm.Greens(0.2)}
        legend_elements = build_legend(cols_prettierForCaption)
        plt.figure(figsize=(7.5,11))
        plt.gca().legend(handles=legend_elements,loc='center')
        plt.gca().set_axis_off()
        plt.show()
        plt.savefig('./Figure4/For SI/MDI_colorbar.png')
        plt.savefig('./Figure4/For SI/MDI_colorbar.svg')


        colors = []
        for v in f_trunc:
            for key in cols:
                if key in v[0]:
                    colors.append(cols[key])
        plt.figure(figsize=(18,18))
        plt.barh([x[0] for x in f_trunc],[x[1] for x in f_trunc],color=colors,xerr=[MDI_sd[x[0]] for x in f_trunc])
        plt.tight_layout()
        plt.xlabel('MDI')
        plt.title(f'Mean classifier score: {meanscore:.3f}')
        plt.savefig(f'./Figure4/For SI/MDI_{sampleID}_top{nToPlot}.png')
        plt.savefig(f'./Figure4/For SI/MDI_{sampleID}_top{nToPlot}.svg')
        
        #%
        features = [v[0] for v in sorted_MDI_mean]
    
        celltypeA, celltypeB = [], []
        pairs = []
        methods = []
        MDIs = []
        for v in features:
            parts = v.split('_')
            if parts[0] == 'PCF':
                pair = parts[2]
                methods.append(parts[1])
            elif parts[0] == 'Count':
                pair = '-'.join([parts[1],parts[1]])
                methods.append(parts[0])
            else:
                pair = parts[1]
                if parts[0] == 'TDA':
                    methods.append(parts[-1])
                else:
                    methods.append(parts[0])
                
            celltypeA.append(pair.split('-')[0])
            celltypeB.append(pair.split('-')[1])
            MDIs.append(MDI_mean[v])
            pairs.append(pair)
            
        methodArray = np.array(methods)
        celltypeA = np.array(celltypeA)
        celltypeB = np.array(celltypeB)
        MDIs = np.array(MDIs)
        pairs = np.array(pairs)
            
        # Find top nToPlot cell-cell pairs, and then plot the contribution to each pair of each method
        bigList = {}
        totalMDIs = {}
        for pair in np.unique(pairs):
            mask = pairs == pair
            methods = methodArray[mask]
            MDI = MDIs[mask]
            lookup = {}
            for i in range(len(MDI)):
                lookup[methods[i]] = MDI[i]
            bigList[pair] = lookup
            totalMDIs[pair] = np.sum(MDI)
        sorted_totalMDIs = sorted(totalMDIs.items(), key=lambda x:x[1])
    
        nToPlot = 10
        pairsToPlot = sorted_totalMDIs[-nToPlot:]
        pairsToPlot = np.flipud(pairsToPlot )

        plt.figure(figsize=(18,nToPlot+2))
        for thing in pairsToPlot:
            pair = thing[0]
            vals = bigList[pair]
            # Sort vals alphabetically by keys
            methods = sorted(vals.keys(), key=lambda x:x[0])
            
            bottom = 0
            for method in methods:
                MDI = vals[method]
                plt.gca().barh(pair, MDI, label=pair, left=bottom,facecolor=cols[method])
                bottom += MDI
        plt.tight_layout()
        plt.xlabel('Cumulative MDI')
        plt.title(f'Mean classifier score: {meanscore:.3f}')
        plt.savefig(f'./Figure4/For SI/MDI_{sampleID}_byMethod.png')
        plt.savefig(f'./Figure4/For SI/MDI_{sampleID}_byMethod.svg')
