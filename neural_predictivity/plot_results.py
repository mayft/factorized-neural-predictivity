from model_scores import get_dataset,apply_pca
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from brainscore_core.supported_data_standards.brainio.transform import subset
from brainscore_vision.benchmark_helpers.neural_common import average_repetition




models = ['resnet18','vit','resnet18_untrained','vit_untrained']
regions = ['V1','V2','V4','IT']
metrics = ['pls_cv_25','ridge_cv','pls_cv_10']
kinds = ['neurons','pcs','pcs-full']
def vit_rename_layers(row):
    return row['layer'][15:-4]

consistent_pcs = {
        'IT': [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19, 20, 21, 22, 23, 26, 30, 34],
        'V4': [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,19, 20, 21, 24, 26],
        'V2': [0, 1, 2],
        'V1': [0, 1, 2, 4, 6],
    }

datasets = {
        'IT': get_dataset('MajajHong2015.public', 'IT',average=False),
        'V4': get_dataset('MajajHong2015.public', 'V4',average=False),
        'V2': get_dataset('FreemanZiemba2013.public', 'V2',average=False),
        'V1': get_dataset('FreemanZiemba2013.public', 'V1',average=False)
    }

def reliability(dataset, splits=10, seed=125, pca=True,abs=False):
    split_scores = []
    rng = np.random.default_rng(seed=seed)
    coord_values = np.unique(dataset['repetition'].values)
    dims = dataset['repetition'].dims
    repetitions = xr.DataArray(coord_values, coords={'repetition': coord_values}, dims=['repetition']).stack(**{dims[0]: ('repetition',)})
    half = repetitions.shape[0]//2

    for i in range(splits):
        shuffle=rng.permutation(range(repetitions.shape[0]))
        half1 = repetitions.isel(presentation=shuffle[:half])
        half2 = repetitions.isel(presentation=shuffle[half:2*half])
        set1 = average_repetition(subset(dataset,half1,dims_must_match=False)).sortby('neuroid_id','stimulus_id')
        set2 = average_repetition(subset(dataset,half2,dims_must_match=False)).sortby('neuroid_id','stimulus_id')

        if pca:
            set1=apply_pca(set1).values
            set2=apply_pca(set2).values
            #set1,_=to_pcs(set1)
            #set2,_=to_pcs(set2)

        correlation = stats.pearsonr(set1,set2).correlation
        if pca and abs: correlation= np.abs(correlation)

        consistency = 2*correlation/(1+correlation)
        split_scores.append(consistency)

    return np.array(split_scores)

reliabilities = {r: reliability(d,pca=True,splits=10,abs=True) for r,d in datasets.items()}

def get_results():
    dataset = []
    for kind in kinds:
        kind_id = '' if kind == 'neurons' else f'_{kind}'
        k=0
        if kind=='pcs-full':
            k=1
        elif kind=='pcs':
            k=2
        for metric in metrics:
            for model_id in models:
                for region in regions:
                    data=pd.read_csv(f'prediction_results/{model_id}_{region}_{metric}{kind_id}.csv',index_col=0)
                    if model_id =='vit' or model_id=='vit_untrained':
                        data['layer'] = data.apply(vit_rename_layers,axis=1)
                    if model_id =='vit_untrained' or model_id=='resnet18_untrained':
                        data['trained'] ='untrained'
                    else:
                        data['trained']='trained'
                    data['model']=model_id
                    data['metric']=metric
                    data['type'] = kind
                    #data['ceiling'] = internal_consistencies[region][k]
                    #data['ceiled_value'] = data['value']/data['ceiling']
                    data['weight']=1

                    if kind != 'neurons':
                        def consistency(row):
                            return reliabilities[row['region']].mean(axis=0)[row['component']]
                        data['consistency'] = data.apply(consistency,axis=1)
                        data['gap']=data['consistency']-data['value']
                        
                        weighted = data.copy()
                        weighted['type'] = f'{kind}-weighted'
                        weighted['weight'] = weighted['eigenvalues']

                        #totals = {r: data[(data['region']==r)].groupby('component').first()['eigenvalues'].mean() for r in data['region'].unique()}
            
                        #def weighpcs(row):
                        #    return row['eigenvalues']#/totals[row['region']])
                        #weighted['weight']=weighted.apply(weighpcs,axis=1)
                        #weighted['value']=weighted['value']*weighted['weights']
                        #weighted['ceiled_value']=weighted['ceiled_value']*weighted['weights']
                        dataset.append(data)
                        dataset.append(weighted)
                    else:
                        data['component']=data['neuroid_id'].astype('category').cat.codes
                        dataset.append(data)

    dataset = pd.concat(dataset).reset_index(drop=True)
    dataset['weighted']=dataset['value']*dataset['weight']
    dataset['weighted-gap']=dataset['gap']*dataset['weight']
    return dataset

if __name__=='__main__':
    dataset = get_results()
    metrics=['pls_cv_25','ridge_cv','pls_cv_10']
    kinds = ['neurons','pcs','pcs-weighted','pcs-full','pcs-full-weighted']

    for metric in metrics:
        for kind in kinds:
            with sns.axes_style('whitegrid'):
                plt.figure()
                g=sns.catplot(dataset[(dataset['metric']==metric)&(dataset['type']==kind)],row='region',col='layer',x='component',y='weighted',hue='model',kind='strip',jitter=False,alpha=0.2,sharex=False,sharey='row',size=5)
                g.map_dataframe(sns.pointplot,x='component',y='weighted',hue='model',palette='tab10',linewidth=1.5,hue_order=['resnet18','vit','resnet18_untrained','vit_untrained'])
                if kind!= 'neurons':
                    g.map_dataframe(sns.lineplot,x='component',y='consistency',linestyle='--',alpha=0.6,color='#000053')
                g.figure.suptitle(f'{metric} {kind} prediction scores',y=1.02,size=24)
                plt.savefig(f'plots_v6/{metric}-{kind}-plot',dpi=100,bbox_inches='tight',)
                plt.close()


    for metric in metrics:
        for kind in kinds[1:]:
            with sns.axes_style('whitegrid'):
                plt.figure()
                g=sns.catplot(dataset[(dataset['metric']==metric)&(dataset['type']==kind)],row='region',col='layer',x='component',y='weighted-gap',hue='model',kind='strip',alpha=0.2,jitter=False,sharex=False,sharey='row',size=5)
                g.map_dataframe(sns.pointplot,x='component',y='weighted-gap',hue='model',palette='tab10',linewidth=1.5,hue_order=['resnet18','vit','resnet18_untrained','vit_untrained'])
                g.figure.suptitle(f'consistency-prediction gap for {metric} {kind}',y=1.02,size=24)
                plt.savefig(f'plots_v6/{metric}-{kind}-gap-plot',dpi=100,bbox_inches='tight',)
                plt.close()
