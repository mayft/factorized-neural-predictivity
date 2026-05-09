from brainscore_vision.benchmarks import Benchmark
from brainscore_vision.benchmark_helpers.neural_common import average_repetition
from brainscore_vision.model_interface import BrainModel
from brainscore_vision import load_metric, load_dataset
from brainscore_vision.model_helpers.brain_transformation import LayerScores
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper,load_preprocess_images
from brainscore_vision.metrics.regression_correlation import CrossRegressedCorrelation, pls_regression,pearsonr_correlation
from brainscore_core.supported_data_standards.brainio.assemblies import merge_data_arrays

import functools
from torchvision.models import resnet18, alexnet,vit_b_16,resnet50,vit_l_16
import numpy as np
import pandas as pd
import xarray as xr
import sys

consistent_pcs = {
    'IT': [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19, 20, 21, 22, 23, 26, 30, 34],
    'V4': [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,19, 20, 21, 24, 26],
    'V2': [0, 1, 2],
    'V1': [0, 1, 2, 4, 6],
}

#features 5 and 12: max pool, features 7 and 9 and classifier 2 and 5: relu
alexnet_layers = ['features.5', 'features.7', 'features.9', 'features.12', 'classifier.2', 'classifier.5']
# first basic block from each conv block
resnet_layers = ['layer1.0', 'layer2.0', 'layer3.0', 'layer4.0']
# output of every second transformer block
vit_layers = [f'encoder.layers.encoder_layer_{i}.mlp' for i in range(0,12,2)]
vit_l_layers=[f'encoder.layers.encoder_layer_{i}' for i in range(0,24,4)]

models = {
    'resnet18': ('resnet18',resnet18,resnet_layers,True),
    'alexnet': ('alexnet',alexnet,alexnet_layers,True),
    'vit': ('vit',vit_b_16,vit_layers,True),
    'resnet18_untrained': ('resnet18_untrained',resnet18,resnet_layers,False),
    'alexnet_untrained': ('alexnet_untrained',alexnet,alexnet_layers,False),
    'vit_untrained':('vit_untrained',vit_b_16,vit_layers,False),
    'resnet50':('resnet50',resnet50,resnet_layers,True),
    'vit_l':('vit_l',vit_l_16,vit_l_layers,True),
    'resnet50_untrained':('resnet50_untrained',resnet50,resnet_layers,False),
    'vit_l_untrained':('vit_l_untrained',vit_l_16,vit_l_layers,False),
}

metrics = ['pls_cv','ridge_cv']

regions = {    
    'V1': ('FreemanZiemba2013.public', 'V1'),
    'V2': ('FreemanZiemba2013.public', 'V2'),
    'V4': ('MajajHong2015.public', 'V4'),
    'IT': ('MajajHong2015.public', 'IT'),
}



class Model:
    def __init__(self,identifier,model,layers,trained,image_size=224,device='mps'):
        self.identifier = identifier
        self.weights = 'IMAGENET1K_V1' if trained else None
        self.layers = layers
        self.activations = self._get_model(identifier,model,self.weights,image_size,device)
    
    def _get_model(self,identifier,model,weights,image_size,device):
        activation_model = model(weights=weights)
        preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
        wrapper = PytorchWrapper(identifier=identifier, model=activation_model, preprocessing=preprocessing)
        wrapper.image_size = image_size
        #device=torch.device('mps')
        #wrapper._device = device
        #wrapper._model.to(device)
        return wrapper

def get_dataset(identifier,region=None,as_pcs=False,average=True):
    data = load_dataset(identifier)
    data = data.transpose('presentation', 'neuroid', 'time_bin')

    if data.time_bin.size != 1:
        data = data.mean(dim='time_bin', keep_attrs=True)
    else:
        data = data.squeeze('time_bin')

    if as_pcs:
        data = average_repetition(data)
        U, S, Vt = np.linalg.svd(data, full_matrices=False)
        data.values = U * S
        data['component'] = 'neuroid', range(len(data['neuroid']))
        #elf.pcs = self.pcs.isel(neuroid=consistent)

    if region is not None:
        data = data.sel(region=region)
        if data.dims[1] != 'neuroid':
            data = data.stack(neuroid=['neuroid_id'])
        data['region'] = 'neuroid', [region] * len(data['neuroid'])

    data.load()
    if not average:
        return data
    
    data_avg = average_repetition(data)
    return data_avg

def to_pcs(activations):
    X = activations - np.mean(activations,axis=0)
    U, S, Vt = np.linalg.svd(X,full_matrices=False)
    exp_var = (S**2) / (activations.shape[0] - 1)
    pcs = U*S
    return pcs,exp_var

def apply_pca(dataset):
    data = dataset.copy().sortby('neuroid_id','stimulus_id')
    data, eigens = xr.apply_ufunc(to_pcs,data,input_core_dims=[['presentation','neuroid']],output_core_dims=[['presentation','neuroid'],['neuroid']],keep_attrs=True)
    data['pc'] = 'neuroid',  range(len(data['neuroid']))
    data['eigenvalues'] = eigens
    return data

class PCARidgeRegression:
    def __init__(self,assembly,consistent):
        cvk = {'stratification_coord': None}
        self._regression = load_metric('ridge_cv',crossvalidation_kwargs=cvk)
        self.pcs = assembly.copy()
        self.consistent = consistent
        U, S, Vt = np.linalg.svd(self.pcs, full_matrices=False)
        self.pcs.values = U*S
        self.pcs['component'] = 'neuroid', range(len(self.pcs['neuroid']))
        self.pcs = self.pcs.isel(neuroid=consistent)

    def __call__(self, source, target):
        return self._regression(source,self.pcs)

class MyBenchmark(Benchmark):
    def __init__(self,identifier,metric,assembly,region,visual_degrees=8,num_trials=1,consistent=None,pcs=False,n=None):
        self._assembly = assembly.copy()
        self.region = region
        cvk = {'stratification_coord':None}
        self._identifier = f'{identifier}_{metric}_{region}'
        #if metric == 'pcr_cv2' or metric == 'pcr_cv' or metric == 'pcr_cv3':
        #    self._metric = PCARidgeRegression(assembly,consistent)
        #else: 
        if metric == 'pls_cv':
            self._metric = CrossRegressedCorrelation(regression=pls_regression(regression_kwargs={'n_components':n,'max_iter':5000}),
                                               correlation=pearsonr_correlation(),crossvalidation_kwargs=cvk)
            self._identifier=f'{self._identifier}_{n}'    
        else:
            self._metric = load_metric(metric,crossvalidation_kwargs=cvk)
        self._visual_degrees=visual_degrees
        self._number_of_trials = num_trials
        if pcs:
            self._assembly = apply_pca(self._assembly)
            self._assembly['component'] = 'neuroid', range(len(self._assembly['neuroid']))
            if consistent is not None:
                self._assembly = self._assembly.isel(neuroid=consistent)
                self._identifier = f'{self._identifier}_pca'
            else:
                self._identifier = f'{self._identifier}_full_pca'

    @property
    def identifier(self):
        return self._identifier

    def __call__(self, model: BrainModel):
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=model.visual_degrees(), source_visual_degrees=self._visual_degrees)
        model.start_recording(recording_target=self.region,time_bins=[(70,170)])
        source_assembly = model.look_at(stimulus_set, number_of_trials=self._number_of_trials)

        if 'time_bin' in source_assembly.dims and source_assembly.sizes['time_bin'] == 1:
            source_assembly = source_assembly.squeeze('time_bin')

        return self._metric(source_assembly, self._assembly)
    

def get_layer_scores(scores_data):
    benchmarks = {}
    for x,y in scores_data.items():
        scored = xr.DataArray.to_pandas(y.raw.stack(z=('layer','split')))
        scored = scored.melt(ignore_index=False).reset_index()
        components = scored['neuroid_id'].unique()
        pc_map = dict(zip(components,range(len(components))))
        scored['pc'] = scored['neuroid_id'].map(pc_map)
        benchmarks[x] = scored
    return pd.concat(benchmarks).reset_index(drop=True)

def evaluate(model_id,region,metric,n=None,pcs=False,filter_pcs=True):
    dataset=get_dataset(*regions[region])
    model=Model(*models[model_id])
    consistent = consistent_pcs[region] if filter_pcs else None

    model_scorer = LayerScores(model_identifier=model.identifier,activations_model=model.activations,visual_degrees=8)
    benchmark = MyBenchmark(identifier='benchmark', assembly=dataset, region=region,metric=metric,consistent=consistent,pcs=pcs,n=n)
    return model_scorer(benchmark,model.layers)


if __name__ == "__main__":    
    jobs = [
        ('vit','V1','pls_cv',10,False),
        ('vit','V2','pls_cv',10,False),
        ('resnet18','V1','pls_cv',10,False),
        ('resnet18','V2','pls_cv',10,False),
        ('vit_untrained','V1','pls_cv',10,False),
        ('vit_untrained','V2','pls_cv',10,False),
        ('resnet18_untrained','V1','pls_cv',10,False),
        ('resnet18_untrained','V2','pls_cv',10,False),
        ('vit','V1','ridge_cv',None,False),
        ('vit','V2','ridge_cv',None,False),
        ('resnet18','V1','ridge_cv',None,False),
        ('resnet18','V2','ridge_cv',None,False),
        ('vit_untrained','V1','ridge_cv',None,False),
        ('vit_untrained','V2','ridge_cv',None,False),
        ('resnet18_untrained','V1','ridge_cv',None,False),
        ('resnet18_untrained','V2','ridge_cv',None,False),
        ('vit','V1','ridge_cv',None,True),
        ('vit','V2','ridge_cv',None,True),
        ('resnet18','V1','ridge_cv',None,True),
        ('resnet18','V2','ridge_cv',None,True),
        ('vit_untrained','V1','ridge_cv',None,True),
        ('vit_untrained','V2','ridge_cv',None,True),
        ('resnet18_untrained','V1','ridge_cv',None,True),
        ('resnet18_untrained','V2','ridge_cv',None,True),

    ]
    print('staring')
    for arg in sys.argv[1:]:
        args=jobs[int(arg)]
        name = f'{args[0]}_{args[1]}_{args[2]}'+(f'_{args[3]}'if args[3] else '')+('_pcs'if args[4] else '')
        print(f'\n\nevaluating {name}')
        scores= {'':evaluate(*args)}
        get_layer_scores(scores).to_csv(f'prediction_results/{name}.csv')
        if args[4]:
            print(f'\nevaluating {name} full')
            scores= {'':evaluate(*args,filter_pcs=False)}
            get_layer_scores(scores).to_csv(f'prediction_results/{name}-full.csv')
            print('')

