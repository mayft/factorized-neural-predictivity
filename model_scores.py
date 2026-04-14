from brainscore_vision.benchmarks import Benchmark
from brainscore_vision.benchmark_helpers.neural_common import average_repetition
from brainscore_vision.model_interface import BrainModel
from brainscore_vision import load_metric, load_dataset
from brainscore_vision.model_helpers.brain_transformation import LayerScores
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper,load_preprocess_images

import functools
from torchvision.models import resnet18, alexnet,vit_b_16
import numpy as np
import pandas as pd
import xarray as xr


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

def get_dataset(identifier,region=None,as_pcs=False):
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
    data, eigens = xr.apply_ufunc(to_pcs,data,input_core_dims=[['presentation','neuroid']],output_core_dims=[['presentation','neuroid'],['neuroid']])
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
        #self.pcs = self.pcs.isel(neuroid=consistent)

    def __call__(self, source, target):
        return self._regression(source,self.pcs)

class MyBenchmark(Benchmark):
    def __init__(self,identifier,metric,assembly,region,visual_degrees=8,num_trials=1,consistent=None,pcs=False):
        self._assembly = assembly
        self.region = region
        cvk = {'stratification_coord':None}
        #if metric == 'pcr_cv2' or metric == 'pcr_cv' or metric == 'pcr_cv3':
        #    self._metric = PCARidgeRegression(assembly,consistent)
        #else:
        self._metric = load_metric(metric,crossvalidation_kwargs=cvk)
        self._visual_degrees=visual_degrees
        self._number_of_trials = num_trials
        self._identifier = f'{identifier}_{metric}_{region}'
        if pcs:
            self._assembly = apply_pca(self.assembly)
            self._assembly = self._assembly.isel(neuroid=consistent)
            self._identifier = f'{identifier}_{metric}_{region}_pca'

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

if __name__ == "__main__":


    #features 5 and 12: max pool, features 7 and 9 and classifier 2 and 5: relu
    alexnet_layers = ['features.5', 'features.7', 'features.9', 'features.12', 'classifier.2', 'classifier.5']
    # first basic block from each conv block
    resnet_layers = ['layer1.0', 'layer2.0', 'layer3.0', 'layer4.0']
    # output of every second transformer block
    vit_layers = [f'encoder.layers.encoder_layer_{i}.mlp' for i in range(0,12,2)]

    models = {
        'resnet18': Model('resnet18',resnet18,resnet_layers,trained=True),
        'alexnet': Model('alexnet',alexnet,alexnet_layers,trained=True),
        'vit': Model('vit',vit_b_16,vit_layers,trained=True),
        'resnet18_untrained': Model('resnet18_untrained',resnet18,resnet_layers,trained=False),
        'alexnet_untrained': Model('alexnet_untrained',alexnet,alexnet_layers,trained=False),
        'vit_untrained':Model('vit_untrained',vit_b_16,vit_layers,trained=False),
    }

    metrics = ['ridge_cv','pls_cv']

    regions = {
        'IT': get_dataset('MajajHong2015.public', 'IT'),
        'V4': get_dataset('MajajHong2015.public', 'V4'),
        'V2': get_dataset('FreemanZiemba2013.public', 'V2'),
        'V1': get_dataset('FreemanZiemba2013.public', 'V1')
    }

    consistent_pcs = {
        'IT': [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19, 20, 23, 24, 25, 26],
        'V4': [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,17, 18, 21],
        'V2': [0, 1, 2],
        'V1': [0, 1, 2, 4, 6],
    }


    for name, model in models.items():
        model_scorer = LayerScores(model_identifier=name,activations_model=model.activations,visual_degrees=8)
        for metric in metrics:    
            prediction_scores = {}
            for region, dataset in regions.items():
                print (f'scoring {name} on {region} data with {metric}')
                benchmark = MyBenchmark(identifier='benchmark', assembly=dataset, region=region,metric=metric,consistent=consistent_pcs[region],pca=False)
                prediction_scores[region] = model_scorer(benchmark,model.layers)

            get_layer_scores(prediction_scores).to_csv(f'neural_prediction_results_{model.identifier}_{metric}.csv')
