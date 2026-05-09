
import pandas as pd
import xarray as xr

import functools
import torch
import torchvision.models
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.brain_transformation.neural import PreRunLayers

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pytorch_pretrained_vit import ViT

import brainscore_vision
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision import Score
from brainscore_vision.benchmark_helpers.neural_common import average_repetition
from brainscore_vision.model_interface import BrainModel
from brainscore_vision import load_metric, load_dataset, load_model, load_ceiling, load_stimulus_set
from brainscore_vision.model_helpers.brain_transformation import LayerScores, LayerMappedModel, TemporalAligned
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.metrics.regression_correlation import CrossRegressedCorrelation

from brainscore_vision.metrics.regression_correlation import  linear_regression, pearsonr_correlation
from brainscore_vision.model_helpers.brain_transformation.neural import PreRunLayers

'''
4 regions
2 sets

3 metrics
8 datasets
32 layers across all models


(64 activation recordings)
3 models
2 weights
'''

def get_dataset(identifier):
    data = load_dataset(identifier)
    data = data.transpose('presentation', 'neuroid', 'time_bin')
    if data.time_bin.size != 1:
        data = data.mean(dim='time_bin', keep_attrs=True)
    else:
        data = data.squeeze('time_bin')
    data.load()
    return data

def to_pcs(activations):
    X = activations
    U, S, Vt = np.linalg.svd(X,full_matrices=False)
    pcs = U*S
    return pcs

def get_region_data(dataset, region):
    data = dataset.sel(region=region)
    if data.dims[1] != 'neuroid':
        data = data.stack(neuroid=['neuroid_id'])

    data['region'] = 'neuroid', [region] * len(data['neuroid'])
    data_avg = average_repetition(data)
    data_avg = data_avg.sortby('stimulus_id').sortby('neuroid_id')

    pcs = data_avg.copy()
    pcs.values = to_pcs(pcs.values)
    return data_avg, pcs,data

class MyBenchmark(BenchmarkBase):
    def __init__(self,identifier, assembly,region,pca=False,visual_degrees=8,num_trials=1,timebins=[(70,170)], **kwargs):
        super(MyBenchmark, self).__init__(identifier=identifier, **kwargs)
        self._assembly = assembly
        self.region = region
        cvk = {'stratification_coord':None}
        self._metric = CrossRegressedCorrelation(regression=linear_regression(), correlation=pearsonr_correlation(),crossvalidation_kwargs=cvk)
        self._visual_degrees=visual_degrees
        self._number_of_trials = num_trials
        self._pcs = assembly.copy()
        self._pcs.values = to_pcs(self._pcs.values)
        self.pca = pca
        self.timebins = timebins

    @property
    def identifier(self):
        return "linear_regression_"+self.region

    def __call__(self, model: BrainModel) -> Score:
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=model.visual_degrees(), source_visual_degrees=self._visual_degrees)

        model.start_recording(recording_target=self.region)
        source_assembly = model.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        if 'time_bin' in source_assembly.dims and source_assembly.sizes['time_bin'] == 1:
            source_assembly = source_assembly.squeeze('time_bin')

        raw_scores = []
        for layer in np.unique(source_assembly.layer):
            raw_score = self._metric(source_assembly.sel(layer=layer), self._assembly)
            raw_score = raw_score.expand_dims('layer')
            raw_score['layer'] = [layer]
            raw_scores.append(raw_score)
        raw_scores = Score.merge(*raw_scores)
        pc_scores = []
        if self.pca:
            for layer in np.unique(source_assembly.layer):
                pc_score = self._metric(source_assembly.sel(layer=layer), self._pcs)
                pc_score = pc_score.expand_dims('layer')
                pc_score['layer'] = [layer]
                pc_scores.append(pc_score)
            pc_scores = Score.merge(*pc_scores)
        #ceiled_score = raw_score / self.ceiling
        return raw_scores, pc_scores

def get_scores(activations_model,layers,regions,model_degrees):
    region_layer_dict = {region:layers for region in regions.keys()}
    model = LayerMappedModel(identifier=f"{activations_model.identifier}_layers", visual_degrees=model_degrees, activations_model=activations_model,region_layer_map=region_layer_dict)
    device = torch.device('mps')
    model.activations_model._device=device
    model.activations_model._model.to(device)
   # model.layer_model.activations_model._device = device
    model = PreRunLayers(model=model.activations_model, layers=layers, forward=model)
    scores = {}
    pc_scores = {}
    for region, benchmark in regions.items():
        scores[region],pc_scores[region] = benchmark(model)
    return scores,pc_scores

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

def get_model(model,identifier,image_size):
    device = torch.device('mps')
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing)
    wrapper._device = device
    wrapper._model.to(device)
    wrapper.image_size = image_size
    return wrapper


if __name__ == "__main__":

    majaj_data = get_dataset('MajajHong2015.public')
    freeman_data = get_dataset('FreemanZiemba2013.public')

    it_data, it_pcs, it_full = get_region_data(majaj_data, 'IT')
    v4_data, v4_pcs, v4_full = get_region_data(majaj_data, 'V4')
    v2_data, v2_pcs, v2_full = get_region_data(freeman_data, 'V2')
    v1_data, v1_pcs, v1_full = get_region_data(freeman_data, 'V1')

    ceiler = load_ceiling('internal_consistency')
    it_benchmark = MyBenchmark(identifier='it_bench', assembly=it_data, region='IT',ceiling_func=lambda: ceiler(it_full), version=1)
    v4_benchmark = MyBenchmark(identifier='v4_bench', assembly=v4_data, region='V4',ceiling_func=lambda: ceiler(v4_full), version=1)
    v2_benchmark = MyBenchmark(identifier='v2_bench', assembly=v2_data, region='V2',ceiling_func=lambda: ceiler(v2_full), version=1, timebins=[(50, 200)],visual_degrees=4)
    v1_benchmark = MyBenchmark(identifier='v1_bench', assembly=v1_data, region='V1',ceiling_func=lambda: ceiler(v1_full), version=1, timebins=[(50, 200)],visual_degrees=4)

    alexnet_layers = ['features.5', 'features.7', 'features.9', 'features.12', 'classifier.2', 'classifier.5']
    resnet_layers = ['layer1.0', 'layer2.0', 'layer3.0', 'layer4.0']
    vit_layers=[f'vit.encoder.layer.{i}.layernorm_after' for i in range(1)]

    models = {}

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    vmodel = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    models['alexnet_untrained'] = (get_model(torchvision.models.alexnet(weights=None), identifier='alexnet_untrained', image_size=224),alexnet_layers)
    models['alexnet'] = (get_model(torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1), identifier='alexnet', image_size=224),alexnet_layers)
    models['resnet18'] = (load_model('resnet18_imagenet21kP').activations_model,resnet_layers)
    models['resnet18_untrained'] = (load_model('resnet18_random').activations_model,resnet_layers)
    models['vit'] = (get_model(vmodel,'vit-trained',224),vit_layers)#= (load_model('ViT_L_32_imagenet1k').activations_model,vit_layers)
    vit_model =  ViT('L_32_imagenet1k', pretrained=False)
    models['vit_untrained'] = (get_model(vit_model, identifier='vit', image_size=vit_model.image_size[0]),vit_layers)

    for name, model in models.items():
        scores_majaj, pc_majaj = get_scores(activations_model=model[0], layers=model[1],
                                            regions={'V4': v4_benchmark, 'IT': it_benchmark}, model_degrees=8)
        scores_freeman, pc_freeman = get_scores(activations_model=model[0], layers=model[1],
                                                regions={'V1': v1_benchmark, 'V2': v2_benchmark},model_degrees=8)
        scores = scores_freeman | scores_majaj
        #pc_scores = pc_freeman | pc_majaj
        results = get_layer_scores(scores)
        #pc_results =  get_layer_scores(pc_scores)
        results.to_csv(f'{name}--scores-raw.csv')
        #pc_results.to_csv(f'{name}-pc-scores-raw.csv')
