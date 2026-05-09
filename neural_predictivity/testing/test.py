import functools
import torch
import torchvision.models

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images


import brainscore_vision
from brainscore_vision.benchmark_helpers.neural_common import average_repetition
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.model_helpers.brain_transformation import LayerScores, LayerMappedModel, TemporalAligned
from brainscore_vision.benchmark_helpers.screen import place_on_screen


def get_model(weight=torchvision.models.AlexNet_Weights.DEFAULT,identifier='alexnet'):
    model = torchvision.models.alexnet(weights=weight)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    print(wrapper._device)
    return wrapper

if __name__ == "__main__":
    neural_data = brainscore_vision.load_dataset("MajajHong2015.public")
    neural_data = neural_data.transpose('presentation', 'neuroid', 'time_bin')

    benchmark_data = neural_data.squeeze('time_bin')
    #benchmark_data['region'] = 'neuroid', ['IT'] * len(benchmark_data['neuroid'])
    benchmark_data.load()
    benchmark_data_avg = average_repetition(benchmark_data)

    benchmark_data_it = benchmark_data.sel(region='IT')
    benchmark_data_it['region'] = 'neuroid', ['IT'] * len(benchmark_data_it['neuroid'])
    benchmark_data_avg_it = average_repetition(benchmark_data_it)
    benchmark_data_avg_it = benchmark_data_avg_it.sortby('stimulus_id').sortby('neuroid_id')


    stimuli = benchmark_data_it.stimulus_set
    stimuli = place_on_screen(stimuli, target_visual_degrees=8, source_visual_degrees=8)

    untrained_activation = get_model(weight=None,identifier='alexnet-untrained')
    model = LayerMappedModel(identifier=f"alexnetlayers_untrained", visual_degrees=8, activations_model=untrained_activation, region_layer_map={'IT': 'features.12'})
    model.start_recording(BrainModel.RecordingTarget.IT)
    model_assembly = model.look_at(stimuli)

    trained_activation = get_model(identifier='alexnet-trained')
    model2 = LayerMappedModel(identifier=f"alexnetlayers", visual_degrees=8, activations_model=trained_activation, region_layer_map={'IT': 'features.12'})
    model2.start_recording(BrainModel.RecordingTarget.IT)
    model_assembly2 = model2.look_at(stimuli)


    path = stimuli.get_stimulus('8a72e2bfdb8c267b57232bf96f069374d5b21832')
    images = load_preprocess_images(image_filepaths=[path],image_size=224)
    images = torch.from_numpy(images)
    images = images.to(model.activations_model._device)

    layer1 = model.activations_model._model._modules.get('features')._modules.get('12')
    layer2 = model2.activations_model._model._modules.get('features')._modules.get('12')

    layer_results = []
    def hook_function(_layer, _input, output):
        layer_results.append(output)#.cpu().data.numpy())

    hook = layer1.register_forward_hook(hook_function)
    hook2 = layer2.register_forward_hook(hook_function)

    model.activations_model._model.eval()
    out = model.activations_model._model(images)

    model2.activations_model._model.eval()
    out2 = model2.activations_model._model(images)

    act1 = layer_results[0].cpu().data.numpy().flatten()
    act2 = layer_results[1].cpu().data.numpy().flatten()

    for i in range(500):
        print(f'{act1[i]}  {act2[i]}  {model_assembly[0].values[i]}  {model_assembly2[0].values[i]}')