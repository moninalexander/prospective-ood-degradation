import torch
import json
from pathlib import Path
import rich


TRAIN_METRICS = {
    'train_time_stamps',
    'train_loss',
    'train_accuracy',
    'grad_norm',
    'param_norm',
    'delta_local_param_norm',
    'delta_global_param_norm',
    'rel_delta_global_param_norm',
    'initial_parameters_norm',
    'confidence',
    'confidence_correct',
    'confidence_wrong',
    'entropy',
}

COLLECT_METRICS = {
    'collect_time_stamps',
    'source_eval_loss',
    'clean_eval_loss',
    'source_eval_accuracy',
    'clean_eval_accuracy',
    'source_test_loss',
    'source_test_accuracy',
    'clean_test_loss',
    'clean_test_accuracy',
    'degradation_loss',
    'degradation_accuracy',
}



def can_save_file(save_path):

    save_path = Path(save_path)

    if save_path.exists():
        rich.print(
            f'\t[red]File already exists. Not saved: '
            f'{save_path.name}'
        )
        return False

    return True







def load_experiment(path):
    metrics=torch.load(path/'metrics.pt')
    with open(path/'config.json') as file:
        config=json.load(file)
    return metrics, config

def get_metric(metrics, metric_name):

    if metric_name == 'degradation_loss':
        return (
            torch.tensor(metrics['clean_eval_loss'])
            - torch.tensor(metrics['source_eval_loss'])
        )

    if metric_name == 'degradation_accuracy':
        return (
            torch.tensor(metrics['source_eval_accuracy'])
            - torch.tensor(metrics['clean_eval_accuracy'])
        )

    return torch.tensor(metrics[metric_name])

def load_experiments(paths):

    experiments = {}

    for path in paths:
        metrics, config = load_experiment(path)

        experiments[path] = {
            'metrics': metrics,
            'config': config,
        }

        experiments[path]['metrics']['degradation_accuracy']=get_metric(metrics,'degradation_accuracy')
        experiments[path]['metrics']['degradation_loss']=get_metric(metrics,'degradation_loss')

    return experiments

def is_stalled(metrics, threshold):
    source_eval_accuracy=torch.tensor(metrics['source_eval_accuracy'])

    return source_eval_accuracy.max().item()<=threshold





def get_accuracy_degradation_time(
    metrics,
    threshold,
    persistence_window,
    persistence_required): # Persistent degradation for persistence_length steps
    
    degradation_accuracy=get_metric(metrics,'degradation_accuracy')
    time_stamps=get_time_stamps(metrics,'degradation_accuracy')
    degradation_gap=torch.abs(degradation_accuracy)
    
    for start_index in range(len(degradation_gap)-persistence_window+1):
        if degradation_gap[start_index]<=threshold:
            continue
        
        persistence_region=degradation_gap[start_index:start_index+persistence_window]
        num_above_threshold=(persistence_region>threshold).sum().item()
        if num_above_threshold>=persistence_required:
            return time_stamps[start_index].item()
    return float('inf')



def get_time_stamps(metrics, metric_name):

    if metric_name in TRAIN_METRICS:
        return torch.tensor(metrics['train_time_stamps'])

    if metric_name in COLLECT_METRICS:
        return torch.tensor(metrics['collect_time_stamps'])

    raise ValueError(f'Unknown metric: {metric_name}')



def get_seed_folders(path: Path):

    return [subfolder for subfolder in path.iterdir() if subfolder.is_dir()]





def has_valid_pre_degradation_span(
    metrics,
    degradation_time,
    min_accuracy, 
    max_deviation, 
    min_length,
    required_fraction,
):
    source_eval_accuracy=get_metric(metrics,'source_eval_accuracy')
    degradation_accuracy=get_metric(metrics,'degradation_accuracy')
    time_stamps=get_time_stamps(metrics,'degradation_accuracy')
    degradation_index=torch.where(time_stamps==degradation_time)[0].item()
       
    if degradation_index < min_length:
        return False
    
    
    start_index=degradation_index-min_length
    source_acc_pre=source_eval_accuracy[start_index:degradation_index]
    
    # Mandatory condition: source performance must be adequate throughout the pre-degradation span.
    if not (source_acc_pre > min_accuracy).all().item():
        return False
    
    deg_acc_pre=degradation_accuracy[start_index:degradation_index]
    fraction_withn_band=(torch.abs(deg_acc_pre)<max_deviation).float().mean().item()
    if fraction_withn_band<required_fraction:
        return False
    
    
    return True





def classify_experiments(
    experiments,
    degradation_threshold,
    stalled_threshold,
    min_accuracy,
    max_deviation,
    min_length,
    persistence_window,
    persistence_required,
    required_fraction
    ):
        degradation_times = {}
        finite_degradation_times={}
        
        stalled_paths=[]
        no_degradation_paths=[]
        prediction_paths=[]
        no_sustained_pre_degradation_paths=[]

        for path, experiment in experiments.items():
            metrics=experiment['metrics']
            degradation_time=get_accuracy_degradation_time(
                metrics,
                threshold=degradation_threshold,
                persistence_window=persistence_window,
                persistence_required=persistence_required,)
            degradation_times[path]=degradation_time
            
            if is_stalled(metrics, threshold=stalled_threshold):
                stalled_paths.append(path)
            elif degradation_time==float('inf'):
                no_degradation_paths.append(path)
            elif has_valid_pre_degradation_span(
                metrics,
                degradation_time,
                min_accuracy,
                max_deviation, 
                min_length,
                required_fraction,
            ):
                prediction_paths.append(path)
                finite_degradation_times[path]=degradation_time
            else:
                no_sustained_pre_degradation_paths.append(path)

        return prediction_paths, no_degradation_paths, no_sustained_pre_degradation_paths, stalled_paths, degradation_times, finite_degradation_times
