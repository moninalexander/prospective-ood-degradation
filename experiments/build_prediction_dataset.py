import torch
from pathlib import Path
import rich
from definitions.utils import (
    load_experiments, 
    get_seed_folders,
    classify_experiments,
    get_metric,
    TRAIN_METRICS,
    COLLECT_METRICS,
    can_save_file,
)
from configs.local_config import EXPERIMENTS_DIR


STALLED_THRESHOLD=0.54
DEGRADATION_THRESHOLD=0.05 # Operational definition of the degradation time
MAX_DEVIATION=0.05 # Optional pre-degradation stability criterion


MIN_ACCURACY=0.75

PERSISTENCE_WINDOW = 90
PERSISTENCE_REQUIRED = 81

WINDOW_SIZE = 100
PREDICTION_HORIZON = 200
PRE_DEGRADATION_SPAN = WINDOW_SIZE+2*PREDICTION_HORIZON

PRE_DEGRADATION_REQUIRED_FRACTION = 0


SPLIT_SEED=7
TRAIN_FRACTION = 0.8


PREDICTOR_METRICS = [
    'train_loss',
    'train_accuracy',
    'grad_norm',
    'param_norm',
    'delta_local_param_norm',
    'delta_global_param_norm',
    'confidence',
    'confidence_correct',
    'entropy',
]
WINDOW_FEATURES = [
    'current',
    'mean',
    'std',
    'slope',
]


BASE_FOLDER=EXPERIMENTS_DIR/'pc_1_pw_0_sh_1_sw_1'
BASE_FOLDER_CNN=BASE_FOLDER/'cnn_c1_3_c2_3_k1_2_p1_2_k2_2_p2_2_h_4'
BASE_FOLDER_MLP=BASE_FOLDER/'mlp_h1_4_h2_2'

RUN_NAMES_CNN=[
    'cnn_shortcut_run',
]
RUN_NAMES_MLP=[
    'mlp_shortcut_run',
]


def get_window_metrics(metric_long,time_stamps):
    window_metrics={}
    window_metrics['current']=metric_long[-1]
    metric_long=torch.tensor(metric_long)
    length=len(metric_long)
    time_stamps=time_stamps
    window_metrics['mean']=metric_long.mean().item()
    window_metrics['std']=metric_long.std().item()
    window_metrics['slope']=(torch.cov(torch.stack([metric_long,time_stamps]))[0,1]/torch.var(time_stamps)).item()
    return window_metrics
    
def to_labeled_feature_windows(metrics, look_back_time, degradation_time, window_size, prediction_horizon, predictor_metrics_names):
        windows={}

        train_time_stamps=metrics['train_time_stamps']
        train_indices={t:i for (i,t) in enumerate(train_time_stamps)}
        collect_time_stamps=metrics['collect_time_stamps']
        collect_indices={t:i for (i,t) in enumerate(collect_time_stamps)}

        t_min=max(degradation_time-look_back_time,1)
        t_in=t_min+window_size-1
        time_stamps = torch.arange(window_size, dtype=torch.float32)
        for t in range(t_in,degradation_time):
            metrics_t={}
            if t<degradation_time-prediction_horizon:
                label_t=0
            else:
                label_t=1

            start_window_time=t-window_size+1
            start_train_window_index=train_indices[start_window_time]
            end_train_window_index=train_indices[t]
            start_collect_window_index=collect_indices[start_window_time]
            end_collect_window_index=collect_indices[t]

            metrics_train={}
            metrics_collect={}
            
            for metric_name in predictor_metrics_names:
                
                if metric_name in TRAIN_METRICS:
                    metrics_train[metric_name]=get_window_metrics(metrics[metric_name][start_train_window_index:end_train_window_index+1],
                                                    time_stamps)
                
                elif metric_name in COLLECT_METRICS:
                    metrics_collect[metric_name]=get_window_metrics(metrics[metric_name][start_collect_window_index:end_collect_window_index+1],
                                                    time_stamps)

            windows[t]={'label':label_t,
                       'window_metrics': (metrics_train|metrics_collect)}
        
        return windows

def flatten_window_metrics(window_metrics,predictor_metrics,window_features):
    X=[]
    for metric_name in predictor_metrics:
        for feature_name in window_features:
            X.append(window_metrics[metric_name][feature_name])
    return torch.tensor(X,dtype=torch.float32)

def to_full_history_feature_windows(
    metrics,
    degradation_time,
    window_size,
    prediction_horizon,
    min_accuracy,
    predictor_metrics_names,
):
    windows = {}

    train_time_stamps = metrics['train_time_stamps']
    train_indices = {
        t: i
        for i, t in enumerate(train_time_stamps)
    }

    collect_time_stamps = metrics['collect_time_stamps']
    collect_indices = {
        t: i
        for i, t in enumerate(collect_time_stamps)
    }

    source_eval_accuracy = get_metric(
        metrics,
        'source_eval_accuracy',
    )

    time_stamps = torch.arange(
        window_size,
        dtype=torch.float32,
    )

    # Earliest endpoint for which a full feature window exists.
    t_in = max(
        train_time_stamps[0],
        collect_time_stamps[0],
    ) + window_size - 1

    # Degrading trajectory:
    # use all windows ending before degradation.
    if degradation_time != float('inf'):
        t_out = degradation_time - 1

    # No-observed-degradation trajectory:
    # the final H points are censored because we do not
    # observe a full prediction horizon after them.
    else:
        final_observed_time = min(
            train_time_stamps[-1],
            collect_time_stamps[-1],
        )

        t_out = (
            final_observed_time
            - prediction_horizon
        )

    for t in range(t_in, t_out + 1):

        start_window_time = t - window_size + 1

        start_train_window_index = train_indices[
            start_window_time
        ]
        end_train_window_index = train_indices[t]

        start_collect_window_index = collect_indices[
            start_window_time
        ]
        end_collect_window_index = collect_indices[t]

        # Only use windows in which the model has already
        # learned the source task adequately.
        source_acc_window = source_eval_accuracy[
            start_collect_window_index:
            end_collect_window_index + 1
        ]

        if not (
            source_acc_window > min_accuracy
        ).all().item():
            continue

        # Offline label.
        if degradation_time == float('inf'):
            label_t = 0
        elif t < degradation_time - prediction_horizon:
            label_t = 0
        else:
            label_t = 1

        metrics_train = {}
        metrics_collect = {}

        for metric_name in predictor_metrics_names:

            if metric_name in TRAIN_METRICS:
                metrics_train[metric_name] = (
                    get_window_metrics(
                        metrics[metric_name][
                            start_train_window_index:
                            end_train_window_index + 1
                        ],
                        time_stamps,
                    )
                )

            elif metric_name in COLLECT_METRICS:
                metrics_collect[metric_name] = (
                    get_window_metrics(
                        metrics[metric_name][
                            start_collect_window_index:
                            end_collect_window_index + 1
                        ],
                        time_stamps,
                    )
                )

        windows[t] = {
            'label': label_t,
            'window_metrics': (
                metrics_train | metrics_collect
            ),
        }

    return windows

def build_prediction_dataset(
    dataset_folder,
    experiments_all,
    train_paths,
    test_paths,
    finite_degradation_times,
    degradation_threshold,
    max_deviation,
    persistence_window,
    persistence_required,
    pre_degradation_required_fraction,
    min_accuracy,
    pre_degradation_span,
    window_size,
    prediction_horizon,
    predictor_metrics,
    window_features,
    split_seed,
    train_fraction,
):
    
    
    train_dataset_windowed={}
    for path in train_paths:
        windows=to_labeled_feature_windows(
            experiments_all[path]['metrics'],
            pre_degradation_span,
            finite_degradation_times[path],
            window_size,
            prediction_horizon,
            predictor_metrics
        )

        train_dataset_windowed[path]=windows

    test_dataset_windowed={}
    for path in test_paths:
        windows=to_labeled_feature_windows(
            experiments_all[path]['metrics'],
            pre_degradation_span,
            finite_degradation_times[path],
            window_size,
            prediction_horizon,
            predictor_metrics
        )

        test_dataset_windowed[path]=windows

    num_one_train=0
    num_zero_train=0

    for path in train_dataset_windowed:        
        for time in train_dataset_windowed[path]:
            if train_dataset_windowed[path][time]['label']==1:
                num_one_train+=1
            else:
                num_zero_train+=1
    
    
    num_one_test=0
    num_zero_test=0
    
    for path in test_dataset_windowed:        
            for time in test_dataset_windowed[path]:
                if test_dataset_windowed[path][time]['label']==1:
                    num_one_test+=1
                else:
                    num_zero_test+=1
    

    one_baseline_train=num_one_train/(num_one_train+num_zero_train)
    one_baseline_test=num_one_test/(num_one_test+num_zero_test)
    num_one=num_one_train+num_one_test
    num_zero=num_zero_train+num_zero_test
    one_baseline=num_one/(num_one+num_zero)
    rich.print(f'\t\tPositive windows: {num_one_train} {num_one_test} {num_one}')
    rich.print(f'\t\tNegative train windows: {num_zero_train} {num_zero_test} {num_zero}')
    rich.print(f'\t\tBaseline: {max(one_baseline_train,1-one_baseline_train):.2f} {max(one_baseline_test,1-one_baseline_test):.2f} [green]{max(one_baseline,1-one_baseline):.2f}')
    rich.print('-'*50)
    X_train=[]
    y_train=[]
    train_metadata={}
    
    for path in train_dataset_windowed:
        
        times = list(train_dataset_windowed[path].keys())

        if len(times) == 0:
            continue
                
        train_metadata[path]=times
        for time in train_dataset_windowed[path]:
            y_train.append(train_dataset_windowed[path][time]['label'])
            X_train.append(flatten_window_metrics(train_dataset_windowed[path][time]['window_metrics'],
                                                    predictor_metrics,
                                                    window_features))
    y_train=torch.tensor(y_train, dtype=torch.float32)
    X_train=torch.stack(X_train)
            
    
    X_test=[]
    y_test=[]
    test_metadata={}
    
    for path in test_dataset_windowed:
        
        times = list(test_dataset_windowed[path].keys())

        if len(times) == 0:
            continue
        test_metadata[path]=times
        
        for time in test_dataset_windowed[path]:
            y_test.append(test_dataset_windowed[path][time]['label'])
            X_test.append(flatten_window_metrics(test_dataset_windowed[path][time]['window_metrics'],
                                                    predictor_metrics,
                                                    window_features))
    y_test=torch.tensor(y_test, dtype=torch.float32)
    X_test=torch.stack(X_test)


    save_file_name = (
        f'PREDICTION'
        f'_DT{degradation_threshold}'
        f'_MD{max_deviation}'
        f'_PW{persistence_window}'
        f'_PR{persistence_required}'
        f'_PDF{pre_degradation_required_fraction}'
        f'_WF{len(window_features)}'
        f'_PDS{pre_degradation_span}'
        f'_WS{window_size}'
        f'_PH{prediction_horizon}'
        f'_SS{split_seed}.pt'
    )
    
    save_path=dataset_folder/save_file_name
    
    if not can_save_file(save_path):
        return None
    
    

    torch.save(
        {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,

            'train_paths': [
                str(path) for path in train_paths
            ],
            'test_paths': [
                str(path) for path in test_paths
            ],

            'train_metadata': {
                str(path): times
                for path, times in train_metadata.items()
            },
            'test_metadata': {
                str(path): times
                for path, times in test_metadata.items()
            },

            # Predictor inputs
            'predictor_metrics': predictor_metrics,
            'window_features': window_features,
            'window_size': window_size,
            'prediction_horizon': prediction_horizon,

            # Trajectory / degradation definition
            'degradation_threshold': degradation_threshold,
            'persistence_window': persistence_window,
            'persistence_required': persistence_required,

            # Pre-degradation trajectory selection
            'pre_degradation_span': pre_degradation_span,
            'max_deviation': max_deviation,
            'pre_degradation_required_fraction':
                pre_degradation_required_fraction,
            'min_accuracy': min_accuracy,

            # Dataset split
            'split_seed': split_seed,
            'train_fraction': train_fraction,
        },
        save_path,
    )

    rich.print(f'Dataset saved successfully to: [green]{save_file_name}\n')
    
    return train_paths, test_paths


# Builds full history for all datasets including the training
def build_full_history_dataset( 
    dataset_folder,
    experiments_all,
    prediction_paths,
    no_degradation_paths,
    degradation_times,
    degradation_threshold,
    max_deviation,
    persistence_window,
    persistence_required,
    pre_degradation_required_fraction,
    min_accuracy,
    pre_degradation_span,
    window_size,
    prediction_horizon,
    predictor_metrics,
    window_features,
):

    rich.print('=' * 50)
    rich.print('Building full-history evaluation dataset')
    rich.print('-' * 50)

    fullhistory_paths = (
        list(prediction_paths)
        + list(no_degradation_paths)
    )

    full_history_paths = {}
    
    num_paths = len(fullhistory_paths)

    for path_index, path in enumerate(
        fullhistory_paths,
        start=1,
    ):

        print(
        f'\rProcessing: {path_index}/{num_paths}',
        end='',
        flush=True,
        )

        metrics = experiments_all[path]['metrics']
        degradation_time = degradation_times[path]

        windows = to_full_history_feature_windows(
            metrics=metrics,
            degradation_time=degradation_time,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            min_accuracy=min_accuracy,
            predictor_metrics_names=predictor_metrics,
        )

        if len(windows) == 0:
            # print('No eligible windows.')
            # print('-' * 20)
            continue

        times = sorted(windows.keys())

        X = torch.stack(
            [
                flatten_window_metrics(
                    windows[t]['window_metrics'],
                    predictor_metrics,
                    window_features,
                )
                for t in times
            ]
        )

        y = torch.tensor(
            [
                windows[t]['label']
                for t in times
            ],
            dtype=torch.float32,
        )

        if degradation_time == float('inf'):

            path_type = 'no_observed_degradation'

            final_observed_time = min(
                metrics['train_time_stamps'][-1],
                metrics['collect_time_stamps'][-1],
            )

        else:

            path_type = 'degrading'
            final_observed_time = None

        full_history_paths[str(path)] = {
            'X': X,
            'y': y,
            'times': torch.tensor(
                times,
                dtype=torch.long,
            ),
            'path_type': path_type,
            'degradation_time': degradation_time,
            'final_observed_time': final_observed_time,
        }

        num_positive = int(y.sum().item())
        num_negative = len(y) - num_positive

        if path_type == 'no_observed_degradation':
            assert num_positive == 0

    save_file_name = (
        f'FULLHIST'
        f'_DT{degradation_threshold}'
        f'_MD{max_deviation}'
        f'_PW{persistence_window}'
        f'_PR{persistence_required}'
        f'_PDF{pre_degradation_required_fraction}'
        f'_WF{len(window_features)}'
        f'_PDS{pre_degradation_span}'
        f'_WS{window_size}'
        f'_PH{prediction_horizon}.pt'
    )

    save_path = dataset_folder / save_file_name
    
    if not can_save_file(save_path):
        return None
    
    

    torch.save(
        {
            'paths': full_history_paths,

            # Paths included in this evaluation dataset
            'prediction_paths': [
                str(path)
                for path in prediction_paths
            ],
            'no_degradation_paths': [
                str(path)
                for path in no_degradation_paths
            ],

            # Predictor inputs
            'predictor_metrics': predictor_metrics,
            'window_features': window_features,
            'window_size': window_size,
            'prediction_horizon': prediction_horizon,

            # Trajectory / degradation definition
            'degradation_threshold': degradation_threshold,
            'persistence_window': persistence_window,
            'persistence_required': persistence_required,

            # Pre-degradation trajectory selection
            'pre_degradation_span': pre_degradation_span,
            'max_deviation': max_deviation,
            'pre_degradation_required_fraction':
                pre_degradation_required_fraction,
            'min_accuracy': min_accuracy,
        },
        save_path,
    )
    print('\r')
    print('-' * 50)
    rich.print(f'Full-history evaluation dataset saved to: [green]{save_file_name}\n')

    return full_history_paths





########################################
############## Datasets construction
########################################






    
for run_name in RUN_NAMES_CNN+RUN_NAMES_MLP:
    
    if run_name in RUN_NAMES_CNN:
        base_folder=BASE_FOLDER_CNN
        print('='*50)
        rich.print(f'[yellow]CNN run name: {run_name}')  
        print('-'*50)
        rich.print(f'\tLoading...') 
        print('-'*50)
        
    elif run_name in RUN_NAMES_MLP:
        base_folder=BASE_FOLDER_MLP
        print('='*50)
        rich.print(f'[yellow]MLP run name: {run_name}')  
        print('-'*50)
        rich.print(f'\tLoading...') 
        print('-'*50)            
    
    dataset_folder=base_folder/run_name
    paths_all=get_seed_folders(dataset_folder)
    experiments_all=load_experiments(paths_all)
    
    print('\tLoaded successfully!')
    print('-'*50)
    
    rich.print('\tClassification...')
    print('-'*50)
    (   
        prediction_paths, 
        no_degradation_paths,
        no_sustained_pre_degradation_paths,
        stalled_paths, 
        degradation_times,
        finite_degradation_times,
        )=classify_experiments(
            experiments=experiments_all, 
            degradation_threshold=DEGRADATION_THRESHOLD,
            stalled_threshold=STALLED_THRESHOLD,
            min_accuracy=MIN_ACCURACY,
            max_deviation=MAX_DEVIATION,
            min_length=PRE_DEGRADATION_SPAN,
            persistence_window=PERSISTENCE_WINDOW,
            persistence_required=PERSISTENCE_REQUIRED,
            required_fraction=PRE_DEGRADATION_REQUIRED_FRACTION,
            )
    rich.print(f'\tSummary:')
    print('-'*50)
    rich.print(f'\t\tTotal number of experiments: {len(paths_all)}')    
    rich.print(f'\t\t[green]Prediction paths: {len(prediction_paths)}')
    rich.print(f'\t\tNo degradation paths: {len(no_degradation_paths)}')
    rich.print(f'\t\tFast degradation paths: {len(no_sustained_pre_degradation_paths)}')
    rich.print(f'\t\tStalled paths: {len(stalled_paths)}')
    rich.print(f'\t\tSumchecks: {len(stalled_paths)+len(no_sustained_pre_degradation_paths)+len(no_degradation_paths)+len(prediction_paths)==len(paths_all)}')
    rich.print(f'\t\tLength check: {len(finite_degradation_times)==len(prediction_paths)}')
    rich.print('-'*50)
    
    rich.print('\tBuilding prediction dataset')
    rich.print('-'*50)
    split_generator=torch.Generator().manual_seed(SPLIT_SEED)
    shuffled_indices=torch.randperm(len(prediction_paths),generator=split_generator)
    
    num_train=int(len(shuffled_indices)*TRAIN_FRACTION)
    prediction_paths_split=prediction_paths.copy()
    prediction_paths_split=[prediction_paths_split[index] for index in shuffled_indices]
    
    train_paths=prediction_paths_split[:num_train]
    test_paths=prediction_paths_split[num_train:]

    rich.print(f'\t\tTrain dataset size: {len(train_paths)}')
    rich.print(f'\t\tTest dataset size: {len(test_paths)}')
    print('-'*50)
    

# ---------------------------------------
# Balanced dataset used to train predictor
# ---------------------------------------

    build_prediction_dataset(
        dataset_folder,
        experiments_all,
        train_paths,
        test_paths,
        finite_degradation_times,
        DEGRADATION_THRESHOLD,
        MAX_DEVIATION,
        PERSISTENCE_WINDOW,
        PERSISTENCE_REQUIRED,
        PRE_DEGRADATION_REQUIRED_FRACTION,
        MIN_ACCURACY,
        PRE_DEGRADATION_SPAN,
        WINDOW_SIZE,
        PREDICTION_HORIZON,
        PREDICTOR_METRICS,
        WINDOW_FEATURES,
        SPLIT_SEED,
        TRAIN_FRACTION,
    )


# --------------------
# Full-history dataset 
# --------------------
    

    build_full_history_dataset(
                dataset_folder=dataset_folder,
                experiments_all=experiments_all,

                prediction_paths=prediction_paths,
                no_degradation_paths=no_degradation_paths,
                degradation_times=degradation_times,

                degradation_threshold=DEGRADATION_THRESHOLD,
                max_deviation=MAX_DEVIATION,
                persistence_window=PERSISTENCE_WINDOW,
                persistence_required=PERSISTENCE_REQUIRED,
                pre_degradation_required_fraction=
                    PRE_DEGRADATION_REQUIRED_FRACTION,
                min_accuracy=MIN_ACCURACY,
                pre_degradation_span=PRE_DEGRADATION_SPAN,

                window_size=WINDOW_SIZE,
                prediction_horizon=PREDICTION_HORIZON,
                predictor_metrics=PREDICTOR_METRICS,
                window_features=WINDOW_FEATURES,
            )