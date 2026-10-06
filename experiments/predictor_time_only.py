import torch
from pathlib import Path
import rich
import matplotlib.pyplot as plt
from definitions.models import LogisticRegression
from configs.local_config import EXPERIMENTS_DIR




if torch.cuda.is_available():
    device=torch.device('cuda')
elif torch.backends.mps.is_available():
    device=torch.device('mps')
else:
    device=torch.device('cpu')


PREDICTOR_SEED = 1
NUM_EPOCHS=7500
LEARNING_RATE=0.001



PLOT_LABELS = {
    "cnn_train_set: train": "Training run: predictor train",
    "cnn_train_set: test": "Training run: held-out test",
    "cnn_train_set": "Training run",
    "mlp_shortcut_run": "MLP",
}


def get_plot_label(run_name):
    return PLOT_LABELS.get(run_name, run_name)

def construct_file_path(base_folder,dataset_folder,base_file_name):
    return base_folder/dataset_folder/f'{base_file_name}'


def metadata_times_to_tensor(metadata):
    times = torch.tensor(
        [
            t
            for path_times in metadata.values()
            for t in path_times
        ],
        dtype=torch.float32,
    )

    return times.unsqueeze(1).to(device)


def path_times_to_tensor(times):
    return torch.as_tensor(
        times,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)



BASE_FOLDER=EXPERIMENTS_DIR/'pc_1_pw_0_sh_1_sw_1'
BASE_FOLDER_CNN=BASE_FOLDER/'cnn_c1_3_c2_3_k1_2_p1_2_k2_2_p2_2_h_4'
BASE_FOLDER_MLP=BASE_FOLDER/'mlp_h1_4_h2_2'

BASE_PREDICTION_FILENAME='PREDICTION_DT0.05_MD0.05_PW90_PR81_PDF0_WF4_PDS500_WS100_PH200_SS7.pt' # The dataset a model is trained on: used for training
BASE_FULLHIST_FILENAME = 'FULLHIST_DT0.05_MD0.05_PW90_PR81_PDF0_WF4_PDS500_WS100_PH200.pt' # The full history for all datasets including the one used for training


train_run_name='cnn_train_set'       
train_dataset_path=construct_file_path(BASE_FOLDER_CNN,'cnn_shortcut_run',BASE_PREDICTION_FILENAME)

test_dataset_paths = {
    'mlp_shortcut_run': construct_file_path(BASE_FOLDER_MLP,'mlp_shortcut_run',BASE_PREDICTION_FILENAME),
}

full_history_dataset_paths = {
    'cnn_train_set': construct_file_path(BASE_FOLDER_CNN,'cnn_shortcut_run',BASE_FULLHIST_FILENAME),
    'mlp_shortcut_run': construct_file_path(BASE_FOLDER_MLP,'mlp_shortcut_run',BASE_FULLHIST_FILENAME),
}


def train_predictor(
    X_train,
    y_train,
    X_eval,
    y_eval,
    num_epochs,
    learning_rate,
    predictor_seed,
    ):
    mu=X_train.mean(dim=0)
    sigma=X_train.std(dim=0)
    X_train_normalized=(X_train-mu)/sigma
    X_eval_normalized=(X_eval-mu)/sigma
    num_train, num_features=X_train.shape
    num_eval, _=X_eval.shape
    
    torch.manual_seed(predictor_seed)
    torch.cuda.manual_seed_all(predictor_seed)
        
    model=LogisticRegression(num_features).to(device)
    
    
    
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,betas=(0.9,0.999))
    J=torch.nn.BCEWithLogitsLoss()
    

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits=model(X_train_normalized).squeeze(dim=1)
        loss=J(logits,y_train)
        if epoch%100==0:
            with torch.no_grad():
                y_pred=logits>0
                logits_eval=model(X_eval_normalized).squeeze()
                y_pred_eval=logits_eval>0
                rich.print(f'\tEpoch: {epoch}, \t Loss: {loss:.4f},'
                        f'\t Accuracy: {(y_pred==y_train).sum()/num_train:.4f}'
                        f'\t Eval Loss: {J(logits_eval, y_eval):.4f}'
                        f'\t [green]Eval Accuracy: {(y_pred_eval==y_eval).sum()/num_eval:.4f}'
                        )
        loss.backward()
        optimizer.step()
    
    return model, mu, sigma


def evaluate_predictor(model,X,y,mu,sigma,loss_fn):

    num_windows, _ = X.shape
    X_normalized = (X - mu) / sigma

    with torch.no_grad():

        logits = model(X_normalized).squeeze(dim=1)
        y_pred = logits > 0
        loss=loss_fn(logits, y)
        accuracy=(y_pred == y).sum() / num_windows
        probabilities = torch.sigmoid(logits).cpu()
    
    return loss, accuracy, probabilities


def build_trajectory_curves(probabilities, metadata):

    trajectory_curves = {}
    
    index = 0
    
    for path, times in metadata.items():

        if len(times) == 0:
            continue
        
        num_windows = len(times)

        probs_path = probabilities[index:index + num_windows]

        degradation_time = max(times) + 1

        lead_times = torch.tensor(
            [degradation_time - t for t in times]
        )

        trajectory_curves[path] = {
            'lead_times': lead_times,
            'probabilities': probs_path,
        }

        index += num_windows

    assert index == len(probabilities)
    
    return trajectory_curves





def evaluate_full_history_dataset(
    model,
    full_history_data,
    mu,
    sigma,
    verbose=False,
):

    if 'paths' not in full_history_data:
        raise RuntimeError(
            "The loaded FULLHIST file does not contain the key 'paths'. "
            "It may have been overwritten by an older version of "
            "predictor_time_only.py. Regenerate this FULLHIST dataset "
            "before running the time-only predictor."
        )

    results = {}

    for path, path_data in full_history_data['paths'].items():

        X = path_times_to_tensor(path_data['times'])
        y = path_data['y'].to(device)

        assert len(X) == len(y)

        loss, accuracy, probabilities = evaluate_predictor(
            model,
            X,
            y,
            mu,
            sigma,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
        )

        y_cpu = y.cpu()
        predictions = probabilities > 0.5

        positive_mask = y_cpu == 1
        negative_mask = y_cpu == 0

        if positive_mask.any():
            positive_recall = (
                predictions[positive_mask]
                .float()
                .mean()
                .item()
            )
        else:
            positive_recall = None

        if negative_mask.any():
            negative_positive_rate = (
                predictions[negative_mask]
                .float()
                .mean()
                .item()
            )
        else:
            negative_positive_rate = None

        times = path_data['times']

        if path_data['path_type'] == 'degrading':
            degradation_time = path_data['degradation_time']
            lead_times = degradation_time - times
            pre_horizon_positive_rate = negative_positive_rate
            false_positive_rate = None

        else:
            degradation_time = None
            lead_times = None
            pre_horizon_positive_rate = None
            false_positive_rate = negative_positive_rate

        results[path] = {
            'path_type': path_data['path_type'],
            'times': times,
            'lead_times': lead_times,
            'probabilities': probabilities,
            'labels': y_cpu,
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'positive_recall': positive_recall,
            'pre_horizon_positive_rate': pre_horizon_positive_rate,
            'false_positive_rate': false_positive_rate,
            'degradation_time': degradation_time,
        }

        if verbose:
            rich.print(f'\t{Path(path).name}')
            rich.print(f"\t\ttype: {path_data['path_type']}")
            rich.print(f"\t\twindows: {len(y)}")
            if positive_recall is not None:
                rich.print(f"\t\tpositive recall: {positive_recall:.4f}")
            else:
                rich.print(f"\t\tpositive recall: {positive_recall}")

            if pre_horizon_positive_rate is not None:
                rich.print(
                    f"\t\tpre-horizon positive rate: "
                    f"{pre_horizon_positive_rate:.4f}"
                )

            if false_positive_rate is not None:
                rich.print(
                    f"\t\tfalse positive rate: "
                    f"{false_positive_rate:.4f}"
                )
            else:
                rich.print(f"\t\tfalse positive rate: {false_positive_rate}")

    return results





def summarize_full_history_results(
    run_name,
    results,
):

    degrading_results = [
        result
        for result in results.values()
        if result['path_type'] == 'degrading'
    ]

    no_degradation_results = [
        result
        for result in results.values()
        if result['path_type']
        == 'no_observed_degradation'
    ]

    rich.print('=' * 50)
    rich.print(
        f'[yellow]Full-history summary: {run_name}'
    )
    rich.print('-' * 50)

    rich.print(
        f'\tDegrading trajectories: '
        f'{len(degrading_results)}'
    )

    rich.print(
        f'\tNo-observed-degradation trajectories: '
        f'{len(no_degradation_results)}'
    )

    if len(degrading_results) > 0:

        positive_recalls = torch.tensor([
            result['positive_recall']
            for result in degrading_results
            if result['positive_recall'] is not None
        ])

        pre_horizon_positive_rates = torch.tensor([
            result['pre_horizon_positive_rate']
            for result in degrading_results
            if result['pre_horizon_positive_rate'] is not None
        ])

        rich.print()
        rich.print('\tDegrading trajectories:')

        if len(positive_recalls) > 0:
            rich.print(
                f'\t\tMean positive recall: '
                f'{positive_recalls.mean():.4f}'
            )
            rich.print(
                f'\t\tMedian positive recall: '
                f'{positive_recalls.median():.4f}'
            )

        if len(pre_horizon_positive_rates) > 0:
            rich.print(
                f'\t\tMean pre-horizon positive rate: '
                f'{pre_horizon_positive_rates.mean():.4f}'
            )
            rich.print(
                f'\t\tMedian pre-horizon positive rate: '
                f'{pre_horizon_positive_rates.median():.4f}'
            )

    if len(no_degradation_results) > 0:

        false_positive_rates = torch.tensor([
            result['false_positive_rate']
            for result in no_degradation_results
            if result['false_positive_rate'] is not None
        ])

        rich.print()
        rich.print(
            '\tNo-observed-degradation trajectories:'
        )

        if len(false_positive_rates) > 0:
            rich.print(
                f'\t\tMean false positive rate: '
                f'{false_positive_rates.mean():.4f}'
            )
            rich.print(
                f'\t\tMedian false positive rate: '
                f'{false_positive_rates.median():.4f}'
            )

    rich.print('-' * 50)

def plot_degrading_probability_vs_lead_time_per_run(
    results,
    prediction_horizon,
    run_name,
    max_individual_paths=20,
):

    degrading_results = [
        result
        for result in results.values()
        if result['path_type'] == 'degrading'
    ]

    if len(degrading_results) == 0:
        print(f'No degrading trajectories in {run_name}.')
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------------------------------
    # Individual trajectories
    # --------------------------------------------------

    for result in degrading_results[:max_individual_paths]:

        lead_times = result['lead_times'].cpu().numpy()
        probabilities = result['probabilities'].cpu().numpy()

        ax.plot(
            lead_times,
            probabilities,
            alpha=0.15,
            linewidth=0.8,
        )

    # --------------------------------------------------
    # Aggregate trajectories at each lead time
    # --------------------------------------------------

    probabilities_by_lead_time = {}

    for result in degrading_results:

        lead_times = result['lead_times'].cpu()
        probabilities = result['probabilities'].cpu()

        for lead_time, probability in zip(
            lead_times,
            probabilities,
        ):

            lead_time = int(lead_time.item())

            if lead_time not in probabilities_by_lead_time:
                probabilities_by_lead_time[lead_time] = []

            probabilities_by_lead_time[lead_time].append(
                probability.item()
            )

    lead_times = sorted(probabilities_by_lead_time.keys())

    median_probabilities = []
    lower_quantiles = []
    upper_quantiles = []

    for lead_time in lead_times:

        probabilities = torch.tensor(
            probabilities_by_lead_time[lead_time]
        )

        median_probabilities.append(
            probabilities.median().item()
        )

        lower_quantiles.append(
            torch.quantile(
                probabilities,
                0.25,
            ).item()
        )

        upper_quantiles.append(
            torch.quantile(
                probabilities,
                0.75,
            ).item()
        )

    # --------------------------------------------------
    # Median and interquartile range
    # --------------------------------------------------

    ax.fill_between(
        lead_times,
        lower_quantiles,
        upper_quantiles,
        alpha=0.25,
        label='25–75%',
    )

    ax.plot(
        lead_times,
        median_probabilities,
        linewidth=2.5,
        label='Median',
    )

    # Predictor decision threshold
    ax.axhline(
        0.5,
        linestyle='--',
        linewidth=1,
        label='Decision threshold',
    )

    # Nominal prediction horizon
    ax.axvline(
        prediction_horizon,
        linestyle='--',
        linewidth=1,
        label=f'Prediction horizon = {prediction_horizon}',
    )

    ax.set_xlabel('Steps before degradation')
    ax.set_ylabel('Predicted probability')
    ax.set_title(
        f'Predictor probability before degradation: {run_name}'
    )

    ax.set_ylim(0, 1)

    ax.invert_xaxis()

    ax.legend()

    plt.tight_layout()
    
    plt.show()



def plot_full_history_trajectory_diagnostics(
    all_results,
    prediction_horizon,
    max_lead_time=1000,
):

    num_sets = len(all_results)

    if num_sets == 0:
        return

    if num_sets <= 3:
        num_rows = 1
        num_cols = num_sets
    elif num_sets == 4:
        num_rows = 2
        num_cols = 2
    elif num_sets <= 6:
        num_rows = 2
        num_cols = 3
    elif num_sets <= 9:
        num_rows = 3
        num_cols = 3
    else:
        num_cols = 4
        num_rows = (
            num_sets + num_cols - 1
        ) // num_cols

    fig = plt.figure(
        figsize=(
            6 * num_cols,
            5.5 * num_rows,
        )
    )

    outer_grid = fig.add_gridspec(
        num_rows,
        num_cols,
        wspace=0.25,
        hspace=0.35,
    )

    for index, (run_name, results) in enumerate(
        all_results.items()
    ):

        row = index // num_cols
        column = index % num_cols

        inner_grid = outer_grid[
            row,
            column,
        ].subgridspec(
            2,
            1,
            height_ratios=[2, 1],
            hspace=0.08,
        )

        ax_probability = fig.add_subplot(
            inner_grid[0]
        )

        ax_flagged = fig.add_subplot(
            inner_grid[1],
            sharex=ax_probability,
        )

        degrading_results = [
            result
            for result in results.values()
            if result['path_type'] == 'degrading'
        ]

        if len(degrading_results) == 0:
            plot_label = get_plot_label(run_name)
            ax_probability.set_title(
                f'{plot_label}\n'
                f'No degrading trajectories'
            )
            continue

        # ------------------------------------------
        # Individual trajectories
        # ------------------------------------------

        for result in degrading_results:

            ax_probability.plot(
                result['lead_times'],
                result['probabilities'],
                alpha=0.12,
                linewidth=0.8,
            )

        # ------------------------------------------
        # Collect probabilities at each lead time
        # ------------------------------------------

        probabilities_by_lead_time = {}

        for result in degrading_results:

            for lead_time, probability in zip(
                result['lead_times'],
                result['probabilities'],
            ):

                lead_time = int(
                    lead_time.item()
                )

                if lead_time not in probabilities_by_lead_time:
                    probabilities_by_lead_time[
                        lead_time
                    ] = []

                probabilities_by_lead_time[
                    lead_time
                ].append(
                    probability.item()
                )

        lead_times = sorted(
            probabilities_by_lead_time.keys()
        )

        mean_probs = []
        median_probs = []
        lower_quantiles = []
        upper_quantiles = []
        fraction_flagged = []

        for lead_time in lead_times:

            probabilities = torch.tensor(
                probabilities_by_lead_time[
                    lead_time
                ]
            )

            mean_probs.append(
                probabilities.mean().item()
            )

            median_probs.append(
                probabilities.median().item()
            )

            lower_quantiles.append(
                torch.quantile(
                    probabilities,
                    0.25,
                ).item()
            )

            upper_quantiles.append(
                torch.quantile(
                    probabilities,
                    0.75,
                ).item()
            )

            fraction_flagged.append(
                (
                    probabilities > 0.5
                )
                .float()
                .mean()
                .item()
            )

        # ------------------------------------------
        # Aggregate probability statistics
        # ------------------------------------------

        ax_probability.fill_between(
            lead_times,
            lower_quantiles,
            upper_quantiles,
            color='lightsteelblue',
            alpha=0.30,
            label='25–75%',
        )

        ax_probability.plot(
            lead_times,
            mean_probs,
            linewidth=2.3,
            color='red',
            alpha=0.85,
            label='Mean',
        )

        ax_probability.plot(
            lead_times,
            median_probs,
            linewidth=2.3,
            color='slateblue',
            alpha=0.75,
            label='Median',
        )

        ax_probability.axhline(
            0.5,
            linestyle='--',
            linewidth=1,
            color='steelblue',
            alpha=0.8,
            label='Decision threshold',
        )

        ax_probability.axvline(
            prediction_horizon,
            linestyle='--',
            linewidth=1,
            color='steelblue',
            alpha=0.8,
            label=f'Horizon = {prediction_horizon}',
        )

        ax_probability.set_ylim(0, 1)

        plot_label = get_plot_label(run_name)
        ax_probability.set_title(
            f'{plot_label}\n'
            f'{len(degrading_results)} trajectories'
        )

        ax_probability.set_ylabel(
            'Predicted probability'
        )

        ax_probability.legend(
            framealpha=0.0,
        )

        # ------------------------------------------
        # Fraction flagged
        # ------------------------------------------

        ax_flagged.plot(
            lead_times,
            fraction_flagged,
            linewidth=2.3,
            color='slateblue',
            alpha=0.8,
        )

        ax_flagged.axhline(
            0.5,
            linestyle='--',
            linewidth=1,
            color='steelblue',
            alpha=0.8,
        )

        ax_flagged.axvline(
            prediction_horizon,
            linestyle='--',
            linewidth=1,
            color='steelblue',
            alpha=0.8,
        )

        ax_flagged.set_ylim(0, 1)

        ax_flagged.set_ylabel(
            'Fraction flagged'
        )

        ax_flagged.set_xlabel(
            'Steps before degradation'
        )


        ax_probability.set_xlim(
            max_lead_time,
            0,
        )

        plt.setp(
            ax_probability.get_xticklabels(),
            visible=False,
        )

    # fig.suptitle(
    #     'Predictor performance on full-history trajectories',
    #     fontsize=16,
    # )

    plt.show()
    
    plt.close(fig)







def plot_truncated_trajectory_diagnostics(
    all_trajectory_curves,
    prediction_horizon,
):

    num_sets = len(all_trajectory_curves)

    if num_sets == 0:
        return

    if num_sets <= 3:
        num_rows = 1
        num_cols = num_sets
    elif num_sets == 4:
        num_rows = 2
        num_cols = 2
    elif num_sets <= 6:
        num_rows = 2
        num_cols = 3
    elif num_sets <= 9:
        num_rows = 3
        num_cols = 3
    else:
        num_cols = 4
        num_rows = (
            num_sets + num_cols - 1
        ) // num_cols

    fig = plt.figure(
        figsize=(
            6 * num_cols,
            5.5 * num_rows,
        )
    )

    outer_grid = fig.add_gridspec(
        num_rows,
        num_cols,
        wspace=0.25,
        hspace=0.35,
    )

    for index, (run_name, curves) in enumerate(
        all_trajectory_curves.items()
    ):

        row = index // num_cols
        column = index % num_cols

        inner_grid = outer_grid[
            row,
            column,
        ].subgridspec(
            2,
            1,
            height_ratios=[2, 1],
            hspace=0.08,
        )

        ax_probability = fig.add_subplot(
            inner_grid[0]
        )

        ax_flagged = fig.add_subplot(
            inner_grid[1],
            sharex=ax_probability,
        )

        all_probs = torch.stack([
            curve['probabilities']
            for curve in curves.values()
        ])

        lead_times = next(
            iter(curves.values())
        )['lead_times']

        # ------------------------------------------
        # Individual trajectories
        # ------------------------------------------

        for curve in curves.values():

            ax_probability.plot(
                curve['lead_times'],
                curve['probabilities'],
                alpha=0.12,
                linewidth=0.8,
            )

        # ------------------------------------------
        # Aggregate probability statistics
        # ------------------------------------------

        mean_probs = all_probs.mean(dim=0)
        median_probs = all_probs.median(dim=0).values

        lower_quantiles = torch.quantile(
            all_probs,
            0.25,
            dim=0,
        )

        upper_quantiles = torch.quantile(
            all_probs,
            0.75,
            dim=0,
        )

        ax_probability.fill_between(
            lead_times,
            lower_quantiles,
            upper_quantiles,
            color='lightsteelblue',
            alpha=0.30,
            label='25–75%',
        )

        ax_probability.plot(
            lead_times,
            mean_probs,
            linewidth=2.3,
            color='red',
            alpha=0.85,
            label='Mean',
        )

        ax_probability.plot(
            lead_times,
            median_probs,
            linewidth=2.3,
            color='slateblue',
            alpha=0.75,
            label='Median',
        )

        ax_probability.axhline(
            0.5,
            linestyle='--',
            linewidth=1,
            color='steelblue',
            alpha=0.8,
            label='Decision threshold',
        )

        ax_probability.axvline(
            prediction_horizon,
            linestyle='--',
            linewidth=1,
            color='steelblue',
            alpha=0.8,
            label=f'Horizon = {prediction_horizon}',
        )

        ax_probability.set_ylim(0, 1)

        plot_label = get_plot_label(run_name)
        ax_probability.set_title(
            f'{plot_label}\n'
            f'{len(curves)} trajectories'
        )

        ax_probability.set_ylabel(
            'Predicted probability'
        )

        ax_probability.legend(
            framealpha=0.0,
        )

        # ------------------------------------------
        # Fraction flagged
        # ------------------------------------------

        fraction_flagged = (
            (all_probs > 0.5)
            .float()
            .mean(dim=0)
        )

        ax_flagged.plot(
            lead_times,
            fraction_flagged,
            linewidth=2.3,
            color='slateblue',
            alpha=0.8,
        )

        ax_flagged.axhline(
            0.5,
            linestyle='--',
            linewidth=1,
            color='steelblue',
            alpha=0.8,
        )

        ax_flagged.axvline(
            prediction_horizon,
            linestyle='--',
            linewidth=1,
            color='steelblue',
            alpha=0.8,
        )

        ax_flagged.set_ylim(0, 1)

        ax_flagged.set_ylabel(
            'Fraction flagged'
        )

        ax_flagged.set_xlabel(
            'Steps before degradation'
        )

        ax_probability.invert_xaxis()

        plt.setp(
            ax_probability.get_xticklabels(),
            visible=False,
        )

    # fig.suptitle(
    #     'Predictor performance on truncated trajectories',
    #     fontsize=16,
    # )

    plt.show()
    
    
    
    plt.close(fig)





if __name__=="__main__":

    train_data=torch.load(train_dataset_path)

    y_train=train_data['y_train'].to(device)
    y_eval=train_data['y_test'].to(device)

    X_train=metadata_times_to_tensor(
        train_data['train_metadata']
    )
    X_eval=metadata_times_to_tensor(
        train_data['test_metadata']
    )

    assert len(X_train) == len(y_train)
    assert len(X_eval) == len(y_eval)

    rich.print('='*50)
    rich.print(f'[yellow]Time-only logistic regression')
    rich.print('='*50)
    rich.print('Training')
    rich.print('-'*50)
    model, mu, sigma = train_predictor(
        X_train,
        y_train,
        X_eval,
        y_eval,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        predictor_seed=PREDICTOR_SEED,
    )
    rich.print('-'*50)

    all_trajectory_curves = {}

    loss_fn = torch.nn.BCEWithLogitsLoss()


    loss, accuracy, probabilities = evaluate_predictor(
        model,
        X_train,
        y_train,
        mu,
        sigma,
        loss_fn=loss_fn,
    )

    all_trajectory_curves[train_run_name+': train'] = (
        build_trajectory_curves(
            probabilities,
            train_data['train_metadata'],
        )
    )

 

    loss, accuracy, probabilities = evaluate_predictor(
        model,
        X_eval,
        y_eval,
        mu,
        sigma,
        loss_fn=loss_fn,
    )

    all_trajectory_curves[train_run_name+': test'] = (
        build_trajectory_curves(
            probabilities,
            train_data['test_metadata'],
        )
    )

    rich.print('Testing on prediction paths')
    rich.print('-'*50)

    for run_name, dataset_path in test_dataset_paths.items():

        # These two groups are concatenated below, so their
        # trajectory identities must be disjoint.

        test_data = torch.load(dataset_path)
        assert set(test_data['train_metadata']).isdisjoint(
            test_data['test_metadata']
        )

        X_test=torch.cat(
            (
                metadata_times_to_tensor(
                    test_data['train_metadata']
                ),
                metadata_times_to_tensor(
                    test_data['test_metadata']
                ),
            ),
            dim=0,
        )
        y_test = torch.cat(
            (test_data['y_train'],test_data['y_test'])
        ).to(device)

        assert len(X_test) == len(y_test)

        loss, accuracy, probabilities = evaluate_predictor(
            model,
            X_test,
            y_test,
            mu,
            sigma,
            loss_fn=torch.nn.BCEWithLogitsLoss()
        )

        rich.print(
            f'\t{run_name}: '
            f'\tloss={loss:.4f}, '
            f'\taccuracy=[green]{accuracy:.4f}'
        )

        rich.print('-'*50)

        all_trajectory_curves[run_name] = build_trajectory_curves(
            probabilities,
            test_data['train_metadata']|test_data['test_metadata'],
        )

    all_full_history_results = {}
    rich.print('\tTesting on full history paths')
    rich.print('-'*50)
    for run_name, dataset_path in full_history_dataset_paths.items():

        full_history_data = torch.load(dataset_path)

        # The time-only predictor ignores the stored X values,
        # but the dataset protocol must match the training dataset.
        assert (
            full_history_data['window_size']
            == train_data['window_size']
        )

        assert (
            full_history_data['prediction_horizon']
            == train_data['prediction_horizon']
        )

        results = evaluate_full_history_dataset(
            model=model,
            full_history_data=full_history_data,
            mu=mu,
            sigma=sigma,
            verbose=False,
        )

        all_full_history_results[run_name] = results

        summarize_full_history_results(
            run_name=run_name,
            results=results,
        )











    # --------------------------------------------------
    # Trajectory-level diagnostics
    # --------------------------------------------------

    prediction_horizon=train_data['prediction_horizon']

    plot_truncated_trajectory_diagnostics(
        all_trajectory_curves=
            all_trajectory_curves,
        prediction_horizon=
            prediction_horizon,
    )

    plot_full_history_trajectory_diagnostics(
        all_results=
            all_full_history_results,
        prediction_horizon=
            prediction_horizon,
        max_lead_time=800,
    )





