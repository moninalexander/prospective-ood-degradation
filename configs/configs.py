def make_dataset_config(prob_correct_shortcut, prob_wrong_shortcut, patch_H, patch_W):
    return {
        'prob_correct_shortcut': prob_correct_shortcut,
        'prob_wrong_shortcut': prob_wrong_shortcut,
        'nickname': f'pc_{prob_correct_shortcut}_pw_{prob_wrong_shortcut}_sh_{patch_H}_sw_{patch_W}',
        'patch_H':patch_H,
        'patch_W':patch_W,
        'seeds': {'test': 1234,
                  'eval': 2345,
                  'clean': 3456
                  }
    }

def make_cnn_config(
    ch_in,
    ch1,
    ch2,
    conv_ker_1,
    pool_ker_1,
    conv_ker_2,
    pool_ker_2,
    num_hid,
    num_classes,
):
    nickname = (
        f'cnn_'
        f'c1_{ch1}_'
        f'c2_{ch2}_'
        f'k1_{conv_ker_1}_'
        f'p1_{pool_ker_1}_'
        f'k2_{conv_ker_2}_'
        f'p2_{pool_ker_2}_'
        f'h_{num_hid}'
    )

    return {
        'name': 'CNN',
        'nickname': nickname,
        'parameters': {
            'ch_in': ch_in,
            'ch1': ch1,
            'ch2': ch2,
            'conv_ker_1': conv_ker_1,
            'pool_ker_1': pool_ker_1,
            'conv_ker_2': conv_ker_2,
            'pool_ker_2': pool_ker_2,
            'num_hid': num_hid,
            'num_classes': num_classes,
        },
    }
    


def make_mlp_config(
    num_h1,
    num_h2,
    num_classes,
):
    nickname = (
        f'mlp_'
        f'h1_{num_h1}_'
        f'h2_{num_h2}'
    )

    return {
        'name': 'MLP',
        'nickname': nickname,
        'parameters': {
            'num_h1': num_h1,
            'num_h2': num_h2,
            'num_classes': num_classes,
        },
    }
    

def make_training_config(
    num_epochs,
    max_steps,
    learning_rate,
    betas,
    batch_size,
    eval_size,
    extraction_step,
    source_accuracy_threshold,
    early_stop_degradation_threshold,
    early_stop_consecutive_required,
    nickname=None,
):
    if nickname is None:
        nickname = (
            f'epochs_{num_epochs}_'
            f'steps_{max_steps}_'
            f'lr_{learning_rate}_'
            f'bs_{batch_size}_'
            f'eval_{eval_size}_'
            f'extract_{extraction_step}'
        )

    return {
        'nickname': nickname,
        'parameters': {
            'num_epochs': num_epochs,
            'max_steps': max_steps,
            'learning_rate': learning_rate,
            'betas': betas,
            'batch_size': batch_size,
            'eval_size': eval_size,
            'extraction_step': extraction_step,
            'source_accuracy_threshold':source_accuracy_threshold,
            'early_stop_degradation_threshold':early_stop_degradation_threshold,
            'early_stop_consecutive_required':early_stop_consecutive_required
        },
    }


def make_experiment_config(
    architecture_config,
    dataset_config,
    training_config,
    seed
):
    return {
        'architecture': architecture_config,
        'dataset': dataset_config,
        'training': training_config,
        'seed': seed
    }


NUM_CLASSES=2
EARLY_STOP_DEGRADATION_THRESHOLD=0.45
EARLY_STOP_CONSECUTIVE_REQUIRED=100

clean_dataset=make_dataset_config(0,0,1,1)
perfect_shortcut_dataset=make_dataset_config(1,0,1,1)

clean_training = make_training_config(
    num_epochs=30,
    max_steps=12000,
    learning_rate=0.001,
    betas=[0.9, 0.999],
    batch_size=50,
    eval_size=1000,
    extraction_step=50,
    source_accuracy_threshold=2,
    early_stop_degradation_threshold=EARLY_STOP_DEGRADATION_THRESHOLD,
    early_stop_consecutive_required=EARLY_STOP_CONSECUTIVE_REQUIRED,
    nickname="clean_training",
)


perfect_shortcut_training = make_training_config(
    num_epochs=30,
    max_steps=12000,
    learning_rate=0.001,
    betas=[0.9, 0.999],
    batch_size=50,
    eval_size=1000,
    extraction_step=1,
    source_accuracy_threshold=0.99,
    early_stop_degradation_threshold=EARLY_STOP_DEGRADATION_THRESHOLD,
    early_stop_consecutive_required=EARLY_STOP_CONSECUTIVE_REQUIRED,
    nickname="shortcut_training",
)


###########################
############ CNN parameters
###########################


cnn_95 = make_cnn_config(3,3,3,2,2,2,2,4,NUM_CLASSES)


###########################
############ MLP parameters
###########################

mlp_95 = make_mlp_config(4,2,NUM_CLASSES)
