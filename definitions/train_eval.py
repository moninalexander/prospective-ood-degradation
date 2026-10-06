import torch

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset

from definitions.models import MLP, CNN
from definitions.data import ShortcutMNIST
import json, rich
from pathlib import Path
from configs.local_config import DATA_ROOT

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def collect_prediction_stats(logits,y_target):

    with torch.no_grad():

        log_prob=logits.log_softmax(dim=-1)
        prob=log_prob.exp()

        prob_prediction, y_pred=prob.max(dim=-1)
        ind_correct=(y_pred==y_target)

        confidence=prob_prediction.mean().item()
        confidence_correct=prob_prediction[ind_correct].mean().item() 
        confidence_wrong=prob_prediction[~ind_correct].mean().item()
        entropy=-(prob*log_prob).sum(dim=-1).mean().item()
   
    return (confidence,
            confidence_correct,
            confidence_wrong,
            entropy)


# Creating models and datasets

mnist_train_dataset=MNIST(root=DATA_ROOT,train=True,download=True, transform=ToTensor())
mnist_test_dataset=MNIST(root=DATA_ROOT,train=False,download=True, transform=ToTensor())



def create_model(model_config):

    if model_config['name']=='MLP':
        return MLP(**model_config['parameters'])
    elif model_config['name']=='CNN':
        return CNN(**model_config['parameters'])
    else:
        raise ValueError("Model's name should be one of MLP, CNN")


def create_datasets(dataset_config,eval_size,seed):

    train_generator=torch.Generator().manual_seed(seed)
    test_generator=torch.Generator().manual_seed(dataset_config['seeds']['test'])
    eval_generator = torch.Generator().manual_seed(dataset_config['seeds']['eval'])
    clean_generator = torch.Generator().manual_seed(dataset_config['seeds']['clean'])

    prob_correct=dataset_config['prob_correct_shortcut']
    prob_wrong=dataset_config['prob_wrong_shortcut']
    
    eval_size=eval_size

    train_dataset=ShortcutMNIST(mnist_train_dataset,
                                prob_correct,
                                prob_wrong,
                                shortcut_H=dataset_config['patch_H'],
                                shortcut_W=dataset_config['patch_W'],
                                generator=train_generator)
    test_dataset=ShortcutMNIST(mnist_test_dataset,
                               prob_correct,
                               prob_wrong,
                               shortcut_H=dataset_config['patch_H'],
                               shortcut_W=dataset_config['patch_W'],
                               generator=test_generator)
    clean_test_dataset=ShortcutMNIST(mnist_test_dataset,
                                     prob_correct_shortcut=0,
                                     prob_wrong_shortcut=0,
                                     shortcut_H=dataset_config['patch_H'],
                                     shortcut_W=dataset_config['patch_W'],
                                     generator=clean_generator)

    TEST_LENGTH=len(test_dataset)

    eval_rand_indices=torch.randperm(TEST_LENGTH,generator=eval_generator)[:eval_size]
    eval_subset_dataset=Subset(test_dataset,eval_rand_indices)

    clean_test_subset_dataset=Subset(clean_test_dataset,eval_rand_indices)
    return train_dataset, test_dataset, eval_subset_dataset, clean_test_subset_dataset, clean_test_dataset



def create_dataloaders(batch_size,dataset_config,eval_size,seed):

    (train_dataset, 
     test_dataset, 
     eval_subset_dataset,
     clean_test_subset_dataset,
     clean_test_dataset) = create_datasets(dataset_config,eval_size,seed)
    
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=len(test_dataset))

    eval_subset_loader=DataLoader(eval_subset_dataset,batch_size=eval_size)
    clean_test_subset_loader=DataLoader(clean_test_subset_dataset,batch_size=eval_size)
    clean_test_loader=DataLoader(clean_test_dataset,batch_size=len(test_dataset))

    return train_loader, test_loader, eval_subset_loader, clean_test_subset_loader, clean_test_loader


# Metrics

def collect_param_metrics(model:torch.nn.Module, 
                          rolling_param_all:torch.Tensor, 
                          initial_param_all:torch.Tensor,
                          initial_param_norm: float)->tuple:

    with torch.no_grad():

        param_all=torch.cat([param.flatten() for param in model.parameters()])
        param_norm=torch.linalg.vector_norm(param_all).item()
        delta_local_param_norm=torch.linalg.vector_norm(param_all-rolling_param_all).item()
        delta_global_param_norm=torch.linalg.vector_norm(param_all-initial_param_all).item()
        rolling_param_all = param_all.detach().clone()


    return (param_norm, # |\theta(t)|
            delta_local_param_norm, # |\theta(t)-\theta(t-1)|
            delta_global_param_norm, # |\theta(t)-\theta(0)|
            delta_global_param_norm/initial_param_norm, # |\theta(t)-\theta(0)|/|\theta(0)|
            rolling_param_all 
    )


def collect_grad_norm(model:torch.nn.Module)->tuple:
    with torch.no_grad():
        grad_all=torch.cat([param.grad.flatten() for param in model.parameters() if param.grad is not None])
        grad_norm=torch.linalg.vector_norm(grad_all).item()

    return grad_norm
        
    

# Training and evaluation

def eval_run(dataloader, model:torch.nn.Module, loss_fn:torch.nn.Module, device)->tuple:
    with torch.no_grad():
        model.eval()
        running_eval_loss=[]
        total=0
        correct=0
        for batch, (X, y) in enumerate(dataloader):
            X=X.to(device)
            y=y.to(device)
            y_pred=model(X)
            im_per_batch=len(y)
            total+=im_per_batch
            correct+=(torch.argmax(y_pred,dim=1)==y).sum().item()
            loss=im_per_batch*loss_fn(y_pred,y)
            running_eval_loss.append(loss.item())
        return sum(running_eval_loss)/total, correct/total 


def update_early_stop(
    source_accuracy,
    clean_accuracy,
    source_accuracy_threshold,
    early_stop_degradation_threshold,
    early_stop_consecutive_required,
    early_stop_consecutive_count=0,
):

    degradation = source_accuracy - clean_accuracy

    if (
        source_accuracy >= source_accuracy_threshold
        and degradation >= early_stop_degradation_threshold
    ):
        early_stop_consecutive_count += 1
    else:
        early_stop_consecutive_count = 0

    stop = early_stop_consecutive_count >= early_stop_consecutive_required

    return stop, early_stop_consecutive_count


def train_step(X:torch.Tensor, 
               y:torch.Tensor, 
               model:torch.nn.Module, 
               loss_fn:torch.nn.Module, 
               optimizer:torch.optim.Optimizer,
               rolling_parameters:torch.Tensor,
               initial_parameters:torch.Tensor,
               initial_parameters_norm:float)->tuple:

    optimizer.zero_grad()
    model.train()

    y_pred=model(X)

    correct=(torch.argmax(y_pred,dim=1)==y).sum().item()
    accuracy=correct/len(y)
    loss=loss_fn(y_pred,y)
    loss.backward()

    grad_norm_step=collect_grad_norm(model)

    (param_norm_step, 
     delta_local_param_norm_step,
     delta_global_param_norm_step,
     rel_delta_global_param_norm_step,
     rolling_parameters) = collect_param_metrics(model,
                                                 rolling_parameters, 
                                                 initial_parameters,
                                                 initial_parameters_norm)

    (confidence_step,
     confidence_correct_step,
     confidence_wrong_step,
     entropy_step)=collect_prediction_stats(y_pred,y)
    

    optimizer.step()
    return (loss.item(), 
            accuracy,
            grad_norm_step,
            param_norm_step, 
            delta_local_param_norm_step,
            delta_global_param_norm_step,
            rel_delta_global_param_norm_step,
            rolling_parameters,
            confidence_step,
            confidence_correct_step,
            confidence_wrong_step,
            entropy_step
            )



def train_run(train_dataloader,
              test_dataloader,
              test_subset_dataloader,
              clean_test_subset_dataloader,
              clean_test_dataloader,
              model,
              learning_rate,
              beta1,beta2,
              extraction_step, 
              num_epochs,
              max_steps,
              source_accuracy_threshold,
              early_stop_degradation_threshold,
              early_stop_consecutive_required,
              device)->tuple:

    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,betas=(beta1,beta2))
    
    train_time_stamps=[]
    collect_time_stamps=[]
    train_loss=[]
    source_eval_loss=[]
    train_accuracy=[]
    source_eval_accuracy=[]
    grad_norm=[]
    param_norm=[]
    delta_local_param_norm=[]
    delta_global_param_norm=[]
    rel_delta_global_param_norm=[]
    confidence=[]
    confidence_correct=[]
    confidence_wrong=[]
    entropy=[]
    clean_eval_loss=[]
    clean_eval_accuracy=[]

    initial_parameters=torch.cat([param.detach().flatten() for param in model.parameters()]).detach()
    initial_parameters_norm=torch.linalg.vector_norm(initial_parameters).item()

    rolling_parameters=initial_parameters.clone()

    consecutive_count = 0
    steps=0

    manual_break=False
    early_break=False

    for epoch in range(num_epochs):
        print('\n')
        print('-'*100)
        print(f'Epoch: \t {epoch}')
        print('-'*100)
        for batch, (X,y) in enumerate(train_dataloader):
            
            X=X.to(device)
            y=y.to(device)
            train_time_stamps.append(steps)

            (train_loss_step,
             train_accuracy_step,
             grad_norm_step,
             param_norm_step,
             delta_local_param_norm_step,
             delta_global_param_norm_step,
             rel_delta_global_param_norm_step,
             rolling_parameters,
             confidence_step,
             confidence_correct_step,
             confidence_wrong_step,
             entropy_step)=train_step(X,y,model,loss_fn,optimizer,
                                      rolling_parameters,
                                      initial_parameters,
                                      initial_parameters_norm)



            train_loss.append(train_loss_step)
            train_accuracy.append(train_accuracy_step)
            grad_norm.append(grad_norm_step)
            param_norm.append(param_norm_step)
            delta_local_param_norm.append(delta_local_param_norm_step)
            delta_global_param_norm.append(delta_global_param_norm_step)
            rel_delta_global_param_norm.append(rel_delta_global_param_norm_step)

            confidence.append(confidence_step)
            confidence_correct.append(confidence_correct_step)
            confidence_wrong.append(confidence_wrong_step)
            entropy.append(entropy_step)
            

            if steps%extraction_step==0:
                collect_time_stamps.append(steps+1)
                source_eval_loss_step,source_eval_accuracy_step=eval_run(test_subset_dataloader,model,loss_fn,device)
                source_eval_loss.append(source_eval_loss_step)
                source_eval_accuracy.append(source_eval_accuracy_step)

                clean_eval_loss_step,clean_eval_accuracy_step=eval_run(clean_test_subset_dataloader,model,loss_fn,device)
                clean_eval_loss.append(clean_eval_loss_step)
                clean_eval_accuracy.append(clean_eval_accuracy_step)

                stop, consecutive_count = update_early_stop(
                    source_eval_accuracy_step,
                    clean_eval_accuracy_step,
                    source_accuracy_threshold,
                    early_stop_degradation_threshold,
                    early_stop_consecutive_required,
                    early_stop_consecutive_count=consecutive_count)

                if stop:
                    early_break=True
                    break

            if (batch%240==0) and (batch!=0):
                rich.print(
                    f'Batch: {batch}'
                    f'\tSource: {source_eval_accuracy_step:.4f}' 
                    f'\tClean: {clean_eval_accuracy_step:.4f}'
                    f'\tDegradation: {source_eval_accuracy_step-clean_eval_accuracy_step:.4f}'
                    f'\tLoss: {train_loss_step:.4f}'
                )


            steps+=1
            if steps>=max_steps:
                manual_break=True
                break

        if manual_break or early_break:
            break    
        
    source_test_loss, source_test_accuracy=eval_run(test_dataloader,model,loss_fn, device)
    clean_test_loss, clean_test_accuracy=eval_run(clean_test_dataloader,model,loss_fn, device)

    print('-'*100)
    print(f'Source test accuracy: \t {source_test_accuracy} \t\t Source test loss: \t {source_test_loss}\n')
    print(f'Clean test Accuracy: \t {clean_test_accuracy} \t\t Clean test Loss: \t {clean_test_loss}\n')
                
    return (
        train_time_stamps, 
        collect_time_stamps, 
        train_loss, 
        train_accuracy, 
        source_eval_loss, 
        source_eval_accuracy,
        clean_eval_loss, 
        clean_eval_accuracy,

        grad_norm, 
        param_norm, 
        delta_local_param_norm,
        delta_global_param_norm,
        rel_delta_global_param_norm, 
        initial_parameters_norm,
        confidence,
        confidence_correct,
        confidence_wrong,
        entropy,
        source_test_loss,
        source_test_accuracy,
        clean_test_loss,
        clean_test_accuracy)

def save_experiment(config, metrics, model, experiment_folder, run_name):
    
    experiment_path=experiment_folder/config['dataset']['nickname']/config['architecture']['nickname']
    experiment_path=experiment_path/run_name/f'seed_{config['seed']}'

    if experiment_path.exists():
        raise ValueError(
            f"Experiment already exists: {experiment_path}"
        )

    experiment_path.mkdir(parents=True, exist_ok=True)
    
    with open(experiment_path/'config.json','w') as file:
        json.dump(config,file,indent=4)
    torch.save(metrics,experiment_path/'metrics.pt')
    torch.save(model.state_dict(),experiment_path/'model.pt')



def count_parameters(model):
    return sum(param.numel() for param in model.parameters())




def run_experiment(config,experiment_folder,run_name):

    torch.manual_seed(config['seed'])

    (train_loader, 
     test_loader, 
     eval_subset_loader,
     clean_test_subset_loader,
     clean_test_loader) = create_dataloaders(config['training']['parameters']['batch_size'],
                                             config['dataset'],
                                             config['training']['parameters']['eval_size'],
                                             config['seed'])


    device=get_device()

    model=create_model(config['architecture']).to(device)

    learning_rate=config['training']['parameters']['learning_rate']
    [beta1,beta2]=config['training']['parameters']['betas']
    extraction_step=config['training']['parameters']['extraction_step']
    num_epochs=config['training']['parameters']['num_epochs']
    max_steps=config['training']['parameters']['max_steps']

    source_accuracy_threshold = config['training']['parameters']['source_accuracy_threshold']
    early_stop_degradation_threshold = config['training']['parameters']['early_stop_degradation_threshold']
    early_stop_consecutive_required = config['training']['parameters']['early_stop_consecutive_required']

    print(f'\nTraining model with seed={config['seed']} on device={device}')

    (train_time_stamps, 
    collect_time_stamps, 
    train_loss, 
    train_accuracy, 
    source_eval_loss, 
    source_eval_accuracy,
    clean_eval_loss, 
    clean_eval_accuracy,
    grad_norm, 
    param_norm, 
    delta_local_param_norm,
    delta_global_param_norm,
    rel_delta_global_param_norm, 
    initial_parameters_norm,
    confidence,
    confidence_correct,
    confidence_wrong,
    entropy,
    source_test_loss,
    source_test_accuracy,
    clean_test_loss,
    clean_test_accuracy,)=train_run(train_loader,
                                    test_loader,
                                    eval_subset_loader,
                                    clean_test_subset_loader,
                                    clean_test_loader,
                                    model,
                                    learning_rate,
                                    beta1,beta2,
                                    extraction_step,
                                    num_epochs,
                                    max_steps,
                                    source_accuracy_threshold,
                                    early_stop_degradation_threshold,
                                    early_stop_consecutive_required,
                                    device)


    num_parameters=count_parameters(model)
    config['architecture']['num_parameters']=num_parameters

    metrics={
        'train_time_stamps':train_time_stamps,
        'collect_time_stamps':collect_time_stamps,
        'train_loss':train_loss, 
        'train_accuracy':train_accuracy, 
        'source_eval_loss':source_eval_loss, 
        'source_eval_accuracy':source_eval_accuracy,
        'clean_eval_loss':clean_eval_loss, 
        'clean_eval_accuracy':clean_eval_accuracy,
        'grad_norm':grad_norm, 
        'param_norm':param_norm,
        'delta_local_param_norm':delta_local_param_norm,
        'delta_global_param_norm':delta_global_param_norm,
        'rel_delta_global_param_norm':rel_delta_global_param_norm,
        'initial_parameters_norm':initial_parameters_norm,
        'confidence':confidence,
        'confidence_correct':confidence_correct,
        'confidence_wrong':confidence_wrong,
        'entropy':entropy,
        'source_test_loss':source_test_loss,
        'source_test_accuracy':source_test_accuracy,
        'clean_test_loss':clean_test_loss,
        'clean_test_accuracy':clean_test_accuracy
    }
    save_experiment(config,metrics,model,experiment_folder, run_name)


