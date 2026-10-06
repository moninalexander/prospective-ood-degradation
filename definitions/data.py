from torch.utils.data import Dataset
import torch

def add_masked_shortcut(images: torch.Tensor, mask: torch.Tensor, state: int, shortcut_H, shortcut_W)->torch.Tensor:
    if state==0:
        images[mask,0,0:shortcut_H,0:shortcut_W]=1
        images[mask,1,0:shortcut_H,0:shortcut_W]=0
        images[mask,2,0:shortcut_H,0:shortcut_W]=0
    if state==1:
        images[mask,0,0:shortcut_H,0:shortcut_W]=0
        images[mask,1,0:shortcut_H,0:shortcut_W]=0
        images[mask,2,0:shortcut_H,0:shortcut_W]=1
    return images


class ShortcutMNIST(Dataset):
    def __init__(self, 
                 original_dataset, 
                 prob_correct_shortcut=0, 
                 prob_wrong_shortcut=0,
                 shortcut_H=3,
                 shortcut_W=3,
                 generator=None):
        super().__init__()
        self.generator=generator
        self.shortcut_H=shortcut_H
        self.shortcut_W=shortcut_W
        images=(original_dataset.data.float()/255).unsqueeze(1).repeat(1,3,1,1)
        labels=original_dataset.targets.clone()
        length=len(labels)

        labels[labels<5]=0
        labels[labels>=5]=1

        # The mask is constructed using the conditional probability P_+/(P_+ + P_-) for the correct (P_+) and incorrect (P_-) labelling
        
        if prob_correct_shortcut>1 or prob_wrong_shortcut>1 or prob_correct_shortcut<0 or prob_wrong_shortcut<0 or prob_correct_shortcut+prob_wrong_shortcut>1:
            raise ValueError('Both probabilities and their sum should take values in [0,1]')

        mask_shortcut_all=torch.rand(length,generator=self.generator)<prob_correct_shortcut+prob_wrong_shortcut

        mask_shortcut_correct=torch.zeros_like(mask_shortcut_all)
        mask_shortcut_wrong=mask_shortcut_correct

        if prob_correct_shortcut+prob_wrong_shortcut!=0.:
            mask_shortcut_correct=torch.rand(length,generator=self.generator)<=prob_correct_shortcut/(prob_correct_shortcut+prob_wrong_shortcut)
            mask_shortcut_wrong=~mask_shortcut_correct

        self.shortcut_zero_correct=mask_shortcut_all & mask_shortcut_correct & (labels==0)
        self.shortcut_zero_wrong=mask_shortcut_all & mask_shortcut_wrong & (labels==0)
        self.shortcut_one_correct=mask_shortcut_all & mask_shortcut_correct & (labels==1)
        self.shortcut_one_wrong=mask_shortcut_all & mask_shortcut_wrong & (labels==1)


        images=add_masked_shortcut(images,self.shortcut_zero_correct,0,self.shortcut_H,self.shortcut_W)
        images=add_masked_shortcut(images,self.shortcut_zero_wrong,1,self.shortcut_H,self.shortcut_W)
        images=add_masked_shortcut(images,self.shortcut_one_correct,1,self.shortcut_H,self.shortcut_W)
        images=add_masked_shortcut(images,self.shortcut_one_wrong,0,self.shortcut_H,self.shortcut_W)

        self.labels=labels
        self.images=images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,indx):
        return self.images[indx], self.labels[indx]