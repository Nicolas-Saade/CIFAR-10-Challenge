# CIFAR-10-Challenge
A CIFAR-10 Classification Challenge that includes 3 notebooks: a CNN trained from Scratch, a Vision Transformer trained from scratch, Pre-trained Model Ensemble for max performance

# Comprehensive Report:

# N.B.
Note for the assignment, we had accidently submitted our predictions initially on the Without LLMs for our Ensemble using pre-trained models, we re-submitted accordingly on the With LLMs competition.
## Vision Transformer (ViT)
### Loading pre-trained weights
If facing issues with loading pre-trained weights on kaggle. Make sure to have the 2 GPU accelerators on in your environment. Then, do the following 

```python
model = nn.DataParallel(model)
model = nn.DataParallel(model)
```

This is because the model was on nn.DataParallel() originally when its weights was saved. 

The weights for the CNN are in the file `vit_weights.pth`

## Convolutional Neural Network (CNN)
### Loading pr-trained weights and model
To avoid problems when loading the model load the configuration on an A100 GPU or a T4 GPU. Also it is needed to define a configuration when instantiating a model. The first cell in the subsection "Best Model after HPO" is an example of the parameters and can be used as a configuration.

The weights for the CNN are in the file `HERO_FINAL_82_PERCENT.pth`

## Task 2 - Ensemble

### Inference & Weights

load the 15 model weights (`.pth` files that should be saved after each model's training run in the Save Weights sections) that should be uploaded Google Drive, in the path defined by the `drive_folder` variable. Run the variable containing that cell.

After inference completes, the notebook automatically generates the submission.csv.

The weights for the ensemble are all files that have names like `*_Ensemble_*.pth`
