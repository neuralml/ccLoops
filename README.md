# ccLoops
Cerebellar-driven cortical networks. A cortical RNN projects to a cerebellar network, which in turn returns feedback to the RNN (https://www.biorxiv.org/content/10.1101/2022.11.14.516257v1).

<img src="https://user-images.githubusercontent.com/35163856/202711207-297cb6db-2a4f-4b1f-989f-f76600c2d1ee.png" width="900">

## Dependencies
Beyond the standard python libraries, you will need [Pytorch](https://pytorch.org/) (we use version 1.7.0, but later should work) to define the neural network models and as well as [ignite](https://github.com/pytorch/ignite) (version 0.4.2) which wraps the training regime. [Sacred](https://github.com/IDSIA/sacred) (version 0.7.4) is used to record experiment details.

## Steps to run 
Scripts to run the tasks presented in the paper can be found in the /scripts folder. For a given script:
1. Define model/training/dataset hyperparameters*. These can be changed in the respective /configs folder or in the script itself (as a config update).
2. Define the OBSERVE variable (preset to False). If true, then experiment results will be saved in a generated /sims folder.
3. Run the script

*For the consolidation script (in /delass) it's only necessary to define the path to the pretrained model. At the moment this points to /delass/sims/1. 
