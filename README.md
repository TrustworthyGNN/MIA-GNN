# MIA GNN Project Starter


If you meet the version mismatch error for **Lasagne** library, please use following command to
upgrade **Lasagne** library.
<pre>
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
</pre>


# MIA Attack Process
    Step1: Training Target Model using Target Dataset
    Step2: Training Shadow Model using Shadow Dataset
    Step3: Training Attack Model using Posteriors retrieved from Shadow Model

    
Here, **Target Dataset** and **Shadow Dataset** are disjoint.
## Training Target and Shadow Model by GCN model
### TUs: DD, PROTEINS_full, ENZYMES
    # 10: run 10 times ;100:start from 100 epochs; DD : dataset DD
    sh run_TUs_target_shadow_training.sh 10 100 DD

### SPs: CIFAR10, MNIST
    # 10: run 10 times ;100:start from 100 epochs; DD : dataset DD
    sh run_SPs_target_shadow_training.sh 10 100 DD

## Membership Inference Attack

    # For transfer based attack, run 15 times
    sh run_transfer_attach.sh 15 

# Acknowledge

This project references from [benchmarking-gnn](https://github.com/graphdeeplearning/benchmarking-gnns) and 
[DeeperGCN](https://github.com/lightaime/deep_gcns_torch)

If it has any issues, please let me know.