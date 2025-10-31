# Daghstul Bayesian Optimization Experiments

This repository contains experiments comparing different **Bayesian Optimization** acquisition strategies using **BoTorch**.  
The experiments evaluate **Expected Improvement (EI)** and **One-Shot Knowledge Gradient (KG)** methods on analytical benchmark functions.

## Repository Structure

Daghstul/  
│  
├── EI/  
│   └── ...  
│   (Expected Improvement experiments implemented using BoTorch's EI acquisition function)  
│  
├── KG/  
   └── ...  
    (One-Shot Knowledge Gradient experiments implemented using BoTorch's KG acquisition function)  

## Description

### 1. Expected Improvement (EI)
- Implemented using **BoTorch’s Expected Improvement** acquisition function.  
- Evaluated on **analytical benchmark functions**.  
- Focuses on exploiting known promising regions while maintaining exploration.

### 2. Knowledge Gradient (One-Shot KG)
- Implemented using **BoTorch’s One-Shot Knowledge Gradient**.  
- Uses a **Quantile Sampler** with **7 quantiles**.  
- Aims to quantify the **expected value of information** from evaluating new points.

## 🔗 Repository

Find the project here:  
[https://github.com/JuanUngredda/Daghstul](https://github.com/JuanUngredda/Daghstul)

## Environment Setup

This project uses **Conda** for environment management.  
To create and activate the environment, run the following commands:

```
conda env create -f environment.yml
conda activate daghstul
```


## Running the Experiments

Once the environment is activated, simply run:

```
python Launcher.py
```

This script executes all configured experiments for both the **EI** and **KG** methods.