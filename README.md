## Padding-free Interpretable Genomic Predication Directly from Highly Sparse Genotype Data using CMD source code 🥕 



## <font style="color:rgb(31, 35, 40);">Model architecture</font>
![](https://cdn.nlark.com/yuque/0/2025/png/2978547/1754723216521-90d0b0cf-6068-486e-99e1-01380e52b595.png)

### Data
```
The datasets associated with the paper can be downloaded and processed from the online sources mentioned in the paper. link.

```

## <font style="color:rgb(31, 35, 40);">Prerequisites</font>
```
conda create -n gs_mamba2 python=3.10  
conda activate gs_mamba2  
pip install torch numpy pandas scikit-learn scipy tqdm

```

## Data Preparation

+ **Genotype Data**: Three-channel one-hot encoded SNP matrix containing values `0`, `1`, `2`, and `-1` (missing values).
+ **Phenotype Data**: Continuous or categorical phenotype values.
+ **Data File Format**:
    - The first column should be the sample ID (or SNP locus ID).
    - No sex chromosome markers.
    - Data should be split into training and test sets in advance or will be split automatically by the code



## How to Run
1. **Train the model**

```
 python CMD.py \

  --data_path new_data.csv \  
  --pheno_path 1000pheno.txt \  
  --missing_ratio 0.3 \  
  --batch_size 32 \  
  --lr 0.0001 \  
  --epochs 100 \  
  --n_splits 5

```

2. **Model Evaluation**
    - Uses **KFold cross-validation** for training and evaluation
    - Supports **Early Stopping**
    - Outputs performance metrics including Pearson correlation coefficient and RMSE
3. **Save Results**
    - Predicted and true values saved as `.csv`
    - Performance metrics for each cross-validation fold recorded in a log file

