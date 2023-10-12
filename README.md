# SubTree Attention Graph Neural Network Source Code

The official implementation for **"Tailoring Self-Attention for Graph via Rooted Subtrees"** which is accepted to NeurIPS2023 as a poster.

Related materials: [[arxiv](https://arxiv.org/abs/2310.05296)]

---

In this repository, you'll find the requisite code to reproduce the empirical results in our research paper. There are two methods available to operate this code:

1. Utilizing Weights & Biases (WandB), which is our recommended approach.
2. Executing directly through provided scripts.

---

**Dependencies**

The following list presents the versions of the packages used in our experiments.

```
python==3.7.12
pytorch==1.8.0
torch_geometric==2.0.1
torch_sparse==0.6.11
torch_scatter==2.1.1
torch_cluster==1.6.1
torch_spline_conv==1.2.2
wandb==0.12.16
```

---

**Preparing data**

Please unzip data.zip first and put the unzipped data folder in the parent directory of this folder.

```
parent directory
├── data
│   ├── Amazon
│   └── Citation_Full
│   └── ...
└── SubTree-Attention
    ├── best_params_yamls
    └── scripts
    └── ...
```

---

**Option 1: Using Provided Scripts**
We also provide script files that can be run directly.     For CiteSeer, Cora, Deezer-Europe and Film, please use `scripts/exp_setting_1.sh`. For Computers, CoraFull, CS, Photo, Physics and Pubmed, please use `best_params_yamls/scripts/exp_setting_2.sh`.

---

**Option 2: Using Weights & Biases**

1. **Setup:**
   Initially, create the `configs` and `remote` folders within the root directory. These will be used to store wandb files.
2. **Initiating Sweep:**
   We provide the hyperparameters for each dataset within the `best_params_yamls` folder. You can use them to create a sweep. Make sure to specify your wandb username and project name when doing so.

   For CiteSeer, Cora, Deezer-Europe and Film, please use yamls in `best_params_yamls/setting_1`. For Computers, CoraFull, CS, Photo, Physics and Pubmed, please use yamls in `best_params_yamls/setting_2`.

   Here's an example command:

   ```
   python sweep.py --entity=$YOUR_WANDB_ENTITY$ --project=$YOUR_WANDB_PROJECT$ --source=file --info=best_params_yamls/setting_1/citeseer.yaml
   ```
   Please replace `$YOUR_WANDB_ENTITY$` and `$YOUR_WANDB_PROJECT$` with your wandb username and project name respectively.
3. **Initiating Agent:**
   Once you run the above command, you will receive a sweep ID `$SWEEP_ID$` and sweep URL `$SWEEP_URL$`, similar to the example shown below:

   ```
   Create sweep with ID: $SWEEP_ID$
   Sweep URL: $SWEEP_URL$
   ```
   You can now choose to run the program in single process mode or in parallel.

   - For single process execution, use the following command:

   ```
   python agents.py --entity=$YOUR_WANDB_ENTITY$ --project=$YOUR_WANDB_PROJECT$ --sweep_id=$SWEEP_ID$ --gpu_allocate=$INDEX_GPU$:1 --wandb_base=remote --mode=one-by-one --save_model=False
   ```
   - For parallel execution, use the command below:

   ```
   python agents.py --entity=$YOUR_WANDB_ENTITY$ --project=$YOUR_WANDB_PROJECT$ --sweep_id=$SWEEP_ID$ --gpu_allocate=$INDEX_GPU$:$PARALLEL_RUNS$ --wandb_base=temp --mode=parallel --save_model=False
   ```
   Here, the parameter `$INDEX_GPU$:$PARALLEL_RUNS$` specifies that `$PARALLEL_RUNS$` will run concurrently on GPU `$INDEX_GPU$`. In multi-process mode, you can link multiple GPUs to distribute tasks concurrently using `-`.

   For instance:

   ```
   python agents.py --entity=$YOUR_WANDB_ENTITY$ --project=$YOUR_WANDB_PROJECT$ --sweep_id=$SWEEP_ID$ --gpu_allocate=$INDEX_GPU_1$:$PARALLEL_RUNS$-$INDEX_GPU_2$:$PARALLEL_RUNS$ --wandb_base=temp --mode=parallel --save_model=False
   ```
4. **Results Evaluation:**
   The outcomes of your experiment can be viewed at the `$SWEEP_URL$`, which is a webpage hosted on [wandb.ai](https://wandb.ai).
