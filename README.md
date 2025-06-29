# VRKGNet Pipeline

## 🏁 Getting Started

### 🚀 Training Scripts

Use the training scripts in the `train_tools/` folder to start training your models.

#### 🔹 Without Prior Knowledge

Run the following script to train the model using only raw input data (e.g., XYZ + RGB):

```bash
bash train_tools/xyzrgb.sh
```

#### 🔹 With Prior Knowledge

Run the following script to train the model using additional prior knowledge (e.g., semantic features):

```bash
bash train_tools/xyzs.sh
```

### 💾 Model Weights

Trained model checkpoints are saved in the `work_dirs/` directory. 
https://drive.google.com/file/d/1LJhDS9E5M-fbhLQAAG2G65kolCXi8nuG/view?usp=sharing

### 📊 Results and Logs

All output results and logs are stored in the `results/` directory.

