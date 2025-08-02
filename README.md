# 🧠 Deep Learning Path Planning  
*Path Planning Model for ASU Racing Team*

---

## 📑 Table of Contents
- [Problem Definition](#problem-definition)
- [Project Structure](#project-structure)
- [Procedure](#procedure)
- [How to Use](#how-to-use)
- [Results and Visualizations](#results-and-visualizations)
- [Download the Dataset](#download-the-dataset)
- [Contributing](#contributing)

---

## Problem Definition
The goal is to generate the optimal path for an autonomous vehicle navigating between cones of unknown order and quantity.



<p align="center">
  <img width="680" src="https://github.com/user-attachments/assets/5d2f2775-ed5e-4719-92ec-d5c04f04699e" alt="Input Cones and Track Prediction">
</p>

---

## Project Structure

The project is organized as follows:

```
ASURacing-PathPlanning/
├── config/                          # Configuration files
│   ├── data_path.yaml               # Data path configurations
│   └── train_config.yaml            # Training configurations
├── models/                          # Model definitions and checkpoints
│   ├── exported/                    # Exported trained models
│   │   └── model.pt                 # Example trained model
│   ├── model_lstm.py                # LSTM model definition
│   └── __pycache__/                 # Compiled Python files
├── notebooks/                       # Jupyter notebooks for exploration
│   └── deep-learning.ipynb          # Main notebook for analysis
├── scripts/                         # Python scripts for project tasks
│   ├── download_data.py             # Script to download data
│   ├── load_data.py                 # Script to preprocess data
│   ├── random-track-generator/      # Track generation scripts
│   │   ├── main.py                  # Main track generation script
│   │   ├── path.py                  # Path generation utilities
│   │   └── track_generator.py       # Track generation logic
│   ├── train.py                     # Model training script
│   ├── visualize_output.py          # Visualization script
│   └── __pycache__/                 # Compiled Python files
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
└── __init__.py                      # Python package initialization
```
---

## Procedure

1. **Track Generation**  
   Synthetic cone tracks are generated.

2. **Data Loader Pipeline**  
   - Adds noise to data  
   - Normalizes data:  
     - Aligns batch start at origin  
     - Rotates to face x-axis (vehicle-relative frame)  
   - Handles:
     - Padding and masking when left/right cone arrays differ  
     - Interleaving left and right cones into a single input tensor  
     - Random reversing for clockwise + counter-clockwise diversity  
   - Collates batches

3. **Model Training**  
   A Seq2Seq LSTM-based model is trained to predict the centerline.

4. **Evaluation and Visualization**  
   Results are visualized and compared to ground truth.

---

## How to Use

### 1. Clone the repository
```bash
git clone https://github.com/your-username/deep-learning-path-planning.git
cd deep-learning-path-planning
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Download dataset
```bash
python scripts/download_data.py
```
### 4. Visualize Output
```bash
python scripts/visualize_output.py
```

---

## Results and Visualizations
<img width="611" height="451" alt="image" src="https://github.com/user-attachments/assets/915e08e3-894f-4db3-a23f-fc42063720d0" />
> Green: Model Ouput
> Red: Ground Truth

---
## Download the Dataset
The dataset used for training is not uploaded due to size.
Download it from the following link:

👉 [Google Drive Dataset Link](https://drive.usercontent.google.com/download?id=1qY3mOd_fZ2XeBMGrDqEX7HasARWyOnp7&export=download&authuser=0)

Then extract it into the ```data/``` directory.
---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code follows the project’s coding standards and includes appropriate tests.



