# Setup Instructions

## Quick Start Guide

Follow these steps to set up and run the Wine Quality Prediction application.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

**Option A: From Kaggle (Recommended)**
1. Visit: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
2. Download the dataset
3. Place the CSV file in `data/winequality.csv`

**Option B: Use UCI Dataset**
1. Visit: https://archive.ics.uci.edu/ml/datasets/wine+quality
2. Download red wine or white wine dataset
3. Save as `data/winequality.csv`

### 3. Train the Model

1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Navigate to `notebooks/model_training.ipynb`

3. Run all cells to:
   - Load and explore data
   - Preprocess data
   - Train models
   - Save model files

4. After running, you should have these files in the root directory:
   - `model.pkl`
   - `scaler.pkl`
   - `feature_names.pkl`
   - `model_metadata.pkl`
   - `confusion_matrix.pkl`
   - `model_comparison.pkl`

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Troubleshooting

### Issue: Model files not found
**Solution**: Run the `notebooks/model_training.ipynb` notebook first to generate model files.

### Issue: Dataset not found
**Solution**: Download the dataset from Kaggle and place it in `data/winequality.csv`

### Issue: Import errors
**Solution**: Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Streamlit not working
**Solution**: 
1. Check if Streamlit is installed: `pip show streamlit`
2. Reinstall if needed: `pip install --upgrade streamlit`

## Deployment to Streamlit Cloud

1. **Create GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select repository
   - Set main file: `app.py`
   - Click "Deploy"

3. **Important**: Ensure all `.pkl` files are committed to the repository (or use Git LFS for large files)

## File Structure After Setup

```
wine-quality-prediction/
├── app.py
├── requirements.txt
├── README.md
├── model.pkl                    # Generated after training
├── scaler.pkl                   # Generated after training
├── feature_names.pkl            # Generated after training
├── model_metadata.pkl           # Generated after training
├── confusion_matrix.pkl         # Generated after training
├── model_comparison.pkl         # Generated after training
├── data/
│   ├── README.md
│   └── winequality.csv          # Download from Kaggle
└── notebooks/
    └── model_training.ipynb
```

## Next Steps

1. Explore the dataset in the app
2. Make predictions with different wine characteristics
3. View model performance metrics
4. Customize the app for your needs

## Support

For issues or questions:
- Check the main README.md
- Review the notebook comments
- Check Streamlit documentation: https://docs.streamlit.io

