# Traffic Volume Forecasting with Classical and Deep Learning Models

## Overview

This project focuses on forecasting hourly vehicle traffic counts across four junctions using time-series data from the UK Department for Transport. The goal is to model temporal patterns in urban traffic and generate accurate forecasts using both classical statistical techniques (SARIMA) and deep learning methods (LSTM).

## Key Findings

**Model Performance:**

- **SARIMA:** Provided strong baseline forecasts for all junctions with RMSE ranging from **4.61 to 17.87**. Captured seasonality and trend well but slightly underperformed in high-fluctuation junctions.
- **LSTM:** Outperformed SARIMA in all junctions, achieving RMSEs between **2.93 and 6.30**. Able to learn complex temporal dependencies, particularly beneficial for more variable traffic patterns.

**Model Generalization:**

- LSTM consistently demonstrated better generalization across junctions.
- Performance gap between SARIMA and LSTM was most notable in Junction 1, which showed the highest overall traffic volume and variability.

## Practical Considerations

- **Model Stability:** SARIMA models are interpretable and stable but need manual tuning of seasonal parameters. LSTM, while more computationally intensive, is more scalable across junctions.
- **Training Time:** 
  - SARIMA training per junction: ~2 minutes
  - LSTM (with ReLU and tanh, early stopping): ~3 minutes total
- **Stationarity Checks:** Augmented Dickey-Fuller tests confirmed all series were stationary (p < 0.05), validating the use of ARIMA-family models without differencing.
- **Seasonality Patterns:** All junctions displayed strong hourly and weekly seasonality, confirmed through decomposition.

## Model Comparison

| Junction | SARIMA RMSE | LSTM RMSE | Best Model |
|----------|-------------|-----------|------------|
| 1        | 17.87       | 6.30      | LSTM       |
| 2        | 7.12        | 3.86      | LSTM       |
| 3        | 5.99        | 5.71      | LSTM       |
| 4        | 4.61        | 2.94      | LSTM       |

## Future Work

- **Hyperparameter Tuning:** Perform full grid search for SARIMA (AIC/BIC optimization) and LSTM (sequence lengths, hidden units, learning rates).
- **Real-Time Inference:** Explore deployment of LSTM models for live traffic prediction via Streamlit or FastAPI.
- **Spatio-Temporal Modeling:** Extend to multi-junction models capturing inter-junction dependencies using Graph Neural Networks.
- **Hybrid Models:** Investigate SARIMA-LSTM ensembles or attention-based transformers (e.g., Temporal Fusion Transformers).

## Tools and Technologies

- **Pandas, NumPy, Matplotlib, Seaborn:** For data processing and visualization.
- **Statsmodels:** Used to fit SARIMA models and interpret time-series diagnostics.
- **Sklearn:** For metric evaluations (MAE, RMSE) and train-test splitting.
- **PyTorch:** For implementing LSTM models from scratch and managing training loops.
- **Google Colab:** Primary environment used, leveraging GPU acceleration where applicable.
- **Hugging Face Transformers (planned):** Considered for future enhancement with time-series transformer architectures.

## Conclusion

This project demonstrates that deep learning, particularly LSTMs, can meaningfully outperform classical models in traffic forecasting tasks—especially in high-volume and non-linear patterns. However, SARIMA remains a strong baseline, particularly when interpretability and low resource cost are priorities.

The modeling workflow here can be reused or extended for similar urban forecasting applications such as energy consumption, public transport flow, and crowd movement modeling.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

Free software: MIT license  
Dataset source: [UK DfT Traffic Volume Dataset](https://www.kaggle.com/datasets/fedesoriano/uk-traffic-counts)
