# EV Energy Efficiency Prediction with Pyraformer
A Transformer-based time-series model for predicting electric vehicle (EV) battery energy efficiency using real-world driving data.

## Overview
This project focuses on predicting the energy efficiency of electric vehicle (EV) batteries, defined as the distance traveled per unit of State of Charge (SOC). By leveraging Pyraformer, a Transformer-based model with a Pyramidal Attention Module (PAM), we achieved accurate predictions to support better range estimation and alleviate range anxiety for EV users. The model processes real-world data from 205 EVs, capturing both short-term fluctuations and long-term trends in battery performance.
- Purpose: Enable precise energy efficiency predictions to optimize EV battery management and enhance user experience.
- Key Features:
  - Sequence-to-sequence (seq2seq) modeling for time-series forecasting.
  - Multi-scale attention to handle data variability and noise.
  - Prediction of energy efficiency 3 months ahead using 1 month of historical data.
- Duration: September 2022 - May 2024
in collaboration with AiCAR and Handong Global University.

## Tech Stack
- Language: Python
- Libraries/Frameworks: PyTorch, Optuna, Numpy, Pandas
- Models: Pyraformer, LSTM, Transformer
- Tools: Jupyter Notebook, Git
- Data: Real-world BMS and telematics data from 205 EVs (4 models: Ionic 5, Kona EV, EV6, Niro EV)

## My Role & Contributions
As a researcher, I contributed to optimizing the Pyraformer model for energy efficiency prediction. My key responsibilities included:
- **Data Preprocessing**: Processed raw BMS and telematics data, applying filters to exclude charging sessions, short trips (<1 km), and unstable data (first 10 minutes of trips). Used backward elimination to select 15 key features from 31 candidates.
- **Model Optimization**: Fine-tuned the existing Pyraformer model by adjusting its configuration to better handle real-world EV data variability.
- **Hyperparameter Tuning**: Leveraged Optuna to optimize hyperparameters (e.g., 8 attention heads, learning rate of 0.000157), improving model performance.
- **Evaluation**: Conducted comparative experiments against LSTM and Transformer models, validating Pyraformer’s superior performance with a SMAPE of 6.48% and RMSE of 0.466

## Achivements
- Model Performance: Achieved a SMAPE of 6.48% and RMSE of 0.466, outperforming LSTM (SMAPE: 9.74%, RMSE: 0.847) and standard Transformer (SMAPE: 8.95%, RMSE: 0.848).
- Real-World Impact: Enabled accurate 3-month-ahead energy efficiency predictions, supporting better battery health monitoring and range estimation.
- Research Contribution: Published in Proceedings of KSAE 2024 Annual Autumn Conference, titled "Application of a Transformer-Based Model for Predicting Electric Vehicle Energy Efficiency Using Real BMS and CAN Data."

## Acknowledgement
This research was conducted in collaboration with AiCAR and Handong Global University, with support from the following institutions:
- Supported by the Ministry of SMEs and Startups, Republic of Korea, under Project No. S3282558 (Technology Development Program for Startups).
- Supported by the 2023 Software-Centered University Program of the Ministry of Science and ICT and the Institute of Information & Communications Technology Planning & Evaluation (IITP), Republic of Korea (Project No. 2023-0-00055).
- Supported by the High-Performance Computing Support Program of the Ministry of Science and ICT and the National IT Industry Promotion Agency (NIPA), Republic of Korea.

##
*Note: This is part of my portfolio to demonstrate my experience in battery modeling and optimization. Due to the proprietary nature of the project, full code and detailed implementation are not publicly available. For further inquiries, please contact me via edenwldms@gmail.com*
