# Market Anomaly Detector

This project implements a market anomaly detection system using machine learning techniques, specifically Isolation Forest and LSTM models. The goal is to identify anomalies in financial data.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/market-anomaly-detector.git
   cd market-anomaly-detector
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip3 install -r requirements.txt
   ```

## Usage

To run the model training and evaluation, execute the following command:
```bash
python3 api/run_models.py
```

This script will train the Isolation Forest and LSTM models, evaluate their performance, and save the results to a file.

## Results

The results will be saved in the `results` directory.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is open-sourced under the MIT License - see the LICENSE file for details.
