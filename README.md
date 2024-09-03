# news article and stock price analysis

The "News Article and Stock Price Analysis" project aims to investigate the relationship between news articles and stock price movements. By analyzing the sentiment of news headlines and correlating this with stock price data, the project seeks to uncover patterns and insights that can potentially predict market behavior. The analysis includes calculating the Pearson correlation coefficient to understand the relationship between news sentiment and stock price movements.

## Installation
### Creating a Virtual Environment
### Using Conda
If you prefer Conda as your package manager:

1. Open your terminal or command prompt.

2. Navigate to your project directory.

3. Run the following command to create a new Conda environment:

    ```bash
    conda create --name your_env_name python=3.12.5
    ```
    - Replace your_env_name with the desired name for your environment e.g. week0 and 3.12 with your preferred Python version.

4. Activate the environment:

    ```bash
    conda activate your_env_name
    ```

### Using Virtualenv
If you prefer using venv, Python's built-in virtual environment module:

1. Open your terminal or command prompt.

2. Navigate to your project directory.

3. Run the following command to create a new virtual environment:

    ```bash
    python -m venv your_env_name
    ```
    - Replace your_env_name with the desired name for your environment.

4. Activate the environment:

    - On Windows:
        ```bash
        .\your_env_name\scripts\activate
        ```

    - On macOS/Linux:
        ```bash
        source your_env_name/bin/activate
        ```
### Installing Dependencies
Onceyour virtual environment is created and activated. You can install packages and run your Python scripts within this isolated environment. Don't forget to install required packages using pip or conda once the environment is activated.

To run this project locally, you will need Python installed on your system along with the necessary libraries. 
You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Recommendation on installing talib
If you're using Windows and need to install TA-Lib, it is recommended to use Conda to avoid potential installation issues. To install TA-Lib with Conda, use:

```bash
conda install -c conda-forge talib
```

### Clone this package
- To install the network_analysis package, follow these steps:

- Clone the repository:

```bash
git clone https://github.com/your-username/KAIM_w2.git
```
- Navigate to the project directory:

```bash
cd KAIM_w2
Install the required dependencies:
```
```bash
pip install -r requirements.txt
```


## Usage Instructions

Once the dependencies are installed, you can run the analysis notebooks by launching Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```



## Contributions
Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

### Contact
For any questions or additional information please contact Endekalu.simon.haile@gmail.com