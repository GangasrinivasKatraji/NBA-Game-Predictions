
class NBADataAnalyzer:
    """
    A class to analyze NBA data.

    Methods:
    - __init__: Initialize the NBADataAnalyzer with data.
    - get_shape: Get the shape of the data.
    - get_info: Get concise summary information about the data.
    - get_descriptive_stats: Get descriptive statistics of the data.
    - get_first_rows: Get the first 'n' rows of the data.
    - get_last_rows: Get the last 'n' rows of the data.
    - get_missing_values: Get the count of missing values in each column.
    - get_unique_values_categorical: Get unique values for categorical columns.
    - remove_null_values: Remove rows with null values (optionally in place).
    """

    def __init__(self, data):
        """
        Initialize the NBADataAnalyzer with data.

        Parameters:
        - data: The dataset to analyze.
        """
        self.data = data
    
    def get_shape(self):
        """
        Get the shape of the data.

        Returns:
        - tuple: A tuple representing the dimensions of the data (number of rows, number of columns).
        """
        return self.data.shape
    
    def get_info(self):
        """
        Get concise summary information about the data.

        Returns:
        - None: Prints the summary information.
        """
        return self.data.info()
    
    def get_descriptive_stats(self):
        """
        Get descriptive statistics of the data.

        Returns:
        - DataFrame: A DataFrame containing descriptive statistics.
        """
        return self.data.describe()
    
    def get_first_rows(self, n=5):
        """
        Get the first 'n' rows of the data.

        Parameters:
        - n (int): Number of rows to return. Default is 5.

        Returns:
        - DataFrame: The first 'n' rows of the data.
        """
        return self.data.head(n)
    
    def get_last_rows(self, n=5):
        """
        Get the last 'n' rows of the data.

        Parameters:
        - n (int): Number of rows to return. Default is 5.

        Returns:
        - DataFrame: The last 'n' rows of the data.
        """
        return self.data.tail(n)
    
    def get_missing_values(self):
        """
        Get the count of missing values in each column.

        Returns:
        - Series: A Series containing the count of missing values for each column.
        """
        return self.data.isnull().sum()
    
    def get_unique_values_categorical(self):
        """
        Get unique values for categorical columns.

        Returns:
        - dict: A dictionary where keys are categorical column names and values are arrays of unique values.
        """
        unique_values = {}
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            unique_values[column] = self.data[column].unique()
        return unique_values
    
    def remove_null_values(self, inplace=True):
        """
        Remove rows with null values (optionally in place).

        Parameters:
        - inplace (bool): If True, modify the DataFrame in place. Default is True.
        """
        self.data.dropna(inplace=inplace)