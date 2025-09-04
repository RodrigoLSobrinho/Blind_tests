import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class DataLoader:
    """
    Utility class for loading and saving data.
    
    Methods:
        - load_data: Load data from file
        - save_data: Save data to file
        - load_wells_data: Load and merge data from multiple well files
    """
    
    def __init__(self):
        """Initialize the DataLoader class."""
        pass
    
    @staticmethod
    def load_data(file_path, file_type=None):
        """
        Load data from file.
        
        Args:
            file_path (str): Path to the file
            file_type (str): Type of file (default: None, inferred from extension)
            
        Returns:
            pd.DataFrame: Loaded data
        """
        if file_type is None:
            # Infer file type from extension
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.csv':
                file_type = 'csv'
            elif ext == '.pkl':
                file_type = 'pickle'
            elif ext in ['.xls', '.xlsx']:
                file_type = 'excel'
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        
        # Load data based on file type
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'pickle':
            df = pd.read_pickle(file_path)
        elif file_type == 'excel':
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return df
    
    @staticmethod
    def save_data(df, file_path, file_type=None):
        """
        Save data to file.
        
        Args:
            df (pd.DataFrame): Data to save
            file_path (str): Path to save the file
            file_type (str): Type of file (default: None, inferred from extension)
            
        Returns:
            None
        """
        if file_type is None:
            # Infer file type from extension
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.csv':
                file_type = 'csv'
            elif ext == '.pkl':
                file_type = 'pickle'
            elif ext in ['.xls', '.xlsx']:
                file_type = 'excel'
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save data based on file type
        if file_type == 'csv':
            df.to_csv(file_path, index=False)
        elif file_type == 'pickle':
            df.to_pickle(file_path)
        elif file_type == 'excel':
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        print(f"Data saved to {file_path}")

    @staticmethod
    def load_wells_data(base_path, data_type, wells=None):
        """
        Load and merge data from multiple well files based on the type parameter.
        
        Args:
            base_path (str): Base path to the wells directory (e.g., 'data/BDO')
            data_type (str): Type of data to load ('geof', 'geoq', or 'geoq_interpolado')
            wells (list): List of well names to load (default: None, loads all wells)
            
        Returns:
            pd.DataFrame: Merged DataFrame containing data from all specified wells
        """
        if data_type not in ['geof', 'geoq', 'geoq_interpolado']:
            raise ValueError("data_type must be one of: 'geof', 'geoq', 'geoq_interpolado'")
        
        # Get list of well directories
        well_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        
        # Filter wells if specified
        if wells is not None:
            well_dirs = [d for d in well_dirs if d in wells]
        
        if not well_dirs:
            raise ValueError("No well directories found")
        
        # Initialize empty list to store DataFrames
        dfs = []
        
        # Load data from each well
        for well_dir in well_dirs:
            # Construct file path - well name is both folder name and part of file name
            file_name = f"{data_type}_{well_dir}.pkl"
            file_path = os.path.join(base_path, well_dir, file_name)
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File not found for well {well_dir}: {file_path}")
                continue
            
            try:
                # Load data
                df = pd.read_pickle(file_path)
                
                # Add well name column if not present
                if 'WELLNAME' not in df.columns:
                    df['WELLNAME'] = well_dir
                
                dfs.append(df)
                
            except Exception as e:
                print(f"Error loading data for well {well_dir}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("No data was successfully loaded from any well")
        
        # Merge all DataFrames
        merged_df = pd.concat(dfs, ignore_index=True)
        print("Successfully loaded data from all wells")
        print(f"\nSuccessfully merged data from {len(dfs)} wells")
        print(f"Total number of rows: {len(merged_df)}")
        
        return merged_df


class DataPreprocessor:
    """
    Utility class for data preprocessing operations.
    
    Methods:
        - handle_missing_values: Handle missing values in data
        - filter_by_well: Filter data by well name
        - filter_columns: Filter columns based on patterns and specific names
        - filter_by_bacia: Filter data by basin name(s)
        - remove_outliers_by_cutoff: Remove outliers based on specific cutoffs for each variable
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor class."""
        # Define cutoff values for each variable
        self.cutoff_values = {
            "CAL": {"min": 0, "max": 30},      # Caliper (in)
            "GR": {"min": 0, "max": 200},      # Gamma Ray (API)
            "RHOB": {"min": 1.5, "max": 3.0},  # Bulk Density (g/cc)
            "NPHI": {"min": -0.1, "max": 0.6}, # Neutron Porosity (v/v)
            "DT": {"min": 40, "max": 200},     # Sonic (μs/ft)
            "RT": {"min": 0.1, "max": 2000},   # Resistivity (ohm.m)
            "PE": {"min": 0, "max": 10},       # Photoelectric Effect (b/e)
            "COT": {"min": 0, "max": 100}      # Total Organic Carbon (%)
        }

    @staticmethod
    def remove_outliers_by_cutoff(df, exclude_variables=None):
        """
        Remove outliers based on specific cutoffs for each variable.
        Instead of removing rows, values outside the cutoff range are set to NaN.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            exclude_variables (list, optional): List of variables to exclude from processing.
                                             If None, process all variables with defined cutoffs.
            
        Returns:
            pd.DataFrame: DataFrame with outliers set to NaN
        """
        df_processed = df.copy()
        
        # Initialize cutoff values
        cutoff_values = {
            "CAL": {"min": 0, "max": 30},      # Caliper (in)
            "GR": {"min": 0, "max": 260},      # Gamma Ray (API)
            "RHOB": {"min": 1.8, "max": 3.0},  # Bulk Density (g/cm3)
            "NPHI": {"min": -15, "max": 45},   # Neutron Porosity (m3/m3)
            "DT": {"min": 40, "max": 200},     # Sonic (μs/ft)
            "RT": {"min": 0, "max": 2000},     # Resistivity (ohm.m)
            "PE": {"min": 0, "max": 25},       # Photoelectric Effect (b/e)
            "COT": {"min": 0, "max": 100}      # Total Organic Carbon (%)
        }
        
        # If exclude_variables is None, initialize as empty list
        if exclude_variables is None:
            exclude_variables = []
            
        # Get list of available columns in the DataFrame
        available_columns = df.columns.tolist()
        
        # Process each variable that has defined cutoffs and is not in exclude_variables
        for var in cutoff_values:
            if var not in exclude_variables:
                if var in available_columns:
                    min_val = cutoff_values[var]["min"]
                    max_val = cutoff_values[var]["max"]
                    
                    # Count values before processing
                    n_before = df_processed[var].notna().sum()
                    
                    # Set values outside range to NaN
                    mask = (df_processed[var] < min_val) | (df_processed[var] > max_val)
                    df_processed.loc[mask, var] = np.nan
                    
                    # Count values after processing
                    n_after = df_processed[var].notna().sum()
                    n_removed = n_before - n_after
                    
                    # Print statistics
                    print(f"\nProcessing {var}:")
                    print(f"  - Cutoff range: [{min_val}, {max_val}]")
                    print(f"  - Values before: {n_before}")
                    print(f"  - Values after: {n_after}")
                    print(f"  - Values set to NaN: {n_removed} ({(n_removed/n_before*100):.2f}%)")
                else:
                    print(f"\nSkipping {var} - column not found in DataFrame")
            else:
                print(f"\nSkipping {var} as requested")
        
        return df_processed

    @staticmethod
    def filter_by_bacia(df, bacia_col, bacia_names):
        """
        Filter data by basin name(s).
        
        Args:
            df (pd.DataFrame): Input DataFrame
            bacia_col (str): Column name containing basin information
            bacia_names (list): List of basin names to filter
            
        Returns:
            pd.DataFrame: Filtered DataFrame containing only the specified basins
        """
        if not isinstance(bacia_names, list):
            bacia_names = [bacia_names]
            
        df_filtered = df[df[bacia_col].isin(bacia_names)].copy()
        
        # Print information about the filtering
        print(f"\nFiltering data for basins: {', '.join(bacia_names)}")
        print(f"Original number of rows: {len(df)}")
        print(f"Filtered number of rows: {len(df_filtered)}")
        print(f"Number of wells in filtered data: {df_filtered['WELLNAME'].nunique()}")
        
        return df_filtered

    @staticmethod
    def filter_by_well(df, well_name):
        """
        Filter data by well name.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            well_name (str): Well name to filter
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        df_filtered = df[df["WELLNAME"] == well_name].copy()
        return df_filtered
    
    @staticmethod
    def get_missing_percentage(df):
        """
        Calculate percentage of missing values for each column.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.Series: Percentage of missing values
        """
        missing_values_pct = (df.isnull().sum() / df.shape[0] * 100).round(2)
        return missing_values_pct
    
    @staticmethod
    def create_depth_intervals(df, depth_col="DEPTH", interval=0.5):
        """
        Create depth intervals for data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            depth_col (str): Column name for depth (default: "DEPTH")
            interval (float): Interval size (default: 0.5)
            
        Returns:
            pd.DataFrame: DataFrame with depth intervals
        """
        df_intervals = df.copy()
        
        # Create interval column
        df_intervals["depth_interval"] = pd.cut(
            df_intervals[depth_col],
            bins=np.arange(
                df_intervals[depth_col].min(),
                df_intervals[depth_col].max() + interval,
                interval
            )
        )
        
        return df_intervals

    @staticmethod
    def filter_columns(df, drop_patterns=None, drop_columns=None, keep_columns=None):
        """
        Filter columns based on patterns and specific column names.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            drop_patterns (list): List of patterns to drop (e.g., ['_mean', '_median', '_std', '_N'])
            drop_columns (list): List of specific column names to drop
            keep_columns (list): List of columns to keep regardless of patterns
            
        Returns:
            pd.DataFrame: DataFrame with filtered columns
        """
        df_filtered = df.copy()
        
        # Initialize list of columns to drop
        cols_to_drop = []
        
        # Add columns matching patterns to drop
        if drop_patterns:
            for pattern in drop_patterns:
                pattern_cols = [col for col in df.columns if pattern in col]
                cols_to_drop.extend(pattern_cols)
        
        # Add specific columns to drop
        if drop_columns:
            cols_to_drop.extend(drop_columns)
        
        # Remove any columns that should be kept
        if keep_columns:
            cols_to_drop = [col for col in cols_to_drop if col not in keep_columns]
        
        # Drop the columns
        df_filtered = df_filtered.drop(columns=cols_to_drop)
        
        # Print information about dropped columns
        if cols_to_drop:
            print(f"\nDropped {len(cols_to_drop)} columns:")
            for col in cols_to_drop:
                print(f"  - {col}")
            print(f"\nRemaining columns: {len(df_filtered.columns)}")
        
        return df_filtered


class DataAnalyzer:
    """
    Utility class for data analysis operations.
    
    Methods:
        - calculate_statistics: Calculate statistics for data
        - calculate_correlation: Calculate correlation between variables
    """
    
    def __init__(self):
        """Initialize the DataAnalyzer class."""
        pass
    
    @staticmethod
    def calculate_statistics(df, columns=None):
        """
        Calculate statistics for data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (list): Columns to analyze (default: None, all numeric columns)
            
        Returns:
            pd.DataFrame: DataFrame with statistics
        """
        if columns is None:
            # Select only numeric columns
            columns = df.select_dtypes(include=[np.number]).columns
        
        # Calculate statistics
        stats = df[columns].describe().T
        
        # Add additional statistics
        stats["variance"] = df[columns].var()
        stats["skewness"] = df[columns].skew()
        stats["kurtosis"] = df[columns].kurtosis()
        
        return stats
    
    @staticmethod
    def calculate_correlation(df, columns=None, method="pearson"):
        """
        Calculate correlation between variables.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (list): Columns to analyze (default: None, all numeric columns)
            method (str): Correlation method (default: "pearson")
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        if columns is None:
            # Select only numeric columns
            columns = df.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation
        corr_matrix = df[columns].corr(method=method)
        
        return corr_matrix


class PlotHelper:
    """
    Utility class for common plotting functions.
    
    Methods:
        - set_default_style: Set default style for plots
        - add_plot_elements: Add common elements to plot
        - save_plot: Save plot to file
    """
    
    def __init__(self):
        """Initialize the PlotHelper class."""
        pass
    
    @staticmethod
    def set_default_style():
        """
        Set default style for plots.
        
        Returns:
            None
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
    
    @staticmethod
    def add_plot_elements(ax, title=None, xlabel=None, ylabel=None, legend=False, grid=True):
        """
        Add common elements to plot.
        
        Args:
            ax: Matplotlib axis
            title (str): Plot title (default: None)
            xlabel (str): X-axis label (default: None)
            ylabel (str): Y-axis label (default: None)
            legend (bool): Whether to add legend (default: False)
            grid (bool): Whether to add grid (default: True)
            
        Returns:
            None
        """
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if legend:
            ax.legend()
        if grid:
            ax.grid(True)
    
    @staticmethod
    def save_plot(fig, file_path, dpi=300, bbox_inches="tight"):
        """
        Save plot to file.
        
        Args:
            fig: Matplotlib figure
            file_path (str): Path to save the figure
            dpi (int): Resolution (default: 300)
            bbox_inches (str): Bounding box setting (default: "tight")
            
        Returns:
            None
        """
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save figure
        fig.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Figure saved to {file_path}")