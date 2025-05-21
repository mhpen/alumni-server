"""
Data loader for the Alumni Management System.

This module provides functionality to load and manage datasets for the Alumni Management System.
"""
import os
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader for the Alumni Management System.
    
    This class provides functionality to load and manage datasets for the Alumni Management System.
    """
    
    def __init__(self, data_dir="data"):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = data_dir
        
        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize dataset paths
        self.career_path_data = os.path.join(self.data_dir, "career_path_data.csv")
        self.employment_data = os.path.join(self.data_dir, "employment_data.csv")
        
        logger.info(f"DataLoader initialized with data directory: {self.data_dir}")
    
    def load_career_path_dataset(self):
        """
        Load the career path dataset.
        
        Returns:
            Pandas DataFrame with the career path dataset, or None if the file doesn't exist
        """
        if os.path.exists(self.career_path_data):
            logger.info(f"Loading career path dataset from {self.career_path_data}")
            try:
                df = pd.read_csv(self.career_path_data)
                logger.info(f"Loaded career path dataset with shape: {df.shape}")
                return df
            except Exception as e:
                logger.error(f"Error loading career path dataset: {str(e)}")
                return None
        else:
            logger.warning(f"Career path dataset file not found: {self.career_path_data}")
            return None
    
    def load_employment_dataset(self):
        """
        Load the employment dataset.
        
        Returns:
            Pandas DataFrame with the employment dataset, or None if the file doesn't exist
        """
        if os.path.exists(self.employment_data):
            logger.info(f"Loading employment dataset from {self.employment_data}")
            try:
                df = pd.read_csv(self.employment_data)
                logger.info(f"Loaded employment dataset with shape: {df.shape}")
                return df
            except Exception as e:
                logger.error(f"Error loading employment dataset: {str(e)}")
                return None
        else:
            logger.warning(f"Employment dataset file not found: {self.employment_data}")
            return None
    
    def save_career_path_dataset(self, df):
        """
        Save the career path dataset.
        
        Args:
            df: Pandas DataFrame with the career path dataset
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Saving career path dataset to {self.career_path_data}")
            df.to_csv(self.career_path_data, index=False)
            logger.info(f"Saved career path dataset with shape: {df.shape}")
            return True
        except Exception as e:
            logger.error(f"Error saving career path dataset: {str(e)}")
            return False
    
    def save_employment_dataset(self, df):
        """
        Save the employment dataset.
        
        Args:
            df: Pandas DataFrame with the employment dataset
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Saving employment dataset to {self.employment_data}")
            df.to_csv(self.employment_data, index=False)
            logger.info(f"Saved employment dataset with shape: {df.shape}")
            return True
        except Exception as e:
            logger.error(f"Error saving employment dataset: {str(e)}")
            return False
    
    def clean_career_path_data(self, df):
        """
        Clean career path data.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Handle missing values
        if 'actual_job_title' in df.columns:
            df = df.assign(actual_job_title=df['actual_job_title'].fillna('Unemployed'))
            
        if 'predicted_job_title' in df.columns:
            df = df.assign(predicted_job_title=df['predicted_job_title'].fillna('Unemployed'))
            
        # Convert categorical variables
        if 'gender' in df.columns:
            df = df.assign(gender=df['gender'].fillna('Unknown'))
            
        if 'internship_experience' in df.columns:
            df = df.assign(internship_experience=df['internship_experience'].fillna('No'))
            
        if 'employed_after_grad' in df.columns:
            df = df.assign(employed_after_grad=df['employed_after_grad'].fillna('No'))
            
        # Handle skills column
        if 'skills' in df.columns:
            df = df.assign(skills=df['skills'].fillna('General Skills'))
            # Count number of skills
            df = df.assign(num_skills=df['skills'].apply(lambda x: len(str(x).split(','))))
            
        # Convert GPA to float
        if 'gpa' in df.columns:
            df = df.assign(gpa=pd.to_numeric(df['gpa'], errors='coerce'))
            if df['gpa'].isna().any():
                median_gpa = df['gpa'].median()
                df = df.assign(gpa=df['gpa'].fillna(median_gpa))
                
        return df
    
    def clean_employment_data(self, df):
        """
        Clean employment data.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Handle employment status
        if 'employed_after_grad' in df.columns:
            df = df.assign(employed_after_grad=df['employed_after_grad'].fillna('No'))
            
        # Handle time to employment
        if 'time_to_employment' in df.columns:
            # Fill missing time_to_employment with a high value for unemployed
            mask = (df['employed_after_grad'] == 'No') & (df['time_to_employment'].isna())
            if mask.any():
                df_temp = df.copy()
                df_temp.loc[mask, 'time_to_employment'] = 12  # 12 months as a default high value
                df = df_temp
                
            # Fill remaining missing values with median
            if df['time_to_employment'].isna().any():
                median_time = df['time_to_employment'].median()
                df = df.assign(time_to_employment=df['time_to_employment'].fillna(median_time))
                
        # Apply the same cleaning as career path data
        df = self.clean_career_path_data(df)
        
        return df
