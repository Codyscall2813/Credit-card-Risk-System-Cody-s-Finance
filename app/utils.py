"""
OPTIMIZED Utility Functions for Credit Risk AI Platform
Version: 2.0 - Performance Enhanced
"""

import pandas as pd
import numpy as np
import yaml
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import hashlib
from datetime import datetime, timedelta
import re
import time
import functools
import streamlit as st
from pathlib import Path
import warnings
import io
import base64

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/utils.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# OPTIMIZATION 1: Cached configuration loading
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file with caching - OPTIMIZED
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    
    Returns:
    --------
    Dict[str, Any]: Configuration dictionary
    """
    try:
        start_time = time.time()
        
        if not Path(config_path).exists():
            logger.error(f"Configuration file not found: {config_path}")
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        load_time = time.time() - start_time
        logger.info(f"✅ Configuration loaded from {config_path} in {load_time:.3f}s")
        
        # Validate critical configuration sections
        required_sections = ['model', 'scoring', 'risk', 'business_rules']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            logger.warning(f"Missing configuration sections: {missing_sections}")
        
        return config
        
    except yaml.YAMLError as ye:
        logger.error(f"YAML parsing error in {config_path}: {ye}")
        return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

# OPTIMIZATION 2: Enhanced performance decorator
def performance_monitor(func):
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            end_memory = get_memory_usage()
            memory_delta = end_memory - start_memory
            
            # Log performance metrics
            logger.debug(f"⚡ {func.__name__}: {execution_time:.3f}s, Memory: {memory_delta:+.1f}MB")
            
            # Store performance metrics in session state if available
            if hasattr(st, 'session_state') and 'function_performance' not in st.session_state:
                st.session_state.function_performance = []
            
            if hasattr(st, 'session_state'):
                st.session_state.function_performance.append({
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'memory_delta': memory_delta,
                    'timestamp': datetime.now()
                })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

# OPTIMIZATION 3: Enhanced dataframe validation
@performance_monitor
def validate_dataframe(df: pd.DataFrame, required_columns: List[str], 
                      strict_mode: bool = False) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Enhanced dataframe validation with comprehensive checks - OPTIMIZED
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : List[str]
        List of required column names
    strict_mode : bool
        Enable strict validation (data types, ranges)
    
    Returns:
    --------
    Tuple[bool, List[str], Dict[str, Any]]: (is_valid, issues, quality_metrics)
    """
    issues = []
    quality_metrics = {}
    
    try:
        # Basic structure validation
        if df is None or df.empty:
            issues.append("DataFrame is None or empty")
            return False, issues, {}
        
        # Column validation
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        extra_columns = [col for col in df.columns if col not in required_columns]
        if extra_columns:
            logger.info(f"Extra columns found (will be ignored): {extra_columns}")
        
        # Data quality metrics
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        quality_metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_percentage': round(missing_percentage, 2),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        # Data quality validation
        if missing_percentage > 20:
            issues.append(f"High missing data percentage: {missing_percentage:.1f}%")
        
        duplicate_percentage = (quality_metrics['duplicate_rows'] / len(df)) * 100
        if duplicate_percentage > 5:
            issues.append(f"High duplicate percentage: {duplicate_percentage:.1f}%")
        
        # Strict mode validations
        if strict_mode:
            for col in required_columns:
                if col in df.columns:
                    # Check for unexpected data types
                    if df[col].dtype == 'object':
                        # Check for numeric columns that should be numeric
                        if 'amount' in col.lower() or 'score' in col.lower() or 'ratio' in col.lower():
                            try:
                                pd.to_numeric(df[col], errors='raise')
                            except (ValueError, TypeError):
                                issues.append(f"Column '{col}' contains non-numeric values")
                    
                    # Check for reasonable ranges
                    if 'age' in col.lower():
                        invalid_ages = df[(df[col] < 18) | (df[col] > 100)][col].count()
                        if invalid_ages > 0:
                            issues.append(f"Invalid age values found: {invalid_ages} records")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"DataFrame validation failed: {issues}")
        else:
            logger.info(f"✅ DataFrame validation passed: {quality_metrics}")
        
        return is_valid, issues, quality_metrics
        
    except Exception as e:
        logger.error(f"Error in dataframe validation: {e}")
        issues.append(f"Validation error: {str(e)}")
        return False, issues, quality_metrics

# OPTIMIZATION 4: Enhanced currency string handling
@performance_monitor
def clean_currency_string(value: Union[str, int, float]) -> float:
    """
    Enhanced currency string cleaning with better error handling - OPTIMIZED
    
    Parameters:
    -----------
    value : Union[str, int, float]
        Currency string or numeric value
    
    Returns:
    --------
    float: Numeric value
    """
    if pd.isna(value) or value is None:
        return 0.0
    
    # Handle numeric inputs directly
    if isinstance(value, (int, float)):
        return float(value)
    
    try:
        # Convert to string for processing
        str_value = str(value).strip()
        
        if not str_value:
            return 0.0
        
        # Remove common currency symbols and formatting
        cleaned = re.sub(r'[₹$€£¥,\s]', '', str_value)
        
        # Handle common abbreviations
        multipliers = {
            'k': 1000, 'K': 1000,
            'l': 100000, 'L': 100000, 'lac': 100000, 'lakh': 100000,
            'cr': 10000000, 'Cr': 10000000, 'crore': 10000000,
            'm': 1000000, 'M': 1000000, 'million': 1000000,
            'b': 1000000000, 'B': 1000000000, 'billion': 1000000000
        }
        
        # Check for multipliers
        for suffix, multiplier in multipliers.items():
            if cleaned.lower().endswith(suffix.lower()):
                numeric_part = cleaned[:-len(suffix)]
                try:
                    return float(numeric_part) * multiplier
                except ValueError:
                    continue
        
        # Direct numeric conversion
        return float(cleaned)
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert '{value}' to float: {e}")
        return 0.0

# OPTIMIZATION 5: Cached statistics calculation
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def calculate_statistics(data: pd.Series, include_advanced: bool = False) -> Dict[str, float]:
    """
    Calculate comprehensive statistics with caching - OPTIMIZED
    
    Parameters:
    -----------
    data : pd.Series
        Data series to analyze
    include_advanced : bool
        Include advanced statistics (percentiles, moments)
    
    Returns:
    --------
    Dict[str, float]: Statistics dictionary
    """
    try:
        if data.empty:
            return {'error': 'Empty data series'}
        
        # Basic statistics
        stats = {
            'count': int(len(data)),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'q1': float(data.quantile(0.25)),
            'median': float(data.median()),
            'q3': float(data.quantile(0.75)),
            'max': float(data.max()),
            'range': float(data.max() - data.min()),
            'iqr': float(data.quantile(0.75) - data.quantile(0.25))
        }
        
        # Add coefficient of variation
        if stats['mean'] != 0:
            stats['cv'] = stats['std'] / abs(stats['mean'])
        else:
            stats['cv'] = 0
        
        # Advanced statistics
        if include_advanced:
            try:
                stats.update({
                    'skew': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'p5': float(data.quantile(0.05)),
                    'p10': float(data.quantile(0.10)),
                    'p90': float(data.quantile(0.90)),
                    'p95': float(data.quantile(0.95)),
                    'p99': float(data.quantile(0.99)),
                    'missing_count': int(data.isnull().sum()),
                    'missing_percentage': float(data.isnull().sum() / len(data) * 100),
                    'unique_count': int(data.nunique()),
                    'unique_percentage': float(data.nunique() / len(data) * 100)
                })
            except Exception as e:
                logger.warning(f"Error calculating advanced statistics: {e}")
        
        # Handle infinite and NaN values
        for key, value in stats.items():
            if pd.isna(value) or np.isinf(value):
                stats[key] = 0.0
            else:
                stats[key] = round(float(value), 6)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return {'error': str(e)}

# OPTIMIZATION 6: Enhanced application ID generation
def generate_application_id(prefix: str = "APP", include_checksum: bool = False) -> str:
    """
    Generate unique application ID with optional checksum - OPTIMIZED
    
    Parameters:
    -----------
    prefix : str
        ID prefix
    include_checksum : bool
        Include checksum for validation
    
    Returns:
    --------
    str: Unique application ID
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # Include milliseconds
        random_suffix = np.random.randint(1000, 9999)
        
        base_id = f"{prefix}{timestamp}{random_suffix}"
        
        if include_checksum:
            # Calculate simple checksum
            checksum = sum(ord(c) for c in base_id) % 100
            return f"{base_id}{checksum:02d}"
        
        return base_id
        
    except Exception as e:
        logger.error(f"Error generating application ID: {e}")
        return f"{prefix}{int(time.time())}{np.random.randint(1000, 9999)}"

# OPTIMIZATION 7: Enhanced financial calculations
@performance_monitor
def calculate_financial_ratios(
    income: float,
    loan_amount: float,
    tenure_months: int,
    interest_rate: float,
    existing_emi: float = 0,
    include_advanced: bool = False
) -> Dict[str, float]:
    """
    Calculate comprehensive financial ratios - OPTIMIZED
    
    Parameters:
    -----------
    income : float
        Annual income
    loan_amount : float
        Requested loan amount
    tenure_months : int
        Loan tenure in months
    interest_rate : float
        Annual interest rate (percentage)
    existing_emi : float
        Existing EMI obligations
    include_advanced : bool
        Include advanced ratios and metrics
    
    Returns:
    --------
    Dict[str, float]: Financial ratios and metrics
    """
    try:
        # Input validation
        if any(val < 0 for val in [income, loan_amount, tenure_months, interest_rate, existing_emi]):
            raise ValueError("All financial inputs must be non-negative")
        
        if income == 0:
            raise ValueError("Income cannot be zero")
        
        monthly_income = income / 12
        
        # Calculate EMI using exact formula
        if interest_rate == 0:
            emi = loan_amount / tenure_months
        else:
            r = interest_rate / 100 / 12  # Monthly interest rate
            if tenure_months == 0:
                emi = loan_amount
            else:
                numerator = loan_amount * r * ((1 + r) ** tenure_months)
                denominator = ((1 + r) ** tenure_months) - 1
                emi = numerator / denominator if denominator != 0 else loan_amount
        
        # Basic ratios
        ratios = {
            'loan_to_income': round(loan_amount / income, 4),
            'emi': round(emi, 2),
            'total_emi': round(emi + existing_emi, 2),
            'debt_burden_ratio': round((emi + existing_emi) / monthly_income, 4),
            'foir': round((emi + existing_emi) / monthly_income, 4),  # Same as debt_burden_ratio
            'disposable_income': round(monthly_income - (emi + existing_emi), 2),
            'emi_to_income': round(emi / monthly_income, 4),
            'total_interest': round((emi * tenure_months) - loan_amount, 2),
            'total_payment': round(emi * tenure_months, 2)
        }
        
        # Advanced ratios and metrics
        if include_advanced:
            ratios.update({
                'interest_to_principal_ratio': round(ratios['total_interest'] / loan_amount, 4) if loan_amount > 0 else 0,
                'effective_interest_rate': round((ratios['total_interest'] / loan_amount) / (tenure_months / 12) * 100, 4),
                'monthly_savings_potential': round(monthly_income * 0.2, 2),  # 20% savings target
                'debt_service_coverage': round(monthly_income / (emi + existing_emi), 2) if (emi + existing_emi) > 0 else float('inf'),
                'affordability_index': round((monthly_income - (emi + existing_emi)) / monthly_income, 4),
                'leverage_ratio': round(loan_amount / (income * tenure_months / 12), 4)
            })
            
            # Risk indicators
            ratios.update({
                'high_emi_risk': ratios['emi_to_income'] > 0.5,
                'high_debt_burden': ratios['debt_burden_ratio'] > 0.6,
                'stressed_disposable_income': ratios['disposable_income'] < monthly_income * 0.3
            })
        
        # Ensure all values are valid numbers
        for key, value in ratios.items():
            if isinstance(value, bool):
                continue
            if pd.isna(value) or np.isinf(value):
                ratios[key] = 0.0
            else:
                ratios[key] = round(float(value), 4)
        
        return ratios
        
    except Exception as e:
        logger.error(f"Error calculating financial ratios: {e}")
        return {
            'error': str(e),
            'loan_to_income': 0.0,
            'emi': 0.0,
            'debt_burden_ratio': 0.0
        }

# OPTIMIZATION 8: Enhanced risk categorization
def categorize_risk_level(probability: float, custom_thresholds: Optional[Dict[str, float]] = None) -> str:
    """
    Categorize risk level with configurable thresholds - OPTIMIZED
    
    Parameters:
    -----------
    probability : float
        Default probability (0-1)
    custom_thresholds : Optional[Dict[str, float]]
        Custom threshold configuration
    
    Returns:
    --------
    str: Risk level category
    """
    try:
        # Default thresholds
        thresholds = {
            'very_low': 0.02,
            'low': 0.05,
            'moderate': 0.10,
            'high': 0.20,
            'very_high': 0.35
        }
        
        # Override with custom thresholds if provided
        if custom_thresholds:
            thresholds.update(custom_thresholds)
        
        # Validate probability
        if not 0 <= probability <= 1:
            logger.warning(f"Invalid probability: {probability}. Clamping to [0,1]")
            probability = max(0, min(1, probability))
        
        # Categorize
        if probability <= thresholds['very_low']:
            return "Very Low"
        elif probability <= thresholds['low']:
            return "Low"
        elif probability <= thresholds['moderate']:
            return "Moderate"
        elif probability <= thresholds['high']:
            return "High"
        elif probability <= thresholds['very_high']:
            return "Very High"
        else:
            return "Extreme"
            
    except Exception as e:
        logger.error(f"Error categorizing risk level: {e}")
        return "Unknown"

# OPTIMIZATION 9: Enhanced currency formatting
def format_currency(amount: float, currency: str = "₹", locale: str = "indian", 
                   precision: int = 0, compact: bool = False) -> str:
    """
    Enhanced currency formatting with locale support - OPTIMIZED
    
    Parameters:
    -----------
    amount : float
        Numeric amount
    currency : str
        Currency symbol
    locale : str
        Formatting locale (indian, international)
    precision : int
        Decimal places
    compact : bool
        Use compact notation (1.2Cr instead of 1,20,00,000)
    
    Returns:
    --------
    str: Formatted currency string
    """
    try:
        if pd.isna(amount) or amount is None:
            return f"{currency}0"
        
        amount = float(amount)
        
        if compact and abs(amount) >= 1000:
            # Compact notation
            if abs(amount) >= 10000000:  # Crores
                compact_amount = amount / 10000000
                return f"{currency}{compact_amount:.1f}Cr"
            elif abs(amount) >= 100000:  # Lakhs
                compact_amount = amount / 100000
                return f"{currency}{compact_amount:.1f}L"
            elif abs(amount) >= 1000:  # Thousands
                compact_amount = amount / 1000
                return f"{currency}{compact_amount:.1f}K"
        
        # Standard formatting
        if locale == "indian":
            # Indian number system (lakhs, crores)
            if precision == 0:
                return f"{currency}{amount:,.0f}"
            else:
                return f"{currency}{amount:,.{precision}f}"
        else:
            # International formatting
            if precision == 0:
                return f"{currency}{amount:,.0f}"
            else:
                return f"{currency}{amount:,.{precision}f}"
                
    except Exception as e:
        logger.error(f"Error formatting currency: {e}")
        return f"{currency}0"

# OPTIMIZATION 10: Enhanced loan eligibility calculation
@performance_monitor
def calculate_loan_eligibility(
    income: float,
    existing_obligations: float = 0,
    foir_limit: float = 0.5,
    interest_rate: float = 12.0,
    max_tenure_months: int = 360,
    include_scenarios: bool = False
) -> Dict[str, Any]:
    """
    Calculate maximum loan eligibility with scenario analysis - OPTIMIZED
    
    Parameters:
    -----------
    income : float
        Annual income
    existing_obligations : float
        Monthly obligations
    foir_limit : float
        Maximum FOIR allowed
    interest_rate : float
        Annual interest rate
    max_tenure_months : int
        Maximum tenure allowed
    include_scenarios : bool
        Include multiple scenario analysis
    
    Returns:
    --------
    Dict[str, Any]: Comprehensive eligibility analysis
    """
    try:
        monthly_income = income / 12
        available_emi = (monthly_income * foir_limit) - existing_obligations
        
        if available_emi <= 0:
            return {
                'max_loan_amount': 0,
                'max_emi': 0,
                'recommended_tenure': 0,
                'debt_capacity': 0,
                'eligibility_status': 'Not Eligible',
                'reason': 'Insufficient income after existing obligations'
            }
        
        # Calculate for different tenures
        eligibility_by_tenure = {}
        r = interest_rate / 100 / 12
        
        standard_tenures = [36, 60, 120, 180, 240, 300, 360]
        valid_tenures = [t for t in standard_tenures if t <= max_tenure_months]
        
        for tenure in valid_tenures:
            if r > 0:
                max_loan = available_emi * (((1 + r) ** tenure) - 1) / (r * ((1 + r) ** tenure))
            else:
                max_loan = available_emi * tenure
            eligibility_by_tenure[tenure] = round(max_loan, 0)
        
        # Find optimal tenure (maximum loan amount)
        if eligibility_by_tenure:
            optimal_tenure = max(eligibility_by_tenure.keys(), 
                               key=lambda k: eligibility_by_tenure[k])
            max_loan_amount = eligibility_by_tenure[optimal_tenure]
        else:
            optimal_tenure = 0
            max_loan_amount = 0
        
        result = {
            'max_loan_amount': round(max_loan_amount, 0),
            'max_emi': round(available_emi, 0),
            'recommended_tenure': optimal_tenure,
            'debt_capacity': round(available_emi * 12, 0),
            'foir_utilization': round((available_emi + existing_obligations) / monthly_income, 4),
            'eligibility_by_tenure': eligibility_by_tenure,
            'eligibility_status': 'Eligible' if max_loan_amount > 0 else 'Not Eligible'
        }
        
        # Include scenario analysis
        if include_scenarios and max_loan_amount > 0:
            scenarios = {}
            
            # Different FOIR scenarios
            for foir in [0.4, 0.5, 0.6]:
                scenario_emi = (monthly_income * foir) - existing_obligations
                if scenario_emi > 0 and r > 0:
                    scenario_loan = scenario_emi * (((1 + r) ** optimal_tenure) - 1) / (r * ((1 + r) ** optimal_tenure))
                    scenarios[f'foir_{int(foir*100)}'] = round(scenario_loan, 0)
            
            # Different interest rate scenarios
            for rate in [10.0, 12.0, 15.0]:
                r_scenario = rate / 100 / 12
                if r_scenario > 0:
                    scenario_loan = available_emi * (((1 + r_scenario) ** optimal_tenure) - 1) / (r_scenario * ((1 + r_scenario) ** optimal_tenure))
                    scenarios[f'rate_{int(rate)}'] = round(scenario_loan, 0)
            
            result['scenarios'] = scenarios
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating loan eligibility: {e}")
        return {
            'max_loan_amount': 0,
            'max_emi': 0,
            'error': str(e),
            'eligibility_status': 'Error'
        }

# OPTIMIZATION 11: Enhanced business rules validation
@performance_monitor
def validate_business_rules(
    application_data: Dict[str, Any],
    config: Dict[str, Any],
    strict_mode: bool = False
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """
    Enhanced business rules validation with detailed reporting - OPTIMIZED
    
    Parameters:
    -----------
    application_data : Dict[str, Any]
        Application data
    config : Dict[str, Any]
        Configuration with business rules
    strict_mode : bool
        Enable strict validation mode
    
    Returns:
    --------
    Tuple[bool, List[str], List[str], Dict[str, Any]]: (is_valid, errors, warnings, details)
    """
    errors = []
    warnings = []
    validation_details = {}
    rules = config.get('business_rules', {})
    
    try:
        # Age validation
        age = application_data.get('age', 0)
        age_rules = rules.get('age', {})
        
        validation_details['age'] = {
            'value': age,
            'min_required': age_rules.get('min', 18),
            'max_allowed': age_rules.get('max', 70),
            'status': 'unknown'
        }
        
        if age < age_rules.get('min', 18):
            errors.append(f"Age below minimum requirement ({age} < {age_rules.get('min')})")
            validation_details['age']['status'] = 'error'
        elif age > age_rules.get('max', 70):
            errors.append(f"Age exceeds maximum limit ({age} > {age_rules.get('max')})")
            validation_details['age']['status'] = 'error'
        elif age < age_rules.get('warning_min', 21):
            warnings.append("Young applicant may require additional verification")
            validation_details['age']['status'] = 'warning'
        elif age > age_rules.get('warning_max', 65):
            warnings.append("Senior applicant - verify retirement income stability")
            validation_details['age']['status'] = 'warning'
        else:
            validation_details['age']['status'] = 'passed'
        
        # Income validation
        income = application_data.get('income', 0)
        income_rules = rules.get('income', {})
        
        validation_details['income'] = {
            'value': income,
            'min_required': income_rules.get('min', 100000),
            'currency': income_rules.get('currency', 'INR'),
            'status': 'unknown'
        }
        
        if income < income_rules.get('min', 100000):
            errors.append(f"Income below minimum requirement (₹{income:,} < ₹{income_rules.get('min'):,})")
            validation_details['income']['status'] = 'error'
        elif income < income_rules.get('warning_min', 300000):
            warnings.append("Low income may affect loan terms")
            validation_details['income']['status'] = 'warning'
        elif income > income_rules.get('max', 50000000):
            if strict_mode:
                warnings.append("Very high income - enhanced verification required")
            validation_details['income']['status'] = 'warning' if strict_mode else 'passed'
        else:
            validation_details['income']['status'] = 'passed'
        
        # Loan amount validation
        loan_amount = application_data.get('loan_amount', 0)
        loan_rules = rules.get('loan_amount', {})
        
        validation_details['loan_amount'] = {
            'value': loan_amount,
            'min_required': loan_rules.get('min', 50000),
            'max_allowed': loan_rules.get('max', 10000000),
            'status': 'unknown'
        }
        
        if loan_amount < loan_rules.get('min', 50000):
            errors.append(f"Loan amount below minimum (₹{loan_amount:,} < ₹{loan_rules.get('min'):,})")
            validation_details['loan_amount']['status'] = 'error'
        elif loan_amount > loan_rules.get('max', 10000000):
            errors.append(f"Loan amount exceeds maximum limit (₹{loan_amount:,} > ₹{loan_rules.get('max'):,})")
            validation_details['loan_amount']['status'] = 'error'
        elif loan_amount > loan_rules.get('warning_max', 5000000):
            warnings.append(f"High loan amount requires enhanced approval")
            validation_details['loan_amount']['status'] = 'warning'
        else:
            validation_details['loan_amount']['status'] = 'passed'
        
        # Loan-to-Income ratio validation
        if income > 0:
            lti_ratio = loan_amount / income
            lti_rules = rules.get('loan_to_income', {})
            
            validation_details['lti_ratio'] = {
                'value': round(lti_ratio, 2),
                'max_allowed': lti_rules.get('max_ratio', 10),
                'warning_threshold': lti_rules.get('warning_ratio', 5),
                'status': 'unknown'
            }
            
            if lti_ratio > lti_rules.get('max_ratio', 10):
                errors.append(f"Loan-to-income ratio exceeds regulatory limit ({lti_ratio:.1f}x > {lti_rules.get('max_ratio')}x)")
                validation_details['lti_ratio']['status'] = 'error'
            elif lti_ratio > lti_rules.get('warning_ratio', 5):
                warnings.append(f"High loan-to-income ratio ({lti_ratio:.1f}x)")
                validation_details['lti_ratio']['status'] = 'warning'
            else:
                validation_details['lti_ratio']['status'] = 'passed'
        
        # Additional validations in strict mode
        if strict_mode:
            # Tenure validation
            tenure = application_data.get('loan_tenure_months', 0)
            if tenure > 240:  # 20 years
                warnings.append("Very long tenure - verify applicant age compatibility")
            
            # Credit utilization validation
            credit_util = application_data.get('credit_utilization_ratio', 0)
            if credit_util > 90:
                errors.append("Extremely high credit utilization (>90%)")
            elif credit_util > 70:
                warnings.append("High credit utilization may indicate credit stress")
        
        is_valid = len(errors) == 0
        
        # Summary
        validation_details['summary'] = {
            'total_checks': len(validation_details) - 1,  # Exclude summary itself
            'passed': sum(1 for k, v in validation_details.items() 
                         if isinstance(v, dict) and v.get('status') == 'passed'),
            'warnings': sum(1 for k, v in validation_details.items() 
                           if isinstance(v, dict) and v.get('status') == 'warning'),
            'errors': sum(1 for k, v in validation_details.items() 
                         if isinstance(v, dict) and v.get('status') == 'error'),
            'is_valid': is_valid,
            'strict_mode': strict_mode
        }
        
        if not is_valid:
            logger.warning(f"Business rules validation failed: {len(errors)} errors, {len(warnings)} warnings")
        else:
            logger.info(f"✅ Business rules validation passed with {len(warnings)} warnings")
        
        return is_valid, errors, warnings, validation_details
        
    except Exception as e:
        logger.error(f"Error in business rules validation: {e}")
        errors.append(f"Validation system error: {str(e)}")
        return False, errors, warnings, validation_details

# OPTIMIZATION 12: Enhanced audit logging
def create_audit_log(
    action: str,
    user_id: str,
    details: Dict[str, Any],
    status: str = "SUCCESS",
    include_hash: bool = True,
    sensitive_fields: List[str] = None
) -> Dict[str, Any]:
    """
    Create comprehensive audit log entry with privacy protection - OPTIMIZED
    
    Parameters:
    -----------
    action : str
        Action performed
    user_id : str
        User identifier
    details : Dict[str, Any]
        Action details
    status : str
        Action status
    include_hash : bool
        Include data hash for integrity
    sensitive_fields : List[str]
        Fields to mask for privacy
    
    Returns:
    --------
    Dict[str, Any]: Comprehensive audit log entry
    """
    try:
        # Default sensitive fields
        if sensitive_fields is None:
            sensitive_fields = ['password', 'ssn', 'pan', 'aadhar', 'account_number']
        
        # Create masked details for logging
        masked_details = details.copy()
        for field in sensitive_fields:
            if field in masked_details:
                masked_details[field] = '***MASKED***'
        
        # Generate session ID
        session_id = hashlib.md5(f"{user_id}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'user_id': user_id,
            'session_id': session_id,
            'status': status,
            'details': masked_details,
            'user_agent': 'Credit-Risk-Platform/2.0',
            'ip_address': '127.0.0.1',  # Would be populated from request in web app
            'execution_time_ms': 0  # Would be populated by performance monitoring
        }
        
        # Add data integrity hash
        if include_hash:
            data_to_hash = f"{action}{user_id}{status}{json.dumps(details, sort_keys=True)}"
            log_entry['data_hash'] = hashlib.sha256(data_to_hash.encode()).hexdigest()[:16]
        
        # Add risk level based on action
        risk_levels = {
            'prediction': 'LOW',
            'batch_processing': 'MEDIUM',
            'model_update': 'HIGH',
            'configuration_change': 'HIGH',
            'user_management': 'CRITICAL'
        }
        log_entry['risk_level'] = risk_levels.get(action.lower(), 'MEDIUM')
        
        logger.info(f"📋 Audit: {action} by {user_id} - {status} [{log_entry['risk_level']}]")
        
        return log_entry
        
    except Exception as e:
        logger.error(f"Error creating audit log: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'user_id': user_id,
            'status': 'AUDIT_ERROR',
            'error': str(e)
        }

# OPTIMIZATION 13: Enhanced Excel export
@performance_monitor
def export_to_excel(
    dataframes: Dict[str, pd.DataFrame],
    filename: str = None,
    include_metadata: bool = True,
    apply_formatting: bool = True
) -> str:
    """
    Export multiple dataframes to Excel with enhanced formatting - OPTIMIZED
    
    Parameters:
    -----------
    dataframes : Dict[str, pd.DataFrame]
        Dictionary of sheet_name: dataframe
    filename : str
        Output filename (auto-generated if None)
    include_metadata : bool
        Include metadata sheet
    apply_formatting : bool
        Apply Excel formatting
    
    Returns:
    --------
    str: Path to created file or base64 string for download
    """
    try:
        if filename is None:
            filename = f"credit_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        # Create in-memory buffer for web apps
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#1e3d59',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })
            
            number_format = workbook.add_format({'num_format': '#,##0'})
            percentage_format = workbook.add_format({'num_format': '0.00%'})
            currency_format = workbook.add_format({'num_format': '₹#,##0'})
            
            # Export each dataframe
            for sheet_name, df in dataframes.items():
                # Truncate sheet name if too long
                sheet_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name
                
                df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
                worksheet = writer.sheets[sheet_name]
                
                if apply_formatting:
                    # Apply header formatting
                    for col_num, value in enumerate(df.columns.values):
                        worksheet.write(1, col_num, value, header_format)
                    
                    # Auto-adjust column widths
                    for i, col in enumerate(df.columns):
                        max_length = max(
                            df[col].astype(str).map(len).max(),
                            len(str(col))
                        ) + 2
                        worksheet.set_column(i, i, min(max_length, 50))
                    
                    # Apply number formatting based on column names
                    for i, col in enumerate(df.columns):
                        col_lower = col.lower()
                        if 'amount' in col_lower or 'income' in col_lower or 'emi' in col_lower:
                            worksheet.set_column(i, i, None, currency_format)
                        elif 'ratio' in col_lower or 'rate' in col_lower or 'percentage' in col_lower:
                            worksheet.set_column(i, i, None, percentage_format)
                        elif col_lower in ['count', 'score', 'age', 'months']:
                            worksheet.set_column(i, i, None, number_format)
            
            # Add metadata sheet
            if include_metadata:
                metadata = pd.DataFrame({
                    'Property': ['Generated On', 'Generated By', 'Total Sheets', 'Total Records', 'File Version'],
                    'Value': [
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Credit Risk AI Platform v2.0',
                        len(dataframes),
                        sum(len(df) for df in dataframes.values()),
                        '2.0'
                    ]
                })
                
                metadata.to_excel(writer, sheet_name='Metadata', index=False, startrow=1)
                metadata_sheet = writer.sheets['Metadata']
                
                if apply_formatting:
                    for col_num, value in enumerate(metadata.columns.values):
                        metadata_sheet.write(1, col_num, value, header_format)
                    metadata_sheet.set_column(0, 1, 20)
        
        # Return base64 encoded data for web download
        buffer.seek(0)
        excel_data = buffer.getvalue()
        b64_excel = base64.b64encode(excel_data).decode()
        
        logger.info(f"✅ Excel report created: {filename} ({len(excel_data)/1024:.1f} KB)")
        
        return b64_excel
        
    except Exception as e:
        logger.error(f"Error creating Excel report: {e}")
        return None

# Export all optimized functions
__all__ = [
    'load_config',
    'performance_monitor', 
    'validate_dataframe',
    'clean_currency_string',
    'calculate_statistics',
    'generate_application_id',
    'calculate_financial_ratios',
    'categorize_risk_level',
    'format_currency',
    'calculate_loan_eligibility',
    'validate_business_rules',
    'create_audit_log',
    'export_to_excel',
    'get_memory_usage'
]