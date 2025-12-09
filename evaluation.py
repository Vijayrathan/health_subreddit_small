import pandas as pd
from sklearn.metrics import classification_report
from typing import Union, Optional


def evaluate_predictions(
    ground_truth: Union[pd.DataFrame, str],
    predictions: Union[pd.DataFrame, str],
    ground_truth_label_col: str = 'ground_truth_label',
    prediction_label_col: str = 'unified_claim_status',
    merge_key: Optional[str] = None,
    filter_triples_cols: Optional[list] = None,
    verbose: bool = True
) -> str:
    """
    Evaluate predictions against ground truth labels.
    
    Parameters:
    -----------
    ground_truth : pd.DataFrame or str
        Ground truth dataframe or path to parquet file
    predictions : pd.DataFrame or str
        Predictions dataframe or path to parquet file
    ground_truth_label_col : str, default 'ground_truth_label'
        Column name for ground truth labels
    prediction_label_col : str, default 'unified_claim_status'
        Column name for prediction labels
    merge_key : str, optional
        If provided, merge dataframes on this column. Otherwise, join by index.
    filter_triples_cols : list, optional
        List of column names to filter on (rows with "[]" will be dropped).
        Default: ['triples', 'cui_triples']
    verbose : bool, default True
        Whether to print the evaluation report
    
    Returns:
    --------
    str
        Classification report as a string
    """
    # Load dataframes if paths are provided
    if isinstance(ground_truth, str):
        df_annotated = pd.read_parquet(ground_truth)
    else:
        df_annotated = ground_truth.copy()
    
    if isinstance(predictions, str):
        df_semmed = pd.read_parquet(predictions)
    else:
        df_semmed = predictions.copy()
    
    # --- STEP 1: ALIGN THE DATA ---
    if merge_key is not None:
        # Merge on common ID column
        merged_df = pd.merge(df_annotated, df_semmed, on=merge_key, suffixes=('_true', '_pred'))
    else:
        # Join by index (assumes same order)
        merged_df = df_annotated.join(df_semmed, lsuffix='_true', rsuffix='_pred')
    
    # --- STEP 2: FILTER THE MERGED DATA ---
    if filter_triples_cols is None:
        filter_triples_cols = ['triples', 'cui_triples']
  
    
    # --- STEP 3: RUN EVALUATION ---
    if verbose:
        print(f"Final aligned dataset size: {len(merged_df)}")
    
    # Extract columns from the aligned dataframe
    y_true = merged_df[ground_truth_label_col]
    y_pred = merged_df[prediction_label_col]
    
    # Generate classification report (as dictionary for CSV export, as string for display)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_str = classification_report(y_true, y_pred)
    
    # Convert to DataFrame and save to CSV
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv("classification_report_semmed.csv")
    
    if verbose:
        print("\n--- Detailed Report ---")
        print(report_str)
    
    return report_str


# Example usage:
if __name__ == "__main__":
    # Example 1: Using file paths
    #pubmed evaluation
    # evaluate_predictions(
    #     ground_truth="./ground_truth/reddit_health_full_annotated.parquet",
    #     predictions="./pubmed_verifier/triples_pubmed_unified_claims.parquet"
    # )
    #semmed evaluation
    evaluate_predictions(
        ground_truth="./ground_truth/reddit_health_full_annotated.parquet",
        predictions="./semmed_verifier/triples_semmed_unified_claims.parquet"
    )
    # Example 2: Using dataframes directly
    # df_gt = pd.read_parquet("../ground_truth/reddit_health_full_annotated.parquet")
    # df_pred = pd.read_parquet("../semmed_verifier/triples_semmed_unified_claims.parquet")
    # evaluate_predictions(df_gt, df_pred)
    
    # Example 3: With custom column names and merge key
    # evaluate_predictions(
    #     ground_truth=df_gt,
    #     predictions=df_pred,
    #     ground_truth_label_col='label',
    #     prediction_label_col='predicted_label',
    #     merge_key='id'
    # )