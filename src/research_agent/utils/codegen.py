"""Shared prompt context for Python code generation in flows."""

from __future__ import annotations


def build_code_generation_user_prompt(
    *,
    query: str,
    dataset_path: str,
    plan: str,
    dataset_summary: str,
    code_prompt: str,
    run_id: str,
    iteration: int = 0,
    feedback_context: str = "",
) -> str:
    """Build the user message for the code-generation LLM step."""
    figures_dir = f"reports/figures/{run_id}"
    feedback_block = f"\nPrior feedback:\n{feedback_context}\n" if feedback_context else ""
    return (
        f"{code_prompt}\n\n"
        f"--- Task ---\n"
        f"Query: {query}\n"
        f"Dataset path (use in pd.read_csv or os.environ['DATASET_PATH']): {dataset_path}\n"
        f"Never hardcode absolute machine paths.\n"
        f"Figures directory (use for plt.savefig): os.environ['FIGURES_DIR']\n"
        f"  (resolves to project path: {figures_dir}/)\n"
        f"Iteration: {iteration}\n"
        f"Dataset summary:\n{dataset_summary}\n"
        f"Plan:\n{plan}\n"
        f"{feedback_block}"
        "Implementation notes:\n"
        "- Use sklearn Pipeline + ColumnTransformer for mixed numeric/categorical columns.\n"
        "- For classification, use train_test_split(..., stratify=y) when both classes exist.\n"
        "- For categorical imputation, use SimpleImputer(strategy='most_frequent') on 2D arrays.\n"
        "- Pandas: never use df['col'].fillna(x, inplace=True); use df['col'] = df['col'].fillna(x).\n"
        "- Pandas 3.x: do not pass axis=1 with columns= in drop(); use df.drop(columns=[...]).\n"
        "- Pandas 3.x: prefer include=['object','string'] over include='object' in select_dtypes.\n"
        "- Outlier/IQR tasks: split train/test first; compute bounds on training data only, "
        "then clip train and test separately.\n"
        "- After imputation, print missing counts before/after to verify success.\n"
        "- Create FIGURES_DIR with os.makedirs(..., exist_ok=True) before savefig.\n"
        "Output only a single ```python code block```."
    )
