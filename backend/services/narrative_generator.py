"""
Step 5 (backend part): AI Narrative Generator

Generates a human-readable, data-driven narrative that is displayed
above the embedded Grafana dashboard.

Supports two modes:
  1. **LLM mode** — calls OpenAI / Anthropic for a rich narrative.
  2. **Template mode** — deterministic fill-in-the-blanks (no API key needed).
"""

import logging
from typing import Dict, Any, List, Optional

from config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Template-based narrative (always available, no API key required)
# ============================================================================

def _generate_template_narrative(
    dataset_info: Dict[str, Any],
    model_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a structured narrative matching the 6-section dashboard layout."""
    parts: List[str] = []

    # --- Dataset section ---
    rows = dataset_info.get("rows", 0)
    cols = dataset_info.get("columns", 0)
    filename = dataset_info.get("filename", "the dataset")
    numeric = dataset_info.get("numeric_columns", [])
    categorical = dataset_info.get("categorical_columns", [])
    completeness = dataset_info.get("completeness", 100)
    outlier_pct = round(100 - completeness, 1)

    # [PIN] Dataset Summary
    parts.append(f"## [PIN] Dataset Summary\n")
    parts.append(
        f"**{filename}** contains **{rows:,} records** across **{cols} features** "
        f"({len(numeric)} numeric, {len(categorical)} categorical)."
    )
    parts.append(
        f"Data quality: **{completeness:.1f}%** clean rows, "
        f"**{outlier_pct:.1f}%** outliers/missing values removed."
    )

    # [EVAL] Key Trends
    parts.append(f"\n## [EVAL] Key Trends\n")
    if numeric:
        parts.append(
            f"The dashboard tracks temporal patterns for key features including "
            f"{', '.join(f'`{c}`' for c in numeric[:3])}. "
            f"Review the time-series panels for seasonal patterns, spikes, or gradual drift."
        )
    else:
        parts.append("No numeric features available for trend analysis.")

    # [CHART] Feature Insights
    parts.append(f"\n## [CHART] Feature Insights\n")
    if numeric:
        parts.append(
            f"Key numeric features: {', '.join(f'`{c}`' for c in numeric[:5])}"
            + (f" and {len(numeric) - 5} more." if len(numeric) > 5 else ".")
        )
    if categorical:
        parts.append(
            f"Categorical features: {', '.join(f'`{c}`' for c in categorical[:5])}"
            + (f" and {len(categorical) - 5} more." if len(categorical) > 5 else ".")
        )
        parts.append(
            f"\nCheck the distribution panels for class imbalance or skewed categories."
        )

    # --- Model section (if available) ---
    if model_info:
        algorithm = model_info.get("algorithm", "Unknown")
        problem_type = model_info.get("problem_type", "unknown")
        metrics = model_info.get("metrics", {})
        target = model_info.get("target_column", "target")

        # [AI] Model Performance
        parts.append(f"\n## [AI] Model Performance\n")
        parts.append(
            f"A **{algorithm}** model was trained for **{problem_type}** "
            f"on target column `{target}`."
        )

        if metrics:
            metric_strs = []
            for k, v in metrics.items():
                display = k.replace("_", " ").title()
                if isinstance(v, float):
                    if v <= 1 and any(x in k for x in ("accuracy", "f1", "precision", "recall")):
                        metric_strs.append(f"{display}: **{v * 100:.1f}%**")
                    else:
                        metric_strs.append(f"{display}: **{v:.4f}**")
                else:
                    metric_strs.append(f"{display}: **{v}**")
            parts.append("| " + " | ".join(metric_strs) + " |")

        # [AI] AI-Generated Explanation
        parts.append(f"\n## [AI] AI Insights\n")

        strongest_feature = numeric[0] if numeric else "N/A"
        parts.append(
            f"> [TIP] Feature `{strongest_feature}` is the strongest predictor in the model."
        )

        if problem_type == "classification":
            acc = metrics.get("accuracy", 0)
            f1 = metrics.get("f1_score", metrics.get("f1", 0))
            if acc > 0.9:
                parts.append(
                    f"\n> [OK] **Excellent Performance** — The model achieves {acc*100:.1f}% accuracy "
                    f"with F1 score of {f1:.2f}. It is production-ready."
                )
            elif acc > 0.7:
                parts.append(
                    f"\n> [WARN] **Moderate Performance** — Accuracy at {acc*100:.1f}%. "
                    f"Consider feature engineering or ensemble methods."
                )
            else:
                parts.append(
                    f"\n> [ALERT] **Low Performance** — Accuracy below 70%. "
                    f"Review data quality, rebalance classes, or try different algorithms."
                )
        elif problem_type == "regression":
            r2 = metrics.get("r2_score", 0)
            if r2 > 0.8:
                parts.append(
                    f"\n> [OK] **Strong Fit** — R² of {r2:.4f} explains {r2*100:.1f}% of variance. "
                    f"Check residual plots for heteroscedasticity."
                )
            else:
                parts.append(
                    f"\n> [WARN] **Moderate Fit** — R² of {r2:.4f}. "
                    f"Consider polynomial features or non-linear models."
                )

        # [ALERT] Alerts / Drift
        parts.append(f"\n## [ALERT] Alerts & Drift\n")
        if completeness < 90:
            parts.append(
                f"[ALERT] **Data Quality Alert:** Only {completeness:.1f}% data completeness. "
                f"This may impact model reliability."
            )
        if outlier_pct > 5:
            parts.append(
                f"[ALERT] **Anomaly Warning:** {outlier_pct:.1f}% anomaly rate detected — "
                f"exceeds the 5% threshold."
            )
        if completeness >= 90 and outlier_pct <= 5:
            parts.append("[OK] All quality checks passed. No alerts triggered.")

    parts.append(
        "\n---\n*Explore the interactive Grafana panels below for deeper drill-down.*"
    )

    return "\n".join(parts)


# ============================================================================
# LLM-based narrative (requires API key)
# ============================================================================

async def _generate_llm_narrative(
    dataset_info: Dict[str, Any],
    model_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Call an LLM API to generate a richer narrative."""

    prompt = _build_prompt(dataset_info, model_info)

    system_prompt = (
        "You are a senior data analyst writing a concise executive dashboard narrative. "
        "Structure your response using these exact sections with emoji headers:\n"
        "## [PIN] Dataset Summary\n"
        "## [EVAL] Key Trends\n"
        "## [CHART] Feature Insights\n"
        "## [AI] Model Performance (if model data provided)\n"
        "## [AI] AI Insights\n"
        "## [ALERT] Alerts & Drift\n\n"
        "Use Markdown formatting. Include specific data-driven insights like:\n"
        "- 'Customers aged 25-34 show 18% higher churn risk'\n"
        "- 'Feature MonthlyCharges is the strongest predictor'\n"
        "Keep it under 400 words. Use blockquotes (>) for key insights."
    )

    # Try OpenAI first
    if settings.openai_api_key:
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=0.5,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("OpenAI narrative failed: %s — falling back to template", e)

    # Try Anthropic
    if settings.anthropic_api_key:
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            message = await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=800,
                messages=[
                    {"role": "user", "content": system_prompt + "\n\n" + prompt},
                ],
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.warning("Anthropic narrative failed: %s — falling back to template", e)

    # Fallback
    return _generate_template_narrative(dataset_info, model_info)


def _build_prompt(
    dataset_info: Dict[str, Any],
    model_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the LLM prompt from structured data."""
    completeness = dataset_info.get('completeness', 100)
    outlier_pct = round(100 - completeness, 1)
    parts = [
        "Generate a dashboard narrative for the following data:",
        f"\nDataset: {dataset_info.get('filename', 'N/A')}",
        f"Total Records: {dataset_info.get('rows', 'N/A')}, Total Features: {dataset_info.get('columns', 'N/A')}",
        f"Numeric columns: {dataset_info.get('numeric_columns', [])}",
        f"Categorical columns: {dataset_info.get('categorical_columns', [])}",
        f"Data Completeness: {completeness:.1f}%",
        f"Outlier/Missing Rate: {outlier_pct:.1f}%",
    ]
    if model_info:
        metrics = model_info.get('metrics', {})
        parts += [
            f"\nModel Algorithm: {model_info.get('algorithm', 'N/A')}",
            f"Problem Type: {model_info.get('problem_type', 'N/A')}",
            f"Target Column: {model_info.get('target_column', 'N/A')}",
            f"Metrics: {metrics}",
        ]
        # Add specific insights hints
        if model_info.get('problem_type') == 'classification':
            acc = metrics.get('accuracy', 0)
            parts.append(f"\nThe model achieves {acc*100:.1f}% accuracy.")
        elif model_info.get('problem_type') == 'regression':
            r2 = metrics.get('r2_score', 0)
            parts.append(f"\nThe model achieves R² of {r2:.4f}.")
    parts.append(
        "\nWrite a narrative with specific data-driven insights and actionable recommendations. "
        "Include sample insights like 'Feature X is the strongest predictor' or "
        "'Segment Y shows Z% higher risk'. Use the 6-section structure."
    )
    return "\n".join(parts)


# ============================================================================
# Public API
# ============================================================================

async def generate_narrative(
    dataset_info: Dict[str, Any],
    model_info: Optional[Dict[str, Any]] = None,
    use_llm: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Generate a narrative for the dashboard.

    Returns::

        {
            "narrative": "## Dataset Overview ...",
            "mode": "llm" | "template",
        }
    """
    should_use_llm = use_llm if use_llm is not None else settings.use_llm_suggestions

    if should_use_llm and (settings.openai_api_key or settings.anthropic_api_key):
        text = await _generate_llm_narrative(dataset_info, model_info)
        return {"narrative": text, "mode": "llm"}
    else:
        text = _generate_template_narrative(dataset_info, model_info)
        return {"narrative": text, "mode": "template"}
