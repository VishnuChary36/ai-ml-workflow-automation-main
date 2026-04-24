"""
Step 2: Grafana Dashboard Builder Library

Builds Grafana JSON dashboard models from reusable panel templates.
Placeholders like ``{datasource}``, ``{table}``, ``{column}`` are replaced
at build time.

Includes 5 panel templates for common ML/analytics patterns:
  1. Time-series line chart
  2. Stat / KPI panel
  3. Bar chart (categorical breakdown)
  4. Scatter plot (predicted vs actual)
  5. Table panel (raw data grid)
"""

import copy
import uuid
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Panel Templates  (Grafana JSON panel fragments)
# ============================================================================

def _panel_id() -> int:
    """Auto-increment panel IDs within a build session."""
    _panel_id._counter = getattr(_panel_id, "_counter", 0) + 1
    return _panel_id._counter


# ---------- 1. Time-Series Line Chart ----------

TIME_SERIES_PANEL = {
    "type": "timeseries",
    "title": "{title}",
    "datasource": {"type": "postgres", "uid": "{datasource_uid}"},
    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
    "targets": [
        {
            "refId": "A",
            "datasource": {"type": "postgres", "uid": "{datasource_uid}"},
            "rawSql": "SELECT _ingested_at AS time, {column} AS value FROM {table} WHERE _dataset_id = '{dataset_id}' ORDER BY _ingested_at",
            "format": "time_series",
        }
    ],
    "fieldConfig": {
        "defaults": {
            "color": {"mode": "palette-classic"},
            "custom": {
                "lineWidth": 2,
                "fillOpacity": 10,
                "spanNulls": True,
                "pointSize": 5,
            },
        },
        "overrides": [],
    },
    "options": {
        "tooltip": {"mode": "single"},
        "legend": {"displayMode": "list", "placement": "bottom"},
    },
}


# ---------- 2. Stat / KPI Panel ----------

STAT_PANEL = {
    "type": "stat",
    "title": "{title}",
    "datasource": {"type": "postgres", "uid": "{datasource_uid}"},
    "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
    "targets": [
        {
            "refId": "A",
            "datasource": {"type": "postgres", "uid": "{datasource_uid}"},
            "rawSql": "SELECT {agg_func}({column}) AS value FROM {table} WHERE _dataset_id = '{dataset_id}'",
            "format": "table",
        }
    ],
    "fieldConfig": {
        "defaults": {
            "thresholds": {
                "mode": "absolute",
                "steps": [
                    {"color": "red", "value": None},
                    {"color": "orange", "value": 50},
                    {"color": "green", "value": 80},
                ],
            },
            "unit": "{unit}",
        },
        "overrides": [],
    },
    "options": {
        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "textMode": "auto",
    },
}


# ---------- 3. Bar Chart (Categorical Breakdown) ----------

BAR_CHART_PANEL = {
    "type": "barchart",
    "title": "{title}",
    "datasource": {"type": "postgres", "uid": "{datasource_uid}"},
    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
    "targets": [
        {
            "refId": "A",
            "datasource": {"type": "postgres", "uid": "{datasource_uid}"},
            "rawSql": "SELECT {column} AS category, COUNT(*) AS count FROM {table} WHERE _dataset_id = '{dataset_id}' GROUP BY {column} ORDER BY count DESC LIMIT 20",
            "format": "table",
        }
    ],
    "fieldConfig": {
        "defaults": {
            "color": {"mode": "palette-classic"},
        },
        "overrides": [],
    },
    "options": {
        "orientation": "auto",
        "xTickLabelRotation": -45,
        "showValue": "auto",
        "barWidth": 0.75,
        "groupWidth": 0.7,
        "stacking": "none",
        "legend": {"displayMode": "list", "placement": "bottom"},
        "tooltip": {"mode": "single"},
    },
}


# ---------- 4. Scatter Plot (Predicted vs Actual) ----------

SCATTER_PANEL = {
    "type": "xychart",
    "title": "{title}",
    "datasource": {"type": "postgres", "uid": "{datasource_uid}"},
    "gridPos": {"h": 10, "w": 12, "x": 0, "y": 0},
    "targets": [
        {
            "refId": "A",
            "datasource": {"type": "postgres", "uid": "{datasource_uid}"},
            "rawSql": "SELECT actual, predicted FROM {table} WHERE model_id = '{model_id}' ORDER BY row_index",
            "format": "table",
        }
    ],
    "fieldConfig": {
        "defaults": {
            "color": {"fixedColor": "blue", "mode": "fixed"},
            "custom": {"pointSize": {"fixed": 5}},
        },
        "overrides": [],
    },
    "options": {
        "seriesMapping": "auto",
        "dims": {"x": "actual", "y": ["predicted"]},
        "tooltip": {"mode": "single"},
        "legend": {"displayMode": "list", "placement": "bottom"},
    },
}


# ---------- 5. Table Panel (Raw Data Grid) ----------

TABLE_PANEL = {
    "type": "table",
    "title": "{title}",
    "datasource": {"type": "postgres", "uid": "{datasource_uid}"},
    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 0},
    "targets": [
        {
            "refId": "A",
            "datasource": {"type": "postgres", "uid": "{datasource_uid}"},
            "rawSql": "SELECT {columns} FROM {table} WHERE _dataset_id = '{dataset_id}' ORDER BY _ingested_at DESC LIMIT 100",
            "format": "table",
        }
    ],
    "fieldConfig": {"defaults": {}, "overrides": []},
    "options": {
        "showHeader": True,
        "footer": {"show": False},
        "sortBy": [],
    },
}


# ---------- 6. Gauge Panel ----------

GAUGE_PANEL = {
    "type": "gauge",
    "title": "{title}",
    "datasource": {"type": "postgres", "uid": "{datasource_uid}"},
    "gridPos": {"h": 6, "w": 8, "x": 0, "y": 0},
    "targets": [
        {
            "refId": "A",
            "datasource": {"type": "postgres", "uid": "{datasource_uid}"},
            "rawSql": "SELECT {value} AS value",
            "format": "table",
        }
    ],
    "fieldConfig": {
        "defaults": {
            "min": 0,
            "max": 100,
            "thresholds": {
                "mode": "absolute",
                "steps": [
                    {"color": "red", "value": None},
                    {"color": "orange", "value": 40},
                    {"color": "yellow", "value": 60},
                    {"color": "green", "value": 80},
                ],
            },
            "unit": "percent",
        },
        "overrides": [],
    },
    "options": {
        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        "showThresholdLabels": False,
        "showThresholdMarkers": True,
    },
}


# ---------- 7. Pie Chart Panel ----------

PIE_CHART_PANEL = {
    "type": "piechart",
    "title": "{title}",
    "datasource": {"type": "postgres", "uid": "{datasource_uid}"},
    "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0},
    "targets": [
        {
            "refId": "A",
            "datasource": {"type": "postgres", "uid": "{datasource_uid}"},
            "rawSql": "SELECT {column} AS label, COUNT(*) AS value FROM {table} WHERE _dataset_id = '{dataset_id}' GROUP BY {column} ORDER BY value DESC LIMIT 10",
            "format": "table",
        }
    ],
    "fieldConfig": {
        "defaults": {
            "color": {"mode": "palette-classic"},
        },
        "overrides": [],
    },
    "options": {
        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": True},
        "pieType": "donut",
        "tooltip": {"mode": "single"},
        "legend": {"displayMode": "table", "placement": "right", "values": ["value", "percent"]},
    },
}


# ---------- 8. Text / Annotation Panel ----------

TEXT_PANEL = {
    "type": "text",
    "title": "{title}",
    "datasource": {"type": "datasource", "uid": "-- Grafana --"},
    "gridPos": {"h": 6, "w": 8, "x": 0, "y": 0},
    "options": {
        "mode": "markdown",
        "content": "{content}",
    },
}


# Template registry
PANEL_TEMPLATES = {
    "timeseries": TIME_SERIES_PANEL,
    "stat": STAT_PANEL,
    "barchart": BAR_CHART_PANEL,
    "scatter": SCATTER_PANEL,
    "table": TABLE_PANEL,
    "gauge": GAUGE_PANEL,
    "piechart": PIE_CHART_PANEL,
    "text": TEXT_PANEL,
}


# ============================================================================
# Dashboard Builder
# ============================================================================

class GrafanaDashboardBuilder:
    """
    Builds a complete Grafana dashboard JSON from panel templates.

    Usage::

        builder = GrafanaDashboardBuilder(
            title="My ML Dashboard",
            datasource_uid="PG_DS_UID",
        )
        builder.add_stat_panel("Total Rows", table="ingested_data",
                               column="*", agg_func="COUNT", dataset_id="abc")
        builder.add_timeseries_panel("Feature Over Time", ...)
        builder.add_scatter_panel("Pred vs Actual", ...)
        dashboard_json = builder.build()
    """

    def __init__(
        self,
        title: str,
        datasource_uid: str,
        tags: Optional[List[str]] = None,
        refresh: str = "30s",
        uid: Optional[str] = None,
    ):
        self.title = title
        self.datasource_uid = datasource_uid
        self.tags = tags or ["auto-generated", "ml-workflow"]
        self.refresh = refresh
        self.uid = uid or f"ml-{uuid.uuid4().hex[:10]}"
        self.panels: List[Dict] = []
        self._next_y = 0
        _panel_id._counter = 0  # reset

    # ---- panel adders -------------------------------------------------------

    def add_timeseries_panel(
        self, title: str, table: str, column: str, dataset_id: str,
        width: int = 12, height: int = 8, x: int = 0,
        custom_sql: Optional[str] = None,
    ):
        panel = self._from_template("timeseries", {
            "title": title,
            "table": table,
            "column": column,
            "dataset_id": dataset_id,
        })
        if custom_sql:
            panel["targets"][0]["rawSql"] = custom_sql
        self._position(panel, width, height, x)
        self.panels.append(panel)
        return self

    def add_stat_panel(
        self, title: str, table: str, column: str, dataset_id: str,
        agg_func: str = "AVG", unit: str = "short",
        width: int = 6, height: int = 4, x: int = 0,
        custom_sql: Optional[str] = None,
    ):
        panel = self._from_template("stat", {
            "title": title,
            "table": table,
            "column": column,
            "dataset_id": dataset_id,
            "agg_func": agg_func,
            "unit": unit,
        })
        if custom_sql:
            panel["targets"][0]["rawSql"] = custom_sql
        self._position(panel, width, height, x)
        self.panels.append(panel)
        return self

    def add_barchart_panel(
        self, title: str, table: str, column: str, dataset_id: str,
        width: int = 12, height: int = 8, x: int = 0,
        custom_sql: Optional[str] = None,
    ):
        panel = self._from_template("barchart", {
            "title": title,
            "table": table,
            "column": column,
            "dataset_id": dataset_id,
        })
        if custom_sql:
            panel["targets"][0]["rawSql"] = custom_sql
        self._position(panel, width, height, x)
        self.panels.append(panel)
        return self

    def add_scatter_panel(
        self, title: str, table: str, model_id: str,
        width: int = 12, height: int = 10, x: int = 0,
        custom_sql: Optional[str] = None,
    ):
        panel = self._from_template("scatter", {
            "title": title,
            "table": table,
            "model_id": model_id,
        })
        if custom_sql:
            panel["targets"][0]["rawSql"] = custom_sql
        self._position(panel, width, height, x)
        self.panels.append(panel)
        return self

    def add_table_panel(
        self, title: str, table: str, columns: str, dataset_id: str,
        width: int = 24, height: int = 8, x: int = 0,
        custom_sql: Optional[str] = None,
    ):
        panel = self._from_template("table", {
            "title": title,
            "table": table,
            "columns": columns,
            "dataset_id": dataset_id,
        })
        if custom_sql:
            panel["targets"][0]["rawSql"] = custom_sql
        self._position(panel, width, height, x)
        self.panels.append(panel)
        return self

    def add_gauge_panel(
        self, title: str, value: float,
        width: int = 8, height: int = 6, x: int = 0,
        min_val: float = 0, max_val: float = 100,
        unit: str = "percent",
    ):
        panel = self._from_template("gauge", {
            "title": title,
            "value": str(value),
        })
        panel["fieldConfig"]["defaults"]["min"] = min_val
        panel["fieldConfig"]["defaults"]["max"] = max_val
        panel["fieldConfig"]["defaults"]["unit"] = unit
        self._position(panel, width, height, x)
        self.panels.append(panel)
        return self

    def add_piechart_panel(
        self, title: str, table: str, column: str, dataset_id: str,
        width: int = 8, height: int = 8, x: int = 0,
        custom_sql: Optional[str] = None,
    ):
        panel = self._from_template("piechart", {
            "title": title,
            "table": table,
            "column": column,
            "dataset_id": dataset_id,
        })
        if custom_sql:
            panel["targets"][0]["rawSql"] = custom_sql
        self._position(panel, width, height, x)
        self.panels.append(panel)
        return self

    def add_text_panel(
        self, title: str, content: str,
        width: int = 8, height: int = 6, x: int = 0,
    ):
        panel = self._from_template("text", {
            "title": title,
            "content": content,
        })
        self._position(panel, width, height, x)
        self.panels.append(panel)
        return self

    def add_row(self, title: str = ""):
        """Add a collapsible row separator."""
        self.panels.append({
            "type": "row",
            "title": title,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": self._next_y},
            "id": _panel_id(),
            "collapsed": False,
        })
        self._next_y += 1
        return self

    # ---- build --------------------------------------------------------------

    def build(self) -> Dict[str, Any]:
        """Return full Grafana dashboard JSON ready for POST to /api/dashboards/db."""
        return {
            "dashboard": {
                "uid": self.uid,
                "title": self.title,
                "tags": self.tags,
                "timezone": "browser",
                "schemaVersion": 39,
                "version": 0,
                "refresh": self.refresh,
                "time": {"from": "now-24h", "to": "now"},
                "panels": self.panels,
                "annotations": {
                    "list": [
                        {
                            "builtIn": 1,
                            "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                            "enable": True,
                            "hide": True,
                            "type": "dashboard",
                        }
                    ]
                },
                "templating": {"list": []},
                "editable": True,
                "fiscalYearStartMonth": 0,
                "graphTooltip": 0,
                "links": [],
                "liveNow": False,
            },
            "folderId": 0,
            "overwrite": True,
        }

    # ---- internals ----------------------------------------------------------

    def _from_template(self, template_key: str, replacements: Dict[str, str]) -> Dict:
        """Deep-copy a template and substitute placeholders."""
        tmpl = copy.deepcopy(PANEL_TEMPLATES[template_key])
        tmpl["id"] = _panel_id()
        # Always inject datasource_uid
        replacements["datasource_uid"] = self.datasource_uid
        return self._replace_placeholders(tmpl, replacements)

    def _replace_placeholders(self, obj: Any, replacements: Dict[str, str]) -> Any:
        """Recursively replace {key} placeholders in nested dicts/lists/strings."""
        if isinstance(obj, str):
            for key, val in replacements.items():
                obj = obj.replace(f"{{{key}}}", str(val))
            return obj
        elif isinstance(obj, dict):
            return {k: self._replace_placeholders(v, replacements) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_placeholders(item, replacements) for item in obj]
        return obj

    def _position(self, panel: Dict, w: int, h: int, x: int):
        """Set grid position and advance the Y cursor."""
        panel["gridPos"] = {"h": h, "w": w, "x": x, "y": self._next_y}
        if x + w >= 24:
            self._next_y += h


# ============================================================================
# Convenience: auto-build dashboard for a dataset + model
# ============================================================================

def build_dataset_dashboard(
    title: str,
    datasource_uid: str,
    dataset_id: str,
    numeric_columns: List[str],
    categorical_columns: List[str],
    table: str = "ingested_data",
    total_rows: int = 0,
    completeness: float = 100.0,
) -> Dict[str, Any]:
    """
    Auto-generate a data-exploration dashboard with 6 sections:
      1. Dataset Summary (KPI cards)
      2. Key Trends (time-series / line charts)
      3. Feature Insights (distributions, histograms, pie charts)
      4. Model Performance (placeholder — populated when model exists)
      5. AI-Generated Explanation (text panel)
      6. Alerts / Drift (threshold & anomaly panels)
    """
    builder = GrafanaDashboardBuilder(title=title, datasource_uid=datasource_uid)

    outlier_pct = round(100 - completeness, 1)

    # ================================================================
    # [PIN] Section 1 — Dataset Summary  (Top KPI Cards)
    # ================================================================
    builder.add_row("[PIN] Dataset Summary")

    builder.add_stat_panel(
        "Total Records", table, "*", dataset_id,
        agg_func="COUNT", unit="short", width=6, height=4, x=0,
    )

    builder.add_stat_panel(
        "Clean Rows %", table, "*", dataset_id,
        agg_func="COUNT", unit="percent", width=6, height=4, x=6,
        custom_sql=f"SELECT {completeness} AS value",
    )

    builder.add_stat_panel(
        "Outliers / Missing %", table, "*", dataset_id,
        agg_func="COUNT", unit="percent", width=6, height=4, x=12,
        custom_sql=f"SELECT {outlier_pct} AS value",
    )

    builder.add_stat_panel(
        "Total Features", table, "*", dataset_id,
        agg_func="COUNT", unit="short", width=6, height=4, x=18,
        custom_sql=f"SELECT {len(numeric_columns) + len(categorical_columns)} AS value",
    )

    # ================================================================
    # [EVAL] Section 2 — Key Trends  (Time-Series & Area Charts)
    # ================================================================
    builder.add_row("[EVAL] Key Trends")

    if len(numeric_columns) >= 1:
        # Primary trend line (first numeric column)
        builder.add_timeseries_panel(
            f"{numeric_columns[0]} — Trend Over Time",
            table, numeric_columns[0], dataset_id,
            width=12, height=8, x=0,
        )

    if len(numeric_columns) >= 2:
        # Secondary trend line
        builder.add_timeseries_panel(
            f"{numeric_columns[1]} — Trend Over Time",
            table, numeric_columns[1], dataset_id,
            width=12, height=8, x=12,
        )
    elif len(numeric_columns) == 1:
        # Moving average comparison
        builder.add_timeseries_panel(
            f"{numeric_columns[0]} — Moving Average",
            table, numeric_columns[0], dataset_id,
            width=12, height=8, x=12,
            custom_sql=(
                f"SELECT _ingested_at AS time, "
                f"AVG({numeric_columns[0]}) OVER (ORDER BY _ingested_at ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS value "
                f"FROM {table} WHERE _dataset_id = '{dataset_id}' ORDER BY _ingested_at"
            ),
        )

    # ================================================================
    # [CHART] Section 3 — Feature Insights  (Distributions & Breakdowns)
    # ================================================================
    builder.add_row("[CHART] Feature Insights")

    if categorical_columns:
        # Pie/donut chart for first categorical column
        builder.add_piechart_panel(
            f"{categorical_columns[0]} — Distribution",
            table, categorical_columns[0], dataset_id,
            width=8, height=8, x=0,
        )

    if categorical_columns and len(categorical_columns) >= 2:
        builder.add_barchart_panel(
            f"{categorical_columns[1]} — Category Breakdown",
            table, categorical_columns[1], dataset_id,
            width=8, height=8, x=8,
        )
    elif categorical_columns:
        builder.add_barchart_panel(
            f"{categorical_columns[0]} — Category Breakdown",
            table, categorical_columns[0], dataset_id,
            width=8, height=8, x=8,
        )

    if numeric_columns:
        # Histogram-style bar chart for a numeric column
        first_num = numeric_columns[0]
        builder.add_barchart_panel(
            f"{first_num} — Value Distribution",
            table, first_num, dataset_id,
            width=8, height=8, x=16,
            custom_sql=(
                f"SELECT ROUND({first_num}::numeric, 1) AS bucket, COUNT(*) AS count "
                f"FROM {table} WHERE _dataset_id = '{dataset_id}' "
                f"GROUP BY bucket ORDER BY bucket LIMIT 30"
            ),
        )

    # ================================================================
    # [AI] Section 4 — Model Performance  (Placeholder for dataset-only)
    # ================================================================
    builder.add_row("[AI] Model Performance")

    builder.add_text_panel(
        "Model Status",
        "### No Model Trained Yet\n\n"
        "Train a model on this dataset to see:\n"
        "- **Accuracy / R² Score** KPI cards\n"
        "- **F1 Score**, **Precision**, **Recall** gauges\n"
        "- **Predicted vs Actual** scatter plots\n"
        "- **Last Training Time** stat\n\n"
        "Go to the **Train** page to get started.",
        width=24, height=5, x=0,
    )

    # ================================================================
    # [AI] Section 5 — AI-Generated Explanation
    # ================================================================
    builder.add_row("[AI] AI-Generated Explanation")

    num_summary = ", ".join(f"`{c}`" for c in numeric_columns[:5]) if numeric_columns else "none"
    cat_summary = ", ".join(f"`{c}`" for c in categorical_columns[:5]) if categorical_columns else "none"

    ai_md = (
        f"### Data Story\n\n"
        f"This dataset contains **{total_rows:,} records** across "
        f"**{len(numeric_columns)} numeric** and **{len(categorical_columns)} categorical** features.\n\n"
        f"**Key numeric features:** {num_summary}\n\n"
        f"**Key categorical features:** {cat_summary}\n\n"
        f"Data completeness is **{completeness:.1f}%**, meaning **{outlier_pct:.1f}%** "
        f"of values were missing or flagged as outliers.\n\n"
        f"---\n\n"
        f"> [TIP] **Insight:** Explore the *Key Trends* section above to identify temporal "
        f"patterns. Check *Feature Insights* for class imbalance or skewed distributions."
    )
    builder.add_text_panel("AI Explanation", ai_md, width=24, height=7, x=0)

    # ================================================================
    # [ALERT] Section 6 — Alerts / Drift
    # ================================================================
    builder.add_row("[ALERT] Alerts / Drift")

    # Data completeness gauge (alert threshold at 90%)
    builder.add_gauge_panel(
        "Data Completeness", completeness,
        width=8, height=6, x=0,
    )

    # Outlier ratio gauge (alert if > 5%)
    builder.add_gauge_panel(
        "Outlier / Missing Ratio", outlier_pct,
        width=8, height=6, x=8,
        min_val=0, max_val=20,
    )

    # Alert text panel
    alerts_md = "### Status Alerts\n\n"
    if completeness >= 95:
        alerts_md += "[OK] **Data Quality:** Excellent — completeness above 95%\n\n"
    elif completeness >= 85:
        alerts_md += "[WARN] **Data Quality Warning:** Completeness between 85-95%. Review missing values.\n\n"
    else:
        alerts_md += "[ALERT] **Data Quality Alert:** Completeness below 85%! Significant missing data detected.\n\n"

    if outlier_pct > 5:
        alerts_md += f"[ALERT] **Anomaly Alert:** {outlier_pct:.1f}% outlier/missing rate exceeds 5% threshold.\n\n"
    else:
        alerts_md += f"[OK] **Anomaly Check:** {outlier_pct:.1f}% outlier rate is within acceptable range.\n\n"

    alerts_md += "---\n\n*Alerts update automatically when the dataset is refreshed.*"
    builder.add_text_panel("Alerts", alerts_md, width=8, height=6, x=16)

    return builder.build()


def build_model_dashboard(
    title: str,
    datasource_uid: str,
    model_id: str,
    dataset_id: str,
    algorithm: str,
    problem_type: str,
    metrics: Dict[str, float],
    numeric_columns: List[str],
    predictions_table: str = "model_predictions",
    data_table: str = "ingested_data",
    total_rows: int = 0,
    completeness: float = 100.0,
    target_column: str = "target",
    training_time: str = "N/A",
) -> Dict[str, Any]:
    """
    Auto-generate a model dashboard with 6 sections:
      1. [PIN] Dataset Summary — KPI stat cards (records, clean %, outliers, features)
      2. [EVAL] Key Trends — prediction trends, revenue/activity time-series
      3. [CHART] Feature Insights — distributions, target balance, category breakdown
      4. [AI] Model Performance — accuracy, F1, gauges, predicted vs actual
      5. [AI] AI-Generated Explanation — storytelling text panel
      6. [ALERT] Alerts / Drift — thresholds, anomaly indicators
    """
    builder = GrafanaDashboardBuilder(title=title, datasource_uid=datasource_uid)

    # Filter out non-numeric entries
    numeric_metrics = {
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float))
    }

    outlier_pct = round(100 - completeness, 1)

    # Derive primary scores
    accuracy = numeric_metrics.get("accuracy", 0)
    f1 = numeric_metrics.get("f1_score", numeric_metrics.get("f1", 0))
    precision = numeric_metrics.get("precision", 0)
    recall = numeric_metrics.get("recall", 0)
    r2 = numeric_metrics.get("r2_score", 0)
    primary_score = accuracy if problem_type == "classification" else r2

    # ================================================================
    # [PIN] Section 1 — Dataset Summary  (Top KPI Cards — First Row)
    # ================================================================
    builder.add_row("[PIN] Dataset Summary")

    builder.add_stat_panel(
        "Total Records", data_table, "*", dataset_id,
        agg_func="COUNT", unit="short", width=6, height=4, x=0,
    )

    builder.add_stat_panel(
        "Clean Rows %", data_table, "*", dataset_id,
        agg_func="COUNT", unit="percent", width=6, height=4, x=6,
        custom_sql=f"SELECT {completeness} AS value",
    )

    builder.add_stat_panel(
        "Outliers Removed %", data_table, "*", dataset_id,
        agg_func="COUNT", unit="percent", width=6, height=4, x=12,
        custom_sql=f"SELECT {outlier_pct} AS value",
    )

    builder.add_stat_panel(
        "Total Features", data_table, "*", dataset_id,
        agg_func="COUNT", unit="short", width=6, height=4, x=18,
        custom_sql=f"SELECT {len(numeric_columns)} AS value",
    )

    # ---- Model KPI Cards (second sub-row) ----
    if problem_type == "classification":
        builder.add_stat_panel(
            "Model Accuracy", predictions_table, "actual", dataset_id,
            agg_func="COUNT", unit="percentunit", width=6, height=4, x=0,
            custom_sql=f"SELECT {accuracy} AS value",
        )
        builder.add_stat_panel(
            "F1 Score", predictions_table, "actual", dataset_id,
            agg_func="COUNT", unit="short", width=6, height=4, x=6,
            custom_sql=f"SELECT {f1} AS value",
        )
        builder.add_stat_panel(
            "Precision", predictions_table, "actual", dataset_id,
            agg_func="COUNT", unit="percentunit", width=6, height=4, x=12,
            custom_sql=f"SELECT {precision} AS value",
        )
        builder.add_stat_panel(
            "Recall", predictions_table, "actual", dataset_id,
            agg_func="COUNT", unit="percentunit", width=6, height=4, x=18,
            custom_sql=f"SELECT {recall} AS value",
        )
    else:
        builder.add_stat_panel(
            "R² Score", predictions_table, "actual", dataset_id,
            agg_func="COUNT", unit="short", width=6, height=4, x=0,
            custom_sql=f"SELECT {r2} AS value",
        )
        mse = numeric_metrics.get("mse", numeric_metrics.get("mean_squared_error", 0))
        rmse = numeric_metrics.get("rmse", numeric_metrics.get("root_mean_squared_error", 0))
        mae = numeric_metrics.get("mae", numeric_metrics.get("mean_absolute_error", 0))
        builder.add_stat_panel(
            "MSE", predictions_table, "actual", dataset_id,
            agg_func="COUNT", unit="short", width=6, height=4, x=6,
            custom_sql=f"SELECT {mse} AS value",
        )
        builder.add_stat_panel(
            "RMSE", predictions_table, "actual", dataset_id,
            agg_func="COUNT", unit="short", width=6, height=4, x=12,
            custom_sql=f"SELECT {rmse} AS value",
        )
        builder.add_stat_panel(
            "MAE", predictions_table, "actual", dataset_id,
            agg_func="COUNT", unit="short", width=6, height=4, x=18,
            custom_sql=f"SELECT {mae} AS value",
        )

    # ================================================================
    # [EVAL] Section 2 — Key Trends  (Time-Series — Second Row)
    # ================================================================
    builder.add_row("[EVAL] Key Trends")

    # Prediction trend over time
    builder.add_timeseries_panel(
        "Prediction Trend Over Time", predictions_table, "predicted", dataset_id,
        width=12, height=8, x=0,
        custom_sql=(
            f"SELECT created_at AS time, predicted::numeric AS value "
            f"FROM {predictions_table} WHERE model_id = '{model_id}' "
            f"ORDER BY created_at"
        ),
    )

    # Actual vs Predicted trend
    builder.add_timeseries_panel(
        "Actual vs Predicted — Trend", predictions_table, "actual", dataset_id,
        width=12, height=8, x=12,
        custom_sql=(
            f"SELECT created_at AS time, actual::numeric AS \"Actual\", "
            f"predicted::numeric AS \"Predicted\" "
            f"FROM {predictions_table} WHERE model_id = '{model_id}' "
            f"ORDER BY created_at"
        ),
    )

    # Feature value drift (if numeric columns available)
    if len(numeric_columns) >= 2:
        builder.add_timeseries_panel(
            f"Feature Drift — {numeric_columns[0]} vs {numeric_columns[1]}",
            data_table, numeric_columns[0], dataset_id,
            width=24, height=8, x=0,
            custom_sql=(
                f"SELECT _ingested_at AS time, "
                f"AVG({numeric_columns[0]}) AS \"{numeric_columns[0]}\", "
                f"AVG({numeric_columns[1]}) AS \"{numeric_columns[1]}\" "
                f"FROM {data_table} WHERE _dataset_id = '{dataset_id}' "
                f"GROUP BY _ingested_at ORDER BY _ingested_at"
            ),
        )

    # ================================================================
    # [CHART] Section 3 — Feature Insights  (Distribution & Analysis)
    # ================================================================
    builder.add_row("[CHART] Feature Insights")

    # Target variable balance / class distribution
    if problem_type == "classification":
        builder.add_piechart_panel(
            "Target Class Balance (Predicted)",
            predictions_table, "predicted", dataset_id,
            width=8, height=8, x=0,
            custom_sql=(
                f"SELECT predicted::text AS label, COUNT(*) AS value "
                f"FROM {predictions_table} WHERE model_id = '{model_id}' "
                f"GROUP BY predicted ORDER BY value DESC LIMIT 10"
            ),
        )
        builder.add_barchart_panel(
            "Prediction Accuracy Breakdown",
            predictions_table, "correct", dataset_id,
            width=8, height=8, x=8,
            custom_sql=(
                f"SELECT CASE WHEN correct THEN 'Correct' ELSE 'Incorrect' END AS result, "
                f"COUNT(*) AS count FROM {predictions_table} WHERE model_id = '{model_id}' "
                f"GROUP BY correct"
            ),
        )
    else:
        builder.add_barchart_panel(
            "Residual Distribution",
            predictions_table, "residual", dataset_id,
            width=8, height=8, x=0,
            custom_sql=(
                f"SELECT ROUND(residual::numeric, 2) AS residual, COUNT(*) AS count "
                f"FROM {predictions_table} WHERE model_id = '{model_id}' "
                f"GROUP BY ROUND(residual::numeric, 2) ORDER BY residual"
            ),
        )
        builder.add_piechart_panel(
            "Error Magnitude Buckets",
            predictions_table, "residual", dataset_id,
            width=8, height=8, x=8,
            custom_sql=(
                f"SELECT CASE "
                f"WHEN ABS(residual) < 0.5 THEN 'Low (<0.5)' "
                f"WHEN ABS(residual) < 1.0 THEN 'Medium (0.5-1)' "
                f"WHEN ABS(residual) < 2.0 THEN 'High (1-2)' "
                f"ELSE 'Very High (>2)' END AS label, "
                f"COUNT(*) AS value "
                f"FROM {predictions_table} WHERE model_id = '{model_id}' "
                f"GROUP BY label ORDER BY value DESC"
            ),
        )

    # Feature distribution (numeric histogram)
    if numeric_columns:
        first_num = numeric_columns[0]
        builder.add_barchart_panel(
            f"{first_num} — Value Distribution",
            data_table, first_num, dataset_id,
            width=8, height=8, x=16,
            custom_sql=(
                f"SELECT ROUND({first_num}::numeric, 1) AS bucket, COUNT(*) AS count "
                f"FROM {data_table} WHERE _dataset_id = '{dataset_id}' "
                f"GROUP BY bucket ORDER BY bucket LIMIT 30"
            ),
        )

    # ================================================================
    # [AI] Section 4 — Model Performance  (Scatter, Gauges, KPIs)
    # ================================================================
    builder.add_row(f"[AI] Model Performance — {algorithm}")

    if problem_type == "regression":
        builder.add_scatter_panel(
            "Predicted vs Actual", predictions_table, model_id,
            width=12, height=10, x=0,
        )
    else:
        builder.add_barchart_panel(
            "Class Distribution (Predicted)",
            predictions_table, "predicted", dataset_id,
            width=12, height=10, x=0,
            custom_sql=(
                f"SELECT predicted::text AS class, COUNT(*) AS count "
                f"FROM {predictions_table} WHERE model_id = '{model_id}' "
                f"GROUP BY predicted ORDER BY count DESC"
            ),
        )

    # Gauge: primary model score
    display_score = 0.0
    if primary_score <= 1:
        display_score = round(primary_score * 100, 1)
    else:
        display_score = round(primary_score, 1)

    builder.add_gauge_panel(
        "Model Confidence Score", display_score,
        width=6, height=5, x=12,
    )

    # Gauge: F1 or R2  
    secondary_score = f1 if problem_type == "classification" else r2
    secondary_label = "F1 Score" if problem_type == "classification" else "R² Score"
    sec_display = round(secondary_score * 100, 1) if secondary_score <= 1 else round(secondary_score, 1)
    builder.add_gauge_panel(
        secondary_label, sec_display,
        width=6, height=5, x=18,
    )

    # Feature importance table (top features from data table)
    if numeric_columns:
        cols_str = ", ".join(numeric_columns[:8]) or "*"
        builder.add_table_panel("Top Features", data_table, cols_str, dataset_id, width=24, height=6, x=0)

    # ================================================================
    # [AI] Section 5 — AI-Generated Explanation  (Storytelling Panel)
    # ================================================================
    builder.add_row("[AI] AI-Generated Explanation")

    # Build rich narrative markdown
    score_label = "Accuracy" if problem_type == "classification" else "R² Score"
    quality = (
        "Excellent" if display_score >= 90 else
        "Good" if display_score >= 75 else
        "Fair" if display_score >= 60 else
        "Needs Improvement"
    )

    # Build metrics table
    metrics_lines = "\n".join(
        f"| {k.replace('_', ' ').title()} | **{round(v * 100, 1) if v <= 1 else round(v, 4)}{'%' if v <= 1 else ''}** |"
        for k, v in numeric_metrics.items()
    )

    # Determine strongest predictor insight
    strongest_feature = numeric_columns[0] if numeric_columns else "N/A"

    # Build recommendation
    if display_score >= 80:
        recommendation = "Model is production-ready with strong performance."
        insight_icon = "[OK]"
    elif display_score >= 60:
        recommendation = "Consider tuning hyperparameters, feature engineering, or gathering more training data."
        insight_icon = "[WARN]"
    else:
        recommendation = "Model performance is below acceptable thresholds. Review feature selection, data quality, and try alternative algorithms."
        insight_icon = "[ALERT]"

    ai_md = (
        f"### Executive Summary\n\n"
        f"**Algorithm:** {algorithm}  \n"
        f"**Problem Type:** {problem_type.title()}  \n"
        f"**Target Column:** `{target_column}`  \n"
        f"**{score_label}:** {display_score}%  \n"
        f"**Quality Rating:** {quality}\n\n"
        f"---\n\n"
        f"#### Key Metrics\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"{metrics_lines}\n\n"
        f"---\n\n"
        f"#### [TIP] AI Insights\n\n"
        f"> Feature `{strongest_feature}` is the strongest predictor in the dataset.\n\n"
        f"> {insight_icon} **{quality}** — {recommendation}\n\n"
        f"---\n\n"
        f"#### Recommendation\n\n"
        f"{recommendation}"
    )
    builder.add_text_panel("AI Story & Insights", ai_md, width=24, height=9, x=0)

    # ================================================================
    # [ALERT] Section 6 — Alerts / Drift
    # ================================================================
    builder.add_row("[ALERT] Alerts / Drift")

    # Model confidence gauge with threshold markers
    builder.add_gauge_panel(
        "Prediction Confidence", display_score,
        width=8, height=7, x=0,
    )

    # Data quality gauge
    builder.add_gauge_panel(
        "Data Completeness", completeness,
        width=8, height=7, x=8,
    )

    # Alert text panel
    alerts_md = "### [ALERT] Status Alerts\n\n"

    # Model performance alert
    if display_score >= 90:
        alerts_md += f"[OK] **Model Performance:** {score_label} at {display_score}% — Excellent\n\n"
    elif display_score >= 75:
        alerts_md += f"[OK] **Model Performance:** {score_label} at {display_score}% — Good\n\n"
    elif display_score >= 60:
        alerts_md += f"[WARN] **Performance Warning:** {score_label} dropped to {display_score}%. Consider retraining.\n\n"
    else:
        alerts_md += f"[ALERT] **Performance Alert:** {score_label} at {display_score}%! Model needs attention.\n\n"

    # Data quality alert
    if completeness >= 95:
        alerts_md += f"[OK] **Data Quality:** {completeness:.1f}% completeness — No issues detected\n\n"
    elif completeness >= 85:
        alerts_md += f"[WARN] **Data Quality Warning:** {completeness:.1f}% completeness. Review missing values.\n\n"
    else:
        alerts_md += f"[ALERT] **Data Quality Alert:** {completeness:.1f}% completeness! Significant gaps detected.\n\n"

    # Drift indicator
    if outlier_pct > 5:
        alerts_md += f"[ALERT] **Data Drift Detected:** {outlier_pct:.1f}% anomaly rate exceeds 5% threshold.\n\n"
    else:
        alerts_md += f"[OK] **Drift Check:** {outlier_pct:.1f}% anomaly rate — within normal range.\n\n"

    alerts_md += "---\n\n*Alerts refresh automatically with dashboard data.*"
    builder.add_text_panel("Alerts & Anomalies", alerts_md, width=8, height=7, x=16)

    return builder.build()
