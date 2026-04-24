"""
Step 1: Postgres Ingest Service

Parses uploaded datasets, preprocesses with pandas, and writes rows into
Postgres so Grafana can query them directly via the PostgreSQL datasource.

Three target tables are managed:
  - ingested_data   : raw/preprocessed feature rows
  - model_predictions : prediction vs actual for each trained model
  - grafana_dashboards : tracks provisioned Grafana dashboard UIDs
"""

import uuid
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, JSON, Text,
    Boolean, create_engine, MetaData, Table, inspect,
    text,
)
from sqlalchemy.orm import Session

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PG_TYPE_MAP = {
    "int64": "BIGINT",
    "int32": "INTEGER",
    "float64": "DOUBLE PRECISION",
    "float32": "REAL",
    "bool": "BOOLEAN",
    "datetime64[ns]": "TIMESTAMP",
    "object": "TEXT",
    "category": "TEXT",
}


def _pg_col_type(dtype) -> str:
    return _PG_TYPE_MAP.get(str(dtype), "TEXT")


def _sanitise_col(name: str) -> str:
    """Make column name Postgres-safe."""
    return name.strip().replace(" ", "_").replace("-", "_").replace(".", "_").lower()


# ---------------------------------------------------------------------------
# Ingest Service
# ---------------------------------------------------------------------------

class PostgresIngestService:
    """
    Writes pandas DataFrames into Postgres tables so they can be queried
    by Grafana's built-in PostgreSQL datasource.
    """

    INGEST_TABLE = "ingested_data"
    PREDICTIONS_TABLE = "model_predictions"

    def __init__(self, db: Session):
        self.db = db

    # ----- public API -------------------------------------------------------

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        task_id: str,
        table_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Write a preprocessed DataFrame into Postgres.

        Columns are sanitised and mapped to appropriate PG types.
        Each row gets a ``_dataset_id`` and ``_task_id`` tag so multiple
        datasets can coexist in the same table.

        Returns summary dict with rows written and column mapping.
        """
        table = table_name or self.INGEST_TABLE
        df = df.copy()

        # Sanitise column names
        col_map = {c: _sanitise_col(c) for c in df.columns}
        df.rename(columns=col_map, inplace=True)

        # Add metadata columns
        df["_dataset_id"] = dataset_id
        df["_task_id"] = task_id
        df["_ingested_at"] = datetime.utcnow()
        df["_row_id"] = [str(uuid.uuid4())[:12] for _ in range(len(df))]

        # Ensure table exists with the right schema
        self._ensure_table(table, df)

        # Bulk insert via pandas → SQLAlchemy engine
        engine = self.db.get_bind()
        rows_before = self._count_rows(table, dataset_id)
        df.to_sql(table, engine, if_exists="append", index=False, method="multi", chunksize=500)
        rows_after = self._count_rows(table, dataset_id)

        logger.info("Ingested %d rows into %s for dataset %s", len(df), table, dataset_id)

        return {
            "table": table,
            "dataset_id": dataset_id,
            "task_id": task_id,
            "rows_written": rows_after - rows_before,
            "columns": list(df.columns),
            "column_types": {c: _pg_col_type(df[c].dtype) for c in df.columns},
        }

    def write_predictions(
        self,
        model_id: str,
        task_id: str,
        dataset_id: str,
        y_actual: pd.Series,
        y_predicted: pd.Series,
        target_column: str,
        problem_type: str,
        algorithm: str,
        feature_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Write prediction-vs-actual rows into ``model_predictions``.

        This powers the *Predicted vs Actual* Grafana panel.
        """
        table = self.PREDICTIONS_TABLE

        pred_df = pd.DataFrame({
            "model_id": model_id,
            "task_id": task_id,
            "dataset_id": dataset_id,
            "target_column": target_column,
            "problem_type": problem_type,
            "algorithm": algorithm,
            "actual": y_actual.values,
            "predicted": y_predicted.values,
            "residual": (y_actual.values - y_predicted.values) if problem_type == "regression" else None,
            "correct": (y_actual.values == y_predicted.values) if problem_type == "classification" else None,
            "row_index": range(len(y_actual)),
            "created_at": datetime.utcnow(),
        })

        # Optionally attach top-N features for drill-down
        if feature_df is not None:
            for col in list(feature_df.columns)[:5]:
                safe = _sanitise_col(f"feat_{col}")
                pred_df[safe] = feature_df[col].values

        self._ensure_table(table, pred_df)

        engine = self.db.get_bind()
        pred_df.to_sql(table, engine, if_exists="append", index=False, method="multi", chunksize=500)

        logger.info("Wrote %d prediction rows for model %s", len(pred_df), model_id)

        return {
            "table": table,
            "model_id": model_id,
            "rows_written": len(pred_df),
            "problem_type": problem_type,
        }

    def get_ingested_columns(self, dataset_id: str, table: Optional[str] = None) -> List[str]:
        """Return column names for a given dataset in the ingested table."""
        tbl = table or self.INGEST_TABLE
        try:
            result = self.db.execute(
                text(f"SELECT * FROM {tbl} WHERE _dataset_id = :did LIMIT 1"),
                {"did": dataset_id},
            )
            return [c for c in result.keys() if not c.startswith("_")]
        except Exception:
            return []

    def get_numeric_columns(self, dataset_id: str) -> List[str]:
        """Return numeric column names for a dataset."""
        tbl = self.INGEST_TABLE
        try:
            inspector = inspect(self.db.get_bind())
            columns = inspector.get_columns(tbl)
            numeric_types = {"BIGINT", "INTEGER", "DOUBLE PRECISION", "REAL", "NUMERIC", "FLOAT"}
            return [
                c["name"] for c in columns
                if str(c["type"]).upper() in numeric_types and not c["name"].startswith("_")
            ]
        except Exception:
            return []

    # ----- internals --------------------------------------------------------

    def _ensure_table(self, name: str, df: pd.DataFrame):
        """Create table if missing; add columns if DataFrame has new ones."""
        engine = self.db.get_bind()
        insp = inspect(engine)
        if not insp.has_table(name):
            # Create via pandas trick with 0 rows
            df.head(0).to_sql(name, engine, if_exists="fail", index=False)
            logger.info("Created table %s", name)
        else:
            existing = {c["name"] for c in insp.get_columns(name)}
            for col in df.columns:
                if col not in existing:
                    col_type = _pg_col_type(df[col].dtype)
                    self.db.execute(text(f'ALTER TABLE {name} ADD COLUMN "{col}" {col_type}'))
                    self.db.commit()
                    logger.info("Added column %s (%s) to %s", col, col_type, name)

    def _count_rows(self, table: str, dataset_id: str) -> int:
        try:
            result = self.db.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE _dataset_id = :did"),
                {"did": dataset_id},
            )
            return result.scalar() or 0
        except Exception:
            return 0
