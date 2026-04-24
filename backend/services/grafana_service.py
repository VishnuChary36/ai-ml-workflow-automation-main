"""
Steps 3 & 4: Grafana API Service

- POSTs dashboard JSON to the Grafana HTTP API
- Stores the returned dashboard UID
- Returns an embeddable ``<iframe>`` URL
- Manages datasource provisioning
"""

import logging
from typing import Dict, Any, Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)


class GrafanaAPIError(Exception):
    """Raised when Grafana API returns an error."""


class GrafanaService:
    """
    Thin wrapper around the Grafana HTTP API.

    Requires env vars:
        GRAFANA_URL       – e.g. http://localhost:3001
        GRAFANA_API_KEY   – a Service Account token with Editor perms
        GRAFANA_DATASOURCE_UID – UID of the pre-configured PostgreSQL datasource
    """

    def __init__(self):
        self.base_url = (settings.grafana_url or "http://localhost:3001").rstrip("/")
        self.api_key = settings.grafana_api_key or ""
        self.datasource_uid = settings.grafana_datasource_uid or ""
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ----- dashboard CRUD ----------------------------------------------------

    def create_or_update_dashboard(self, dashboard_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST ``/api/dashboards/db`` — create or overwrite a dashboard.

        Returns::

            {
                "uid": "abc123",
                "url": "/d/abc123/my-title",
                "embed_url": "http://grafana:3001/d/abc123/my-title?kiosk",
                "status": "success",
            }
        """
        url = f"{self.base_url}/api/dashboards/db"
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, json=dashboard_json, headers=self._headers)

        if resp.status_code not in (200, 201):
            logger.error("Grafana API error: %s %s", resp.status_code, resp.text)
            raise GrafanaAPIError(
                f"Grafana returned {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        uid = data.get("uid", "")
        dash_url = data.get("url", "")

        return {
            "uid": uid,
            "url": dash_url,
            "embed_url": f"{self.base_url}{dash_url}?kiosk",
            "id": data.get("id"),
            "status": data.get("status", "success"),
            "version": data.get("version"),
        }

    def get_dashboard(self, uid: str) -> Dict[str, Any]:
        """GET ``/api/dashboards/uid/{uid}``."""
        url = f"{self.base_url}/api/dashboards/uid/{uid}"
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, headers=self._headers)
        if resp.status_code != 200:
            raise GrafanaAPIError(f"Dashboard {uid} not found: {resp.text}")
        return resp.json()

    def delete_dashboard(self, uid: str) -> bool:
        """DELETE ``/api/dashboards/uid/{uid}``."""
        url = f"{self.base_url}/api/dashboards/uid/{uid}"
        with httpx.Client(timeout=15) as client:
            resp = client.delete(url, headers=self._headers)
        return resp.status_code == 200

    def get_embed_url(self, uid: str, slug: str = "") -> str:
        """Build the embeddable kiosk-mode URL for an iframe."""
        path = f"/d/{uid}/{slug}" if slug else f"/d/{uid}"
        return f"{self.base_url}{path}?orgId=1&kiosk"

    # ----- datasource --------------------------------------------------------

    def ensure_postgres_datasource(self) -> str:
        """
        Create the Postgres datasource if it doesn't exist.
        Returns the datasource UID.
        """
        if self.datasource_uid:
            return self.datasource_uid

        # Parse DB URL from settings
        db_url = settings.database_url

        # Skip if not PostgreSQL (e.g. SQLite)
        if not db_url.startswith("postgresql"):
            logger.warning("Database is not PostgreSQL (%s) — skipping Grafana datasource creation", db_url.split("://")[0])
            return ""

        # postgresql://user:pass@host:port/dbname
        from urllib.parse import urlparse
        parsed = urlparse(db_url)

        ds_payload = {
            "name": "MLWorkflow-Postgres",
            "type": "postgres",
            "access": "proxy",
            "url": f"{parsed.hostname}:{parsed.port or 5432}",
            "user": parsed.username or "postgres",
            "database": parsed.path.lstrip("/") or "mlworkflow",
            "secureJsonData": {"password": parsed.password or "postgres"},
            "jsonData": {
                "sslmode": "disable",
                "maxOpenConns": 10,
                "maxIdleConns": 5,
                "connMaxLifetime": 14400,
                "postgresVersion": 1400,
                "timescaledb": False,
            },
        }

        url = f"{self.base_url}/api/datasources"
        with httpx.Client(timeout=15) as client:
            resp = client.post(url, json=ds_payload, headers=self._headers)

        if resp.status_code in (200, 201):
            data = resp.json()
            self.datasource_uid = data.get("datasource", {}).get("uid", data.get("uid", ""))
            logger.info("Created Grafana datasource: %s", self.datasource_uid)
            return self.datasource_uid
        elif resp.status_code == 409:  # already exists
            # Fetch by name
            with httpx.Client(timeout=15) as client:
                resp2 = client.get(
                    f"{self.base_url}/api/datasources/name/MLWorkflow-Postgres",
                    headers=self._headers,
                )
            if resp2.status_code == 200:
                self.datasource_uid = resp2.json().get("uid", "")
                return self.datasource_uid
        raise GrafanaAPIError(f"Failed to create datasource: {resp.text}")

    # ----- health check ------------------------------------------------------

    def check_health(self) -> Dict[str, Any]:
        """Quick health check against the Grafana instance."""
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    f"{self.base_url}/api/health", headers=self._headers
                )
            if resp.status_code == 200:
                return {"status": "ok", "grafana_url": self.base_url, **resp.json()}
            return {"status": "error", "code": resp.status_code, "detail": resp.text}
        except httpx.ConnectError:
            return {"status": "unreachable", "grafana_url": self.base_url}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
