"""
Secrets management abstraction layer.

Supports multiple backends:
- Environment variables (default, for development)
- AWS Secrets Manager
- Google Cloud Secret Manager
- HashiCorp Vault
- Azure Key Vault

Usage:
    from core.secrets import secrets
    
    # Get a secret
    api_key = secrets.get("OPENAI_API_KEY")
    
    # Get with default
    debug = secrets.get("DEBUG", "false")
"""

import os
import json
from typing import Optional, Dict, Any
from enum import Enum
from functools import lru_cache


class SecretsBackend(str, Enum):
    """Supported secrets backends."""
    ENV = "env"
    AWS = "aws"
    GCP = "gcp"
    VAULT = "vault"
    AZURE = "azure"


class SecretsManager:
    """
    Unified secrets management with support for multiple backends.
    
    Automatically caches secrets to minimize external API calls.
    """
    
    def __init__(
        self,
        backend: SecretsBackend = SecretsBackend.ENV,
        cache_enabled: bool = True,
        aws_region: str = "us-east-1",
        gcp_project_id: Optional[str] = None,
        vault_addr: Optional[str] = None,
        vault_token: Optional[str] = None,
        azure_vault_url: Optional[str] = None,
    ):
        self.backend = backend
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, str] = {}
        
        # Backend-specific configuration
        self.aws_region = aws_region
        self.gcp_project_id = gcp_project_id or os.getenv("GCP_PROJECT_ID")
        self.vault_addr = vault_addr or os.getenv("VAULT_ADDR", "http://localhost:8200")
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self.azure_vault_url = azure_vault_url or os.getenv("AZURE_VAULT_URL")
        
        # Lazy-loaded clients
        self._aws_client = None
        self._gcp_client = None
        self._vault_client = None
        self._azure_client = None
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value.
        
        Args:
            key: Secret name/key
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        # Check cache first
        if self.cache_enabled and key in self._cache:
            return self._cache[key]
        
        value = None
        
        try:
            if self.backend == SecretsBackend.ENV:
                value = self._get_from_env(key)
            elif self.backend == SecretsBackend.AWS:
                value = self._get_from_aws(key)
            elif self.backend == SecretsBackend.GCP:
                value = self._get_from_gcp(key)
            elif self.backend == SecretsBackend.VAULT:
                value = self._get_from_vault(key)
            elif self.backend == SecretsBackend.AZURE:
                value = self._get_from_azure(key)
        except Exception as e:
            print(f"Warning: Failed to get secret '{key}': {e}")
            value = None
        
        # Cache the result
        if value is not None and self.cache_enabled:
            self._cache[key] = value
        
        return value if value is not None else default
    
    def get_json(self, key: str, default: Optional[Dict] = None) -> Optional[Dict]:
        """Get a secret and parse as JSON."""
        value = self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default
        return default
    
    def clear_cache(self):
        """Clear the secrets cache."""
        self._cache.clear()
    
    def _get_from_env(self, key: str) -> Optional[str]:
        """Get secret from environment variables."""
        return os.getenv(key)
    
    def _get_from_aws(self, key: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        if self._aws_client is None:
            try:
                import boto3
                self._aws_client = boto3.client(
                    "secretsmanager",
                    region_name=self.aws_region
                )
            except ImportError:
                raise ImportError("boto3 is required for AWS Secrets Manager. Install with: pip install boto3")
        
        try:
            response = self._aws_client.get_secret_value(SecretId=key)
            secret_string = response.get("SecretString")
            
            # Try to parse as JSON and extract the key
            try:
                secret_dict = json.loads(secret_string)
                # If it's a dict, try to get the key by name
                if isinstance(secret_dict, dict):
                    # Try exact key match first
                    if key in secret_dict:
                        return secret_dict[key]
                    # Try last part of key (e.g., "app/SECRET_KEY" -> "SECRET_KEY")
                    short_key = key.split("/")[-1]
                    if short_key in secret_dict:
                        return secret_dict[short_key]
                    # Return the whole JSON as string
                    return secret_string
            except json.JSONDecodeError:
                pass
            
            return secret_string
            
        except self._aws_client.exceptions.ResourceNotFoundException:
            return None
        except Exception as e:
            print(f"AWS Secrets Manager error: {e}")
            return None
    
    def _get_from_gcp(self, key: str) -> Optional[str]:
        """Get secret from Google Cloud Secret Manager."""
        if self._gcp_client is None:
            try:
                from google.cloud import secretmanager
                self._gcp_client = secretmanager.SecretManagerServiceClient()
            except ImportError:
                raise ImportError(
                    "google-cloud-secret-manager is required for GCP. "
                    "Install with: pip install google-cloud-secret-manager"
                )
        
        if not self.gcp_project_id:
            raise ValueError("GCP_PROJECT_ID must be set for GCP secrets backend")
        
        try:
            # Build the resource name
            name = f"projects/{self.gcp_project_id}/secrets/{key}/versions/latest"
            response = self._gcp_client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            print(f"GCP Secret Manager error: {e}")
            return None
    
    def _get_from_vault(self, key: str) -> Optional[str]:
        """Get secret from HashiCorp Vault."""
        if self._vault_client is None:
            try:
                import hvac
                self._vault_client = hvac.Client(
                    url=self.vault_addr,
                    token=self.vault_token,
                )
            except ImportError:
                raise ImportError("hvac is required for Vault. Install with: pip install hvac")
        
        try:
            # Assume KV v2 secrets engine
            # Key format: "secret/data/path/to/secret"
            env = os.getenv("ENVIRONMENT", "development")
            path = f"ml-workflow/{env}/{key}"
            
            response = self._vault_client.secrets.kv.v2.read_secret_version(path=path)
            data = response.get("data", {}).get("data", {})
            
            # Return the value for the key, or the first value
            if key in data:
                return data[key]
            elif data:
                return list(data.values())[0]
            return None
            
        except Exception as e:
            print(f"Vault error: {e}")
            return None
    
    def _get_from_azure(self, key: str) -> Optional[str]:
        """Get secret from Azure Key Vault."""
        if self._azure_client is None:
            try:
                from azure.identity import DefaultAzureCredential
                from azure.keyvault.secrets import SecretClient
                
                credential = DefaultAzureCredential()
                self._azure_client = SecretClient(
                    vault_url=self.azure_vault_url,
                    credential=credential
                )
            except ImportError:
                raise ImportError(
                    "azure-keyvault-secrets and azure-identity are required for Azure. "
                    "Install with: pip install azure-keyvault-secrets azure-identity"
                )
        
        try:
            # Azure Key Vault doesn't allow underscores, so convert
            azure_key = key.replace("_", "-")
            secret = self._azure_client.get_secret(azure_key)
            return secret.value
        except Exception as e:
            print(f"Azure Key Vault error: {e}")
            return None


# ============================================================================
# Global Instance
# ============================================================================

def _create_secrets_manager() -> SecretsManager:
    """Create the global secrets manager instance."""
    backend_name = os.getenv("SECRETS_BACKEND", "env").lower()
    
    try:
        backend = SecretsBackend(backend_name)
    except ValueError:
        print(f"Unknown secrets backend '{backend_name}', using 'env'")
        backend = SecretsBackend.ENV
    
    return SecretsManager(
        backend=backend,
        cache_enabled=True,
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        gcp_project_id=os.getenv("GCP_PROJECT_ID"),
    )


# Global secrets manager instance
secrets = _create_secrets_manager()


# ============================================================================
# Convenience Functions
# ============================================================================

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a secret value (convenience function)."""
    return secrets.get(key, default)


def get_secret_json(key: str, default: Optional[Dict] = None) -> Optional[Dict]:
    """Get a secret and parse as JSON (convenience function)."""
    return secrets.get_json(key, default)


@lru_cache(maxsize=100)
def get_cached_secret(key: str) -> Optional[str]:
    """Get a secret with function-level caching."""
    return secrets.get(key)
