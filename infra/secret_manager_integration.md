# Secrets Management Integration Guide

This document describes how to integrate secrets management into the AI-ML Workflow Automation platform, replacing hardcoded secrets and `.env` files with secure cloud-based secrets managers.

## Overview

The platform supports multiple secrets management solutions:

- **AWS Secrets Manager** (Recommended for AWS deployments)
- **Google Cloud Secret Manager** (Recommended for GCP deployments)
- **HashiCorp Vault** (Recommended for multi-cloud/on-premise)
- **Azure Key Vault** (Recommended for Azure deployments)

## Current Secrets to Migrate

The following secrets should be moved from `.env` files to a secrets manager:

| Secret                  | Description                  | Category    |
| ----------------------- | ---------------------------- | ----------- |
| `SECRET_KEY`            | JWT signing key              | Application |
| `DATABASE_URL`          | PostgreSQL connection string | Database    |
| `REDIS_URL`             | Redis connection string      | Cache       |
| `OPENAI_API_KEY`        | OpenAI API key               | AI/LLM      |
| `ANTHROPIC_API_KEY`     | Anthropic API key            | AI/LLM      |
| `AWS_ACCESS_KEY_ID`     | AWS access key               | Cloud       |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key               | Cloud       |
| `GCP_SERVICE_ACCOUNT`   | GCP service account JSON     | Cloud       |
| `HF_API_TOKEN`          | Hugging Face token           | AI/LLM      |

## Integration Options

### Option 1: AWS Secrets Manager

#### Setup

1. **Create secrets in AWS Secrets Manager**:

```bash
# Create application secrets
aws secretsmanager create-secret \
    --name ml-workflow/production/app \
    --secret-string '{
        "SECRET_KEY": "your-production-secret-key",
        "DATABASE_URL": "postgresql://user:pass@host:5432/db",
        "REDIS_URL": "redis://host:6379/0"
    }'

# Create API keys secret
aws secretsmanager create-secret \
    --name ml-workflow/production/api-keys \
    --secret-string '{
        "OPENAI_API_KEY": "sk-...",
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "HF_API_TOKEN": "hf_..."
    }'
```

2. **Update `config.py`** to load secrets:

```python
# backend/config.py
import json
import boto3
from botocore.exceptions import ClientError

def get_aws_secrets(secret_name: str, region: str = "us-east-1") -> dict:
    """Load secrets from AWS Secrets Manager."""
    client = boto3.client("secretsmanager", region_name=region)

    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])
    except ClientError as e:
        print(f"Warning: Could not load secret {secret_name}: {e}")
        return {}

# Load secrets based on environment
import os
ENV = os.getenv("ENVIRONMENT", "development")

if ENV == "production":
    app_secrets = get_aws_secrets(f"ml-workflow/{ENV}/app")
    api_secrets = get_aws_secrets(f"ml-workflow/{ENV}/api-keys")
else:
    app_secrets = {}
    api_secrets = {}
```

3. **IAM Policy** for the application:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue"],
      "Resource": ["arn:aws:secretsmanager:us-east-1:*:secret:ml-workflow/*"]
    }
  ]
}
```

### Option 2: Google Cloud Secret Manager

#### Setup

1. **Enable the API**:

```bash
gcloud services enable secretmanager.googleapis.com
```

2. **Create secrets**:

```bash
# Create secret
echo -n "your-secret-key" | gcloud secrets create ml-workflow-secret-key \
    --data-file=- \
    --replication-policy="automatic"

# Create database URL secret
echo -n "postgresql://..." | gcloud secrets create ml-workflow-db-url \
    --data-file=-
```

3. **Update `config.py`**:

```python
from google.cloud import secretmanager

def get_gcp_secret(project_id: str, secret_id: str, version: str = "latest") -> str:
    """Load secret from GCP Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"

    try:
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"Warning: Could not load secret {secret_id}: {e}")
        return ""

# Usage
if ENV == "production":
    PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    secret_key = get_gcp_secret(PROJECT_ID, "ml-workflow-secret-key")
```

### Option 3: HashiCorp Vault

#### Setup

1. **Install and configure Vault**:

```bash
# Start Vault in dev mode (for testing)
vault server -dev

# Set environment
export VAULT_ADDR='http://127.0.0.1:8200'
```

2. **Store secrets**:

```bash
# Enable KV secrets engine
vault secrets enable -path=ml-workflow kv-v2

# Store application secrets
vault kv put ml-workflow/production/config \
    SECRET_KEY="your-secret-key" \
    DATABASE_URL="postgresql://..." \
    REDIS_URL="redis://..."

# Store API keys
vault kv put ml-workflow/production/api-keys \
    OPENAI_API_KEY="sk-..." \
    ANTHROPIC_API_KEY="sk-ant-..."
```

3. **Update `config.py`**:

```python
import hvac

def get_vault_secrets(path: str) -> dict:
    """Load secrets from HashiCorp Vault."""
    client = hvac.Client(
        url=os.getenv("VAULT_ADDR", "http://localhost:8200"),
        token=os.getenv("VAULT_TOKEN"),
    )

    try:
        response = client.secrets.kv.v2.read_secret_version(path=path)
        return response["data"]["data"]
    except Exception as e:
        print(f"Warning: Could not load secrets from {path}: {e}")
        return {}

# Usage
if ENV == "production":
    app_secrets = get_vault_secrets("ml-workflow/production/config")
```

## Updated Configuration Module

Create a new secrets loader module:

```python
# backend/core/secrets.py
"""
Secrets management abstraction layer.
Supports multiple backends: AWS, GCP, Vault, or environment variables.
"""

import os
import json
from typing import Optional, Dict
from enum import Enum


class SecretsBackend(Enum):
    ENV = "env"
    AWS = "aws"
    GCP = "gcp"
    VAULT = "vault"
    AZURE = "azure"


class SecretsManager:
    """Unified secrets management."""

    def __init__(self, backend: SecretsBackend = SecretsBackend.ENV):
        self.backend = backend
        self._cache: Dict[str, str] = {}

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value."""
        # Check cache first
        if key in self._cache:
            return self._cache[key]

        value = None

        if self.backend == SecretsBackend.ENV:
            value = os.getenv(key, default)
        elif self.backend == SecretsBackend.AWS:
            value = self._get_aws_secret(key)
        elif self.backend == SecretsBackend.GCP:
            value = self._get_gcp_secret(key)
        elif self.backend == SecretsBackend.VAULT:
            value = self._get_vault_secret(key)

        if value:
            self._cache[key] = value

        return value or default

    def _get_aws_secret(self, key: str) -> Optional[str]:
        try:
            import boto3
            client = boto3.client("secretsmanager")
            response = client.get_secret_value(SecretId=key)
            return response["SecretString"]
        except Exception:
            return None

    def _get_gcp_secret(self, key: str) -> Optional[str]:
        try:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            project = os.getenv("GCP_PROJECT_ID")
            name = f"projects/{project}/secrets/{key}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception:
            return None

    def _get_vault_secret(self, key: str) -> Optional[str]:
        try:
            import hvac
            client = hvac.Client(
                url=os.getenv("VAULT_ADDR"),
                token=os.getenv("VAULT_TOKEN"),
            )
            path = f"ml-workflow/{os.getenv('ENVIRONMENT', 'dev')}/{key}"
            response = client.secrets.kv.v2.read_secret_version(path=path)
            return response["data"]["data"].get(key)
        except Exception:
            return None


# Global secrets manager instance
_backend = SecretsBackend(os.getenv("SECRETS_BACKEND", "env"))
secrets = SecretsManager(backend=_backend)
```

## CI/CD Integration

### GitHub Actions Secrets

1. Add secrets to GitHub repository settings:

   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_SECRETS_MANAGER_ARN`

2. Update workflow to use secrets:

```yaml
# .github/workflows/deploy.yml
jobs:
  deploy:
    runs-on: ubuntu-latest
    env:
      AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    steps:
      - name: Get secrets from AWS
        run: |
          SECRET=$(aws secretsmanager get-secret-value \
            --secret-id ml-workflow/production/app \
            --query SecretString --output text)
          echo "SECRET_KEY=$(echo $SECRET | jq -r .SECRET_KEY)" >> $GITHUB_ENV
```

### Kubernetes Secrets

For Kubernetes deployments, sync secrets from the secrets manager:

```yaml
# Using External Secrets Operator
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: ml-workflow-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: ml-workflow-secrets
  data:
    - secretKey: SECRET_KEY
      remoteRef:
        key: ml-workflow/production/app
        property: SECRET_KEY
    - secretKey: DATABASE_URL
      remoteRef:
        key: ml-workflow/production/app
        property: DATABASE_URL
```

## Security Best Practices

1. **Rotate secrets regularly**

   - Set up automatic rotation for database passwords
   - Rotate API keys every 90 days

2. **Use least privilege**

   - Grant only necessary permissions to access secrets
   - Use separate secrets for different environments

3. **Audit access**

   - Enable logging for secrets access
   - Monitor for unauthorized access attempts

4. **Never commit secrets**

   - Add `.env` to `.gitignore`
   - Use pre-commit hooks to scan for secrets

5. **Environment separation**
   - Use different secrets for dev/staging/production
   - Never use production secrets in development

## Migration Checklist

- [ ] Choose secrets management solution
- [ ] Create secrets in the chosen solution
- [ ] Update `config.py` to load secrets from manager
- [ ] Update CI/CD pipelines to use secrets
- [ ] Remove secrets from `.env` files
- [ ] Add `.env` to `.gitignore`
- [ ] Test in staging environment
- [ ] Deploy to production
- [ ] Rotate all secrets after migration
- [ ] Document access policies
