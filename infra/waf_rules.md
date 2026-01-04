# Web Application Firewall (WAF) & Abuse Protection

This document describes the WAF configuration and abuse protection measures for the AI-ML Workflow Automation platform.

## Overview

The platform implements multiple layers of protection:

1. **Rate Limiting** (Application layer) - Redis-based per-API-key limits
2. **WAF Rules** (Edge layer) - Cloudflare/AWS WAF rules
3. **Input Validation** (Application layer) - Pydantic schema validation
4. **Authentication** (Application layer) - JWT/API key authentication

## Rate Limiting Configuration

### Default Limits

| Endpoint             | Limit | Window | Notes               |
| -------------------- | ----- | ------ | ------------------- |
| `/api/predict`       | 200   | 60s    | Per API key         |
| `/api/predict/batch` | 50    | 60s    | Per API key         |
| `/api/upload`        | 10    | 60s    | Per user            |
| `/api/train`         | 5     | 60s    | Resource intensive  |
| `/api/deploy`        | 5     | 60s    | Resource intensive  |
| Default              | 100   | 60s    | All other endpoints |

### Response Headers

All responses include rate limit headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
```

### Rate Limit Exceeded Response

```json
{
  "error": "TooManyRequests",
  "message": "Rate limit exceeded. Maximum 100 requests per 60 seconds.",
  "retry_after": 45
}
```

HTTP Status: `429 Too Many Requests`
Header: `Retry-After: 45`

## Cloudflare WAF Configuration

### Recommended Rules

#### 1. Rate Limiting Rules

```hcl
# Terraform configuration for Cloudflare rate limiting

resource "cloudflare_rate_limit" "api_rate_limit" {
  zone_id   = var.cloudflare_zone_id
  threshold = 1000
  period    = 60

  match {
    request {
      url_pattern = "${var.domain}/api/*"
      schemes     = ["HTTP", "HTTPS"]
      methods     = ["GET", "POST", "PUT", "DELETE"]
    }
  }

  action {
    mode    = "simulate"  # Change to "ban" for production
    timeout = 60
    response {
      content_type = "application/json"
      body         = "{\"error\": \"Rate limit exceeded\"}"
    }
  }
}

resource "cloudflare_rate_limit" "prediction_rate_limit" {
  zone_id   = var.cloudflare_zone_id
  threshold = 500
  period    = 60

  match {
    request {
      url_pattern = "${var.domain}/api/predict*"
    }
  }

  action {
    mode    = "ban"
    timeout = 300
  }
}
```

#### 2. Firewall Rules

```hcl
# Block known bad user agents
resource "cloudflare_filter" "bad_user_agents" {
  zone_id     = var.cloudflare_zone_id
  description = "Block known malicious user agents"
  expression  = "(http.user_agent contains \"sqlmap\") or (http.user_agent contains \"nikto\") or (http.user_agent contains \"nmap\")"
}

resource "cloudflare_firewall_rule" "block_bad_ua" {
  zone_id     = var.cloudflare_zone_id
  description = "Block bad user agents"
  filter_id   = cloudflare_filter.bad_user_agents.id
  action      = "block"
}

# Block requests without API key to protected endpoints
resource "cloudflare_filter" "no_api_key" {
  zone_id     = var.cloudflare_zone_id
  description = "Require API key for predictions"
  expression  = "(http.request.uri.path contains \"/api/predict\") and (not http.request.headers[\"x-api-key\"] ne \"\")"
}

resource "cloudflare_firewall_rule" "require_api_key" {
  zone_id     = var.cloudflare_zone_id
  description = "Block predictions without API key"
  filter_id   = cloudflare_filter.no_api_key.id
  action      = "block"
}

# Challenge suspicious traffic
resource "cloudflare_filter" "suspicious_traffic" {
  zone_id     = var.cloudflare_zone_id
  description = "Challenge high-risk traffic"
  expression  = "(cf.threat_score gt 30) or (cf.bot_management.score lt 30)"
}

resource "cloudflare_firewall_rule" "challenge_suspicious" {
  zone_id     = var.cloudflare_zone_id
  description = "Challenge suspicious requests"
  filter_id   = cloudflare_filter.suspicious_traffic.id
  action      = "challenge"
}
```

#### 3. Managed Rulesets

Enable these Cloudflare managed rulesets:

- **OWASP Core Ruleset** - General web application protection
- **Cloudflare Managed Ruleset** - Cloudflare-curated rules
- **Cloudflare OWASP Core Ruleset** - OWASP top 10 protection

```hcl
resource "cloudflare_ruleset" "waf_managed_rules" {
  zone_id     = var.cloudflare_zone_id
  name        = "WAF Managed Rules"
  description = "Enable managed WAF rulesets"
  kind        = "zone"
  phase       = "http_request_firewall_managed"

  # OWASP Core Ruleset
  rules {
    action = "execute"
    action_parameters {
      id = "efb7b8c949ac4650a09736fc376e9aee"  # OWASP Core Ruleset ID
    }
    expression  = "true"
    description = "OWASP Core Ruleset"
    enabled     = true
  }

  # Cloudflare Managed Ruleset
  rules {
    action = "execute"
    action_parameters {
      id = "4814384a9e5d4991b9815dcfc25d2f1f"  # Cloudflare Managed Ruleset ID
    }
    expression  = "true"
    description = "Cloudflare Managed Ruleset"
    enabled     = true
  }
}
```

## AWS WAF Configuration

### Web ACL Rules

```yaml
# CloudFormation template for AWS WAF

AWSTemplateFormatVersion: "2010-09-09"
Description: WAF configuration for ML Workflow API

Resources:
  MLWorkflowWebACL:
    Type: AWS::WAFv2::WebACL
    Properties:
      Name: ml-workflow-waf
      Scope: REGIONAL # Use CLOUDFRONT for CloudFront distributions
      DefaultAction:
        Allow: {}
      VisibilityConfig:
        SampledRequestsEnabled: true
        CloudWatchMetricsEnabled: true
        MetricName: MLWorkflowWAF
      Rules:
        # AWS Managed Rules - Core Rule Set
        - Name: AWSManagedRulesCommonRuleSet
          Priority: 1
          OverrideAction:
            None: {}
          Statement:
            ManagedRuleGroupStatement:
              VendorName: AWS
              Name: AWSManagedRulesCommonRuleSet
          VisibilityConfig:
            SampledRequestsEnabled: true
            CloudWatchMetricsEnabled: true
            MetricName: AWSManagedRulesCommonRuleSet

        # AWS Managed Rules - Known Bad Inputs
        - Name: AWSManagedRulesKnownBadInputsRuleSet
          Priority: 2
          OverrideAction:
            None: {}
          Statement:
            ManagedRuleGroupStatement:
              VendorName: AWS
              Name: AWSManagedRulesKnownBadInputsRuleSet
          VisibilityConfig:
            SampledRequestsEnabled: true
            CloudWatchMetricsEnabled: true
            MetricName: AWSManagedRulesKnownBadInputsRuleSet

        # AWS Managed Rules - SQL Injection
        - Name: AWSManagedRulesSQLiRuleSet
          Priority: 3
          OverrideAction:
            None: {}
          Statement:
            ManagedRuleGroupStatement:
              VendorName: AWS
              Name: AWSManagedRulesSQLiRuleSet
          VisibilityConfig:
            SampledRequestsEnabled: true
            CloudWatchMetricsEnabled: true
            MetricName: AWSManagedRulesSQLiRuleSet

        # Rate limiting rule
        - Name: RateLimitRule
          Priority: 4
          Action:
            Block:
              CustomResponse:
                ResponseCode: 429
                CustomResponseBodyKey: rate-limit-body
          Statement:
            RateBasedStatement:
              Limit: 2000
              AggregateKeyType: IP
          VisibilityConfig:
            SampledRequestsEnabled: true
            CloudWatchMetricsEnabled: true
            MetricName: RateLimitRule

        # Block requests without API key to /api/predict
        - Name: RequireAPIKey
          Priority: 5
          Action:
            Block:
              CustomResponse:
                ResponseCode: 401
          Statement:
            AndStatement:
              Statements:
                - ByteMatchStatement:
                    SearchString: /api/predict
                    FieldToMatch:
                      UriPath: {}
                    PositionalConstraint: STARTS_WITH
                    TextTransformations:
                      - Priority: 0
                        Type: LOWERCASE
                - NotStatement:
                    Statement:
                      SizeConstraintStatement:
                        FieldToMatch:
                          SingleHeader:
                            Name: x-api-key
                        ComparisonOperator: GT
                        Size: 0
                        TextTransformations:
                          - Priority: 0
                            Type: NONE
          VisibilityConfig:
            SampledRequestsEnabled: true
            CloudWatchMetricsEnabled: true
            MetricName: RequireAPIKey

      CustomResponseBodies:
        rate-limit-body:
          ContentType: APPLICATION_JSON
          Content: '{"error": "Rate limit exceeded", "retry_after": 60}'

  # Associate WAF with ALB
  WebACLAssociation:
    Type: AWS::WAFv2::WebACLAssociation
    Properties:
      ResourceArn: !Ref ApplicationLoadBalancerArn
      WebACLArn: !GetAtt MLWorkflowWebACL.Arn
```

## Application-Level Protection

### Input Validation

All API inputs are validated using Pydantic schemas:

```python
from pydantic import BaseModel, Field, validator

class PredictRequest(BaseModel):
    model_id: str = Field(..., max_length=50, pattern=r'^mdl-[a-z0-9]+$')
    data: dict = Field(..., max_items=1000)

    @validator('data')
    def validate_data_size(cls, v):
        # Limit total data size
        import json
        if len(json.dumps(v)) > 1_000_000:  # 1MB limit
            raise ValueError('Request data too large')
        return v
```

### Request Size Limits

Configure in nginx or API gateway:

```nginx
# nginx configuration
client_max_body_size 10M;
client_body_buffer_size 128k;

# Limit request rate per IP
limit_req_zone $binary_remote_addr zone=api:10m rate=100r/s;

location /api/ {
    limit_req zone=api burst=200 nodelay;
    limit_req_status 429;
}
```

### SQL Injection Prevention

All database queries use parameterized queries via SQLAlchemy ORM:

```python
# Safe - parameterized query
result = db.query(Model).filter(Model.id == model_id).first()

# Dangerous - never do this
# result = db.execute(f"SELECT * FROM models WHERE id = '{model_id}'")
```

### XSS Prevention

All responses use proper content types and CSP headers:

```python
from fastapi import Response

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response
```

## Monitoring & Alerting

### Metrics to Monitor

1. **Rate limit hits** - Track when limits are reached
2. **Authentication failures** - Detect brute force attempts
3. **Error rates** - Detect attacks causing errors
4. **Request sizes** - Detect oversized payloads
5. **Geographic distribution** - Detect unusual traffic sources

### Alert Thresholds

| Metric              | Warning | Critical |
| ------------------- | ------- | -------- |
| Rate limit hits/min | 100     | 500      |
| Auth failures/min   | 50      | 200      |
| 5xx errors/min      | 10      | 50       |
| Avg response time   | 500ms   | 2000ms   |

### Logging

All security events are logged:

```python
import structlog

logger = structlog.get_logger()

@app.middleware("http")
async def log_security_events(request, call_next):
    response = await call_next(request)

    if response.status_code == 401:
        logger.warning("authentication_failed",
            path=request.url.path,
            ip=request.client.host,
            user_agent=request.headers.get("user-agent"))

    if response.status_code == 429:
        logger.warning("rate_limit_exceeded",
            path=request.url.path,
            ip=request.client.host)

    return response
```

## Incident Response

### Blocking Malicious IPs

```bash
# Cloudflare - add IP to block list
curl -X POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/firewall/access_rules/rules" \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{
    "mode": "block",
    "configuration": {
      "target": "ip",
      "value": "1.2.3.4"
    },
    "notes": "Blocked due to abuse"
  }'
```

### Emergency Rate Limit Override

In case of DDoS, temporarily reduce limits:

```python
# Emergency configuration
rate_limit_config.default_limit = 10
rate_limit_config.default_window = 60
```

## Compliance Checklist

- [ ] Rate limiting enabled on all API endpoints
- [ ] WAF rules deployed and tested
- [ ] Input validation on all endpoints
- [ ] Security headers configured
- [ ] Logging enabled for security events
- [ ] Alerting configured for security metrics
- [ ] Incident response procedures documented
- [ ] Regular security audits scheduled
