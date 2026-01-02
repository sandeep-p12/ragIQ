# Azure OpenAI Setup

## Overview

RAG IQ supports Azure OpenAI for both LLM operations and embeddings, providing enterprise-grade security and compliance features.

## Authentication Methods

### Azure AD Authentication (Recommended)

Azure AD authentication uses `DefaultAzureCredential` for secure, keyless authentication. This is the default and recommended method.

**Prerequisites**:
- Azure CLI installed and logged in, OR
- Managed Identity configured (for Azure-hosted deployments), OR
- Environment variables set for service principal authentication

**Configuration**:
```bash
AZURE_OPENAI_USE_AZURE_AD=true  # Default, can be omitted
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

**Benefits**:
- No API keys to manage
- Automatic credential rotation
- Enterprise security compliance
- Works with Managed Identities

### API Key Authentication (Fallback)

For environments where Azure AD is not available, API key authentication can be used as a fallback.

**Configuration**:
```bash
AZURE_OPENAI_USE_AZURE_AD=false
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

## Deployment Names

Azure OpenAI uses deployment names instead of model names. Configure separate deployments for:

- **LLM**: Set `AZURE_OPENAI_DEPLOYMENT_NAME` or `PARSEFORGE_LLM_AZURE_DEPLOYMENT_NAME`
- **Embeddings**: Set `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`

If deployment names are not specified, the system will use the model name as the deployment name.

## Provider Selection

You can mix and match providers:
- Use OpenAI for LLM and Azure OpenAI for embeddings
- Use Azure OpenAI for LLM and OpenAI for embeddings
- Use the same provider for both

Set providers via:
- `PARSEFORGE_LLM_PROVIDER`: "openai", "azure_openai", or "none"
- `EMBEDDING_PROVIDER`: "openai" or "azure_openai"

## Configuration Examples

### LLM with Azure OpenAI

```python
from src.config.parsing import ParseForgeConfig

config = ParseForgeConfig(
    llm_provider="azure_openai",
    llm_model="gpt-4o",
    llm_azure_endpoint="https://your-resource.openai.azure.com/",
    llm_azure_deployment_name="gpt-4o-deployment",
    llm_use_azure_ad=True
)
```

### Embeddings with Azure OpenAI

```python
from src.config.retrieval import EmbeddingConfig

config = EmbeddingConfig.from_env(
    provider="azure_openai",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment_name="text-embedding-3-small-deployment",
    use_azure_ad=True
)
```

## Environment Variables

### Required for Azure OpenAI

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

### Optional (if using API key)

```bash
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_USE_AZURE_AD=false
```

### Deployment Names

```bash
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small-deployment
```

### Provider Selection

```bash
PARSEFORGE_LLM_PROVIDER=azure_openai
EMBEDDING_PROVIDER=azure_openai
```

## Complete Example

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small-deployment
AZURE_OPENAI_USE_AZURE_AD=true

# ParseForge LLM Configuration
PARSEFORGE_LLM_PROVIDER=azure_openai
PARSEFORGE_LLM_MODEL=gpt-4o
PARSEFORGE_LLM_AZURE_ENDPOINT=https://your-resource.openai.azure.com/
PARSEFORGE_LLM_AZURE_DEPLOYMENT_NAME=gpt-4o-deployment
PARSEFORGE_LLM_USE_AZURE_AD=true

# Embedding Configuration
EMBEDDING_PROVIDER=azure_openai
```

## Implementation Details

### LLM Provider

**Location**: `src/providers/llm/openai_llm.py`

**Azure OpenAI Support**:
- Uses `AzureOpenAI` client
- Azure AD authentication via `DefaultAzureCredential`
- API key fallback
- Deployment name support

### Embedding Provider

**Location**: `src/providers/embedding/openai_embedding.py`

**Azure OpenAI Support**:
- Uses `AzureOpenAI` client
- Azure AD authentication via `DefaultAzureCredential`
- API key fallback
- Deployment name support

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   - Ensure Azure CLI is logged in: `az login`
   - Check Managed Identity configuration
   - Verify service principal credentials

2. **Deployment Not Found**
   - Verify deployment name matches Azure portal
   - Check deployment is active
   - Ensure correct endpoint URL

3. **API Version Mismatch**
   - Use `2025-01-01-preview` or later
   - Check Azure OpenAI service version

## Next Steps

- **[Configuration](./08-configuration.md)** - Complete configuration guide
- **[Providers](./10-providers.md)** - Provider implementation details
- **[Troubleshooting](./15-troubleshooting.md)** - Common issues and solutions

