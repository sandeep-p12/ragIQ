# Setup and Installation

## Prerequisites

- Python >= 3.10
- `uv` package manager (recommended) or `pip`
- OpenAI API key (or Azure OpenAI credentials)
- Pinecone API key (for vector storage)

## Step 1: Install `uv` (if not installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Step 2: Install Dependencies

### Using `uv` (Recommended)

```bash
uv sync
```

### Using `pip`

```bash
pip install -e .
```

## Step 3: Configure Environment

Create a `.env` file in the project root:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=hybrid-chunking
PINECONE_NAMESPACE=children

# ParseForge
PARSEFORGE_DEVICE=cpu
PARSEFORGE_BATCH_SIZE=50
PARSEFORGE_LLM_PROVIDER=openai
PARSEFORGE_LLM_MODEL=gpt-4o
```

For Azure OpenAI setup, see [Azure OpenAI Configuration](./09-azure-openai.md).

## Step 4: Download AI Models (Optional)

Place model files in `src/ai_models/`:

- `doclayout_yolo_ft.pt`: YOLO layout detection model
- `crnn_vgg16_bn.pt`: Doctr recognition model
- `fast_base.pt`: Doctr detection model

If models are missing, the system will disable those features gracefully.

## Step 5: Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

The UI will open at `http://localhost:8501`.

## Verification

### Test Installation

```python
from src.config.parsing import ParseForgeConfig
from src.pipelines.parsing.parseforge import ParseForge

config = ParseForgeConfig()
parser = ParseForge(config)
print("ParseForge initialized successfully!")
```

### Check Dependencies

```bash
python -c "import openai; import pinecone; print('Dependencies OK')"
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the project root and dependencies are installed
2. **API Key Errors**: Check your `.env` file has correct API keys
3. **Model Not Found**: Models are optional; system will work without them (with reduced features)

For more troubleshooting, see [Troubleshooting Guide](./15-troubleshooting.md).

## Next Steps

- **[Configuration](./08-configuration.md)** - Detailed configuration options
- **[Azure OpenAI Setup](./09-azure-openai.md)** - Configure Azure OpenAI
- **[Usage Examples](./14-api-reference.md)** - Code examples

