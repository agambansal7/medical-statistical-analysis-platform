# Medical Statistical Analysis Platform

A comprehensive platform for automated statistical analysis of medical data with LLM-powered recommendations and interpretations.

## Features

- **Data Upload**: Support for CSV and Excel files
- **Automatic Data Profiling**: Intelligent variable type detection and quality assessment
- **50+ Statistical Methods**: Comprehensive validated statistical tests
- **LLM-Powered Analysis**: Claude AI recommends appropriate analyses based on research questions
- **Detailed Interpretations**: AI-generated explanations of results in clinical context
- **Publication-Ready Visualizations**: High-quality figures for research papers
- **Conversational Interface**: Natural language interaction for analysis

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    React Frontend (Vite + TS)                    │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────────┐ │
│  │  Upload  │  │  Chat Panel  │  │  Results & Visualizations  │ │
│  └──────────┘  └──────────────┘  └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │ REST API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │ Data Endpoints │  │ Analysis Routes │  │ Chat/LLM Routes  │  │
│  └────────────────┘  └─────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Statistical Engine (Python)                   │
│  scipy │ statsmodels │ pingouin │ lifelines │ scikit-learn      │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Anthropic API Key

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run the server
uvicorn backend.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Statistical Methods Available

### Descriptive Statistics
- Mean, SD, SEM, Median, IQR, Range
- Frequencies and percentages
- Table 1 generation

### Comparative Tests
- Independent t-test, Paired t-test, Welch's t-test
- One-way ANOVA, Welch's ANOVA
- Mann-Whitney U, Wilcoxon signed-rank
- Kruskal-Wallis, Friedman test
- Chi-square, Fisher's exact, McNemar's test

### Correlation Analysis
- Pearson, Spearman, Kendall
- Point-biserial, Partial correlation
- Correlation matrices

### Regression Analysis
- Linear regression (simple/multiple)
- Logistic regression (binary/multinomial/ordinal)
- Poisson regression
- Mixed effects models

### Survival Analysis
- Kaplan-Meier estimation
- Log-rank test
- Cox proportional hazards

### Diagnostic Tests
- Sensitivity, Specificity, PPV, NPV
- ROC curves and AUC
- Calibration analysis

### Agreement & Reliability
- Cohen's/Fleiss' Kappa
- ICC
- Bland-Altman analysis
- Cronbach's alpha

### Power Analysis
- Sample size calculations
- Power analysis for various tests

## Usage Workflow

1. **Upload Data**: Drag and drop your CSV or Excel file
2. **Review Profile**: Check the automatic data profiling and warnings
3. **Ask Research Question**: Describe your research question in natural language
4. **Review Plan**: The AI will suggest appropriate analyses
5. **Execute Analyses**: Run the recommended analyses
6. **View Results**: See detailed results with interpretations
7. **Generate Visualizations**: Create publication-ready figures

## API Endpoints

### Sessions
- `POST /api/v1/session/create` - Create new session
- `GET /api/v1/session/{id}` - Get session info
- `DELETE /api/v1/session/{id}` - Delete session

### Data
- `POST /api/v1/data/upload` - Upload data file
- `GET /api/v1/data/profile/{session_id}` - Get data profile
- `GET /api/v1/data/preview/{session_id}` - Get data preview

### Analysis
- `POST /api/v1/analysis/run` - Run statistical analysis
- `GET /api/v1/analysis/types` - Get available analysis types
- `GET /api/v1/analysis/history/{session_id}` - Get analysis history

### Chat
- `POST /api/v1/chat/message` - Send chat message
- `POST /api/v1/chat/research-question` - Analyze research question
- `GET /api/v1/chat/history/{session_id}` - Get chat history

### Visualization
- `POST /api/v1/viz/create` - Create visualization
- `GET /api/v1/viz/suggestions/{session_id}` - Get suggested visualizations

## Development

### Project Structure

```
Statistical analysis/
├── src/                      # Core statistical modules
│   ├── data/                 # Data ingestion & profiling
│   ├── statistics/           # Statistical analysis modules
│   ├── visualization/        # Plotting modules
│   ├── llm/                  # LLM orchestration
│   └── utils/                # Utilities
├── backend/                  # FastAPI backend
│   ├── api/routes/           # API endpoints
│   ├── services/             # Business logic
│   ├── models/               # Pydantic schemas
│   └── core/                 # Configuration
├── frontend/                 # React frontend
│   ├── src/components/       # UI components
│   ├── src/services/         # API client
│   ├── src/hooks/            # React hooks
│   └── src/types/            # TypeScript types
└── tests/                    # Test suite
```

## License

MIT License
