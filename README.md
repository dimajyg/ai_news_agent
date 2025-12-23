# AI News Agent

A sophisticated multi-tenant AI-powered Telegram bot that collects, analyzes, and posts AI/ML research news from multiple sources. The system uses LangGraph for workflow orchestration, ChromaDB for deduplication, and advanced LLM-based processing to generate high-quality research summaries.

## Features

- **Multi-Tenant Architecture**: Support for multiple Telegram groups with individual configurations
- **Multi-Source News Extraction**: arXiv papers, AI websites, Telegram channels
- **Intelligent XML Processing**: Deep research analysis with web tools and validation loops
- **Smart Clustering**: Groups similar articles about the same research topic
- **Vector-Based Deduplication**: Removes previously processed content using ChromaDB
- **Tournament Ranking**: Selects top N most relevant articles
- **Multi-Language Support**: EN ↔ RU translation with configurable target language
- **Keywords Filtering**: Filter content by custom keywords per group
- **Configurable Post Size**: Adjust number of articles per post (default: 10)
- **Automated Telegram Posting**: Daily posts with proper HTML formatting and link preservation
- **Smart Message Splitting**: Respects article boundaries when splitting long messages

## Architecture

### LangGraph Agent Workflow

```mermaid
flowchart TB
    Start([Start]) --> Extract["NewsExtractorNode<br/>Extract from arXiv, websites, Telegram"]
    Extract --> Verify["SourceVerificationNode<br/>Filter by keywords and research relevance"]
    Verify --> XMLUnify["XMLUnificationNode<br/>Convert to XML using XMLResearchAgent"]
    XMLUnify --> Cluster["ClusteringNode<br/>Group similar articles"]
    Cluster --> Dedup["DeduplicationNode<br/>Remove duplicates via ChromaDB"]
    Dedup --> Rank["RankingNode<br/>Score and rank clusters"]
    Rank --> BuildClusters["Build XML Clusters<br/>Organize by cluster ID"]
    BuildClusters --> Merge["XMLMergerNode<br/>Merge XMLs within clusters"]
    Merge --> Tournament["TournamentRankerNode<br/>Select top N articles"]
    Tournament --> PostProcessor["PostProcessorNode<br/>Write<br/>Translate<br/>Format<br/>Validate<br/>Fix"]
    PostProcessor --> Poster["TelegramPosterNode<br/>Post to Telegram channels"]
    Poster --> VectorUpdate["VectorUpdateNode<br/>Save to ChromaDB for deduplication"]
    VectorUpdate --> End([End])

    style Extract fill:#e1f5ff
    style Verify fill:#e1f5ff
    style XMLUnify fill:#fff4e1
    style Cluster fill:#ffe1f5
    style Dedup fill:#ffe1f5
    style Rank fill:#ffe1f5
    style BuildClusters fill:#ffe1f5
    style Merge fill:#e1ffe1
    style Tournament fill:#e1ffe1
    style PostProcessor fill:#f5e1ff
    style Poster fill:#ffe1e1
    style VectorUpdate fill:#ffe1e1

```

### Core Components

#### 1. **NewsExtractorNode** (Extraction Phase)
- Fetches articles from arXiv, websites, and Telegram channels
- Handles multiple source types with unified interface
- Returns raw articles with metadata

#### 2. **SourceVerificationNode** (Filtering Phase)
- Filters articles by custom keywords (if configured)
- Verifies research relevance using keyword matching
- Rejects non-research content

#### 3. **XMLUnificationNode** (Deep Analysis Phase)
- Uses **XMLResearchAgentNode** for each article
- Agentic workflow with tools:
  - `fetch_web_page_content`: Get full article text
  - `web_search`: Find related research
- Generates comprehensive XML with:
  - Title, MainIdea (200-300 words)
  - Uniqueness (100-150 words)
  - FurtherResearch (5 ideas)
  - RelevantWorks, UsefulLinks
- Validation loops ensure XML quality

#### 4. **ClusteringNode** (Grouping Phase)
- Groups similar articles using embeddings
- Configurable max clusters (default: 10)
- Assigns cluster IDs to articles

#### 5. **DeduplicationNode** (Deduplication Phase)
- Checks ChromaDB for similar content
- Removes duplicates based on similarity threshold
- Prevents reposting same research

#### 6. **RankingNode** (Scoring Phase)
- Scores clusters by relevance and novelty
- Selects top N clusters for further processing
- Configurable post_size parameter

#### 7. **XMLMergerNode** (Consolidation Phase)
- Merges multiple XMLs within same cluster
- Creates comprehensive cluster summaries
- Preserves all important information

#### 8. **TournamentRankerNode** (Final Selection Phase)
- Tournament-style ranking of merged documents
- Selects top N articles (configurable via /set_post_size)
- Uses LLM for pairwise comparisons

#### 9. **PostProcessorNode** (Content Generation Phase)
- **Per-article pipeline**:
  1. **Write**: Generate Telegram-formatted text
  2. **Translate**: EN ↔ RU based on group settings
  3. **Format**: Add emoji, bold title, "Read more" link
  4. **Validate**: Check HTML tags and structure
  5. **Fix**: Auto-correct formatting issues
- Processes each article separately through complete pipeline
- Ensures high-quality, properly formatted output

#### 10. **TelegramPosterNode** (Publishing Phase)
- Posts to configured Telegram channels
- Smart message splitting (respects article boundaries)
- HTML formatting with link preservation
- Error handling and retry logic

#### 11. **VectorUpdateNode** (Storage Phase)
- Saves processed XMLs to ChromaDB
- Stores MainIdea for similarity search
- Enables future deduplication

## Quick Start

### Prerequisites
- Python 3.9+
- OpenRouter API key (for Google Gemini)
- Telegram Bot Token
- PostgreSQL database
- Fly.io account (for deployment)

### Local Development

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/ai-news-agent.git
cd ai-news-agent
```

2. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your credentials:
# - OPENROUTER_API_KEY
# - TELEGRAM_BOT_TOKEN
# - DATABASE_URL
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run database migrations**:
```bash
# Apply migrations in migrations/ folder
psql $DATABASE_URL < migrations/001_add_language_keywords.sql
```

5. **Run the bot**:
```bash
python src/main.py
```

## Configuration

### Bot Commands

Users can configure their groups using these Telegram commands:

- `/start` - Initialize bot for the group
- `/config` - Show current group configuration
- `/set_language <en|ru>` - Set post language (English or Russian)
- `/add_keywords <keywords>` - Add filtering keywords (comma-separated)
- `/set_post_size <N>` - Set number of articles per post (default: 10)

### Environment Variables

```bash
# LLM Configuration
OPENROUTER_API_KEY=your_openrouter_key
LLM_MODEL=google/gemini-2.0-flash-exp:free

# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_bot_token

# Database Configuration
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
CHROMA_COLLECTION_NAME=ai_news_articles

# Logging
LOG_LEVEL=INFO
```

### Configuration File (config_openrouter.yaml)

```yaml
llm:
  provider: openrouter
  model: google/gemini-2.0-flash-exp:free
  api_key: ${OPENROUTER_API_KEY}

clustering:
  max_clusters: 10
  embedding_model: text-embedding-ada-002

ranking:
  post_size: 10  # Default number of articles

processing:
  xml:
    main_idea_target_min: 200
    main_idea_target_max: 300
    uniqueness_target_min: 100
    uniqueness_target_max: 150
```

## Project Structure

```
ai-news-agent/
├── src/
│   ├── agents/
│   │   ├── graphs/
│   │   │   └── research_workflow.py      # Main LangGraph workflow
│   │   ├── nodes/
│   │   │   ├── news_extractor.py         # Extract from sources
│   │   │   ├── source_verification.py    # Filter by keywords
│   │   │   ├── xml_unification.py        # Convert to XML
│   │   │   ├── xml_research_agent.py     # Deep research with tools
│   │   │   ├── clustering.py             # Group similar articles
│   │   │   ├── deduplication.py          # Remove duplicates
│   │   │   ├── ranking.py                # Score clusters
│   │   │   ├── xml_merger.py             # Merge cluster XMLs
│   │   │   ├── tournament_ranker.py      # Select top N
│   │   │   ├── post_processor.py         # Generate formatted posts
│   │   │   ├── telegram_poster.py        # Post to Telegram
│   │   │   └── vector_update.py          # Save to ChromaDB
│   │   └── tools/
│   │       └── research_tools.py         # Web fetch & search tools
│   ├── bot/
│   │   └── telegram_bot.py               # Multi-tenant bot logic
│   ├── core/
│   │   ├── llm_provider.py               # LLM abstraction layer
│   │   └── scheduler.py                  # Daily scheduling
│   ├── database/
│   │   ├── models.py                     # SQLAlchemy models
│   │   └── manager.py                    # Database operations
│   ├── extractors/
│   │   ├── arxiv_extractor.py            # arXiv API client
│   │   ├── website_extractor.py          # RSS feed parser
│   │   └── telegram_extractor.py         # Telegram channel scraper
│   ├── processors/
│   │   ├── arxiv_processor.py            # Process arXiv papers
│   │   ├── telegram_processor.py         # Process Telegram posts
│   │   ├── website_processor.py          # Process web articles
│   │   ├── base_processor.py             # Base processor class
│   │   └── xml_schema.py                 # XML validation
│   ├── vector_store/
│   │   └── chroma_manager.py             # ChromaDB interface
│   ├── utils/
│   │   ├── config.py                     # Configuration loader
│   │   ├── logging.py                    # Structured logging
│   │   ├── text.py                       # Text utilities
│   │   └── web_search.py                 # Web search utility
│   └── main.py                           # Application entry point
├── migrations/
│   └── 001_add_language_keywords.sql     # Database migrations
├── config_openrouter.yaml                # Main configuration
├── fly.toml                              # Fly.io deployment config
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## Key Features

### Multi-Language Support
- **English ↔ Russian translation**
- Configurable per group via `/set_language`
- Translation happens in PostProcessorNode
- Preserves HTML formatting and links

### Keywords Filtering
- Filter articles by custom keywords
- Configured via `/add_keywords`
- Applied in SourceVerificationNode
- Supports comma-separated lists

### Configurable Post Size
- Adjust number of articles per post
- Set via `/set_post_size <N>`
- Default: 10 articles
- Affects RankingNode and TournamentRankerNode

### Smart Message Splitting
- Respects article block boundaries
- Never splits HTML tags mid-tag
- Each article stays complete in one message part
- Handles posts up to 25 articles

### Link Preservation
- "Read more" links always included
- Automatic URL addition if UsefulLinks is empty
- HTML `<a href>` tags preserved correctly
- Validation ensures links are present

## Deployment

### Fly.io Deployment

1. **Install Fly CLI**:
```bash
curl -L https://fly.io/install.sh | sh
```

2. **Login to Fly.io**:
```bash
fly auth login
```

3. **Deploy**:
```bash
fly deploy
```

4. **Set secrets**:
```bash
fly secrets set OPENROUTER_API_KEY=your_key
fly secrets set TELEGRAM_BOT_TOKEN=your_token
fly secrets set DATABASE_URL=your_db_url
```

5. **Monitor logs**:
```bash
fly logs
```

### Production Considerations
- Use PostgreSQL for production database
- Set up proper log rotation
- Configure backup strategy for ChromaDB
- Monitor API rate limits (OpenRouter, Telegram)
- Set up health check endpoints
- Use environment-specific configurations

## Development

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_processors.py

# Run with coverage
pytest --cov=src tests/
```

### Code Style
- Follow PEP 8 for Python code
- Use type hints where applicable
- Add docstrings to functions and classes
- Keep functions under 50 lines when possible
- Use meaningful variable names

### Adding New Sources
1. Create extractor in `src/extractors/`
2. Create processor in `src/processors/`
3. Add to NewsExtractorNode
4. Update configuration

## Monitoring and Logging

### Structured Logging
```python
from src.utils.logging import AgentLogger

logger = AgentLogger(__name__)
logger.info("Processing started", extra={"article_count": 10})
logger.error("Failed to process", extra={"error": str(e)})
```

### Metrics Tracked
- Articles extracted per source
- Clustering accuracy
- Deduplication rate
- Post generation success rate
- Telegram posting success rate
- Vector store operations
- Execution time per node

## Troubleshooting

### Common Issues

**Issue**: `'ChromaManager' object has no attribute 'add_document'`
- **Solution**: Update to latest version with `add_document()` method

**Issue**: `HTTP 400: can't parse entities: Unexpected end tag`
- **Solution**: Update to latest version with smart message splitting

**Issue**: Links missing in Telegram posts
- **Solution**: Ensure UsefulLinks validation is enabled in XMLResearchAgentNode

**Issue**: Translation not working
- **Solution**: Check `/set_language` command and verify LLM provider

## Recent Updates

### Version 27 (Latest)
- ✅ Fixed ChromaDB `add_document()` method
- ✅ Smart message splitting respecting article boundaries
- ✅ Refactored processors (57% code reduction)
- ✅ Removed redundant LLM calls in processors
- ✅ Improved architecture with single responsibility principle

### Version 26
- ✅ Added multi-language support (EN ↔ RU)
- ✅ Added keywords filtering
- ✅ Added configurable post size
- ✅ Implemented PostProcessorNode with validation loops
- ✅ Fixed link preservation in Telegram posts

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **LangGraph** - Agent orchestration and workflow management
- **ChromaDB** - Vector storage and semantic search
- **OpenRouter** - LLM API access (Google Gemini)
- **Telegram Bot API** - Messaging and channel posting
- **arXiv** - Open access to research papers
- **Fly.io** - Deployment platform
