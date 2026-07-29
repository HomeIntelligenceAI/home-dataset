# home-dataset

> Part of the **HOME AI** project — HFM-1 (HOME Foundation Model)  
> Organization: [HomeIntelligenceAI](https://github.com/HomeIntelligenceAI)

## Purpose

Data collection, cleaning, and pipeline management for training HFM-1.

### Data Sources
- Telugu Wikipedia dumps
- English Wikipedia (filtered)
- Telugu news articles
- Mixed Telugu-English conversational data
- HOME-specific domain data (home automation, family context)

## Structure

```
home-dataset/
├── src/
│   ├── downloaders/     # Dataset download scripts
│   ├── cleaners/        # Text cleaning pipelines
│   └── exporters/       # Export to tokenizer-ready format
├── scripts/             # One-off processing scripts
├── data/
│   ├── raw/             # Raw downloaded data (git-ignored)
│   └── processed/       # Cleaned, deduplicated data (git-ignored)
└── docs/
    └── sources.md       # Data source documentation
```

## Roadmap
- [ ] Telugu Wikipedia downloader
- [ ] English Wikipedia filtered downloader
- [ ] Text deduplication pipeline
- [ ] Quality filtering (perplexity-based)
- [ ] Export to home-tokenizer format