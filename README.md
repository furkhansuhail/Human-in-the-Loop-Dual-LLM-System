# Human-in-the-Middle (HITM) Dual LLM System

A quality control pipeline that adds automated AI-powered review and human oversight layers between AI response generation and end users. The system uses two different LLMs in complementary roles: one generates responses while the other evaluates quality.

## Overview

The HITM system implements a robust quality assurance workflow for AI-generated content by:

1. **Generating responses** using either Anthropic Claude or OpenAI GPT models
2. **Evaluating quality** using a different LLM as an independent judge
3. **Flagging concerns** automatically based on configurable quality criteria
4. **Enabling human review** for responses that don't meet quality thresholds
5. **Logging all interactions** for continuous improvement and audit trails

## Key Features

### Dual LLM Architecture
- **Flexible Role Assignment**: Choose which LLM generates responses; the other automatically becomes the quality judge
- **Provider Options**: Support for Anthropic Claude (claude-sonnet-4-20250514) and OpenAI GPT (gpt-4o-mini, gpt-4o)
- **Independent Evaluation**: Separate LLMs ensure unbiased quality assessment

### Quality Control Mechanisms
- **LLM-as-a-Judge**: Automated quality evaluation using structured prompts
- **Configurable Thresholds**: Adjust sensitivity for flagging responses
- **Multiple Quality Dimensions**: Safety, accuracy, relevance, completeness, and clarity checks
- **Confidence Scoring**: Numerical assessment of response quality (0-10 scale)

### Human Review Integration
- **Conditional Review**: Only flagged responses require human attention
- **Review Modes**: Auto-approve, forced review, or intelligent routing
- **Three-Option Workflow**: Approve, edit, or reject with feedback
- **Audit Trail**: Complete logging of all decisions and edits

### Developer Experience
- **Interactive CLI**: User-friendly command-line interface
- **Automatic API Key Detection**: Searches for `.env` or `keys.env` files
- **Comprehensive Logging**: JSONL format for easy analysis
- **Statistics Dashboard**: Real-time metrics on system performance

## System Requirements

### Dependencies
```
python >= 3.10
langchain >= 0.1.0
langchain-anthropic >= 0.1.0
langchain-openai >= 0.0.5
python-dotenv >= 1.0.0
```

### API Keys Required
- **Anthropic API Key**: Get from [Anthropic Console](https://console.anthropic.com/)
- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)

## Installation

### 1. Clone or Download
```bash
# Download the hitm_dual_LLM.py file to your project directory
```

### 2. Install Dependencies
```bash
pip install langchain langchain-anthropic langchain-openai python-dotenv
```

### 3. Configure API Keys

Create a `.env` or `keys.env` file in your project directory:

```bash
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
OPENAI_API_KEY=sk-your-openai-key-here
```

**Alternative Methods:**

**Linux/Mac:**
```bash
export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
export OPENAI_API_KEY=sk-your-openai-key-here
```

**Windows:**
```cmd
set ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
set OPENAI_API_KEY=sk-your-openai-key-here
```

## Usage

### Basic Usage

Run the interactive demo:

```bash
python hitm_dual_LLM.py
```

### Workflow

1. **Select Generator LLM**: Choose which LLM generates responses
   - Option 1: Claude generates, GPT-4o-mini judges
   - Option 2: GPT-4o-mini generates, Claude judges
   - Option 3: GPT-4o generates, Claude judges

2. **Enter User Query**: Type the question or request

3. **Choose Review Mode**:
   - Auto: Judge decides if human review is needed
   - Force: Every response requires human review

4. **Review Process** (if flagged):
   - View original response
   - See quality concerns from judge
   - Choose: Approve, Edit, or Reject

5. **Final Delivery**: Approved content is delivered to user

### Example Session

```
User Query: "Explain quantum computing to a 10-year-old"

Review mode:
  [1] Auto (judge decides if review needed)
  [2] Force review for every response
Choice [1/2]: 1

GENERATOR LLM (Claude):
[Response generated...]

JUDGE LLM (GPT-4o-mini):
Quality Score: 8/10
Issues Found: None
PASSED ✓ Auto-approved (no review needed)

FINAL RESPONSE TO USER:
[Approved response delivered]
```

## System Architecture

### Core Components

#### 1. Data Models
- **Request**: User query with metadata and timestamp
- **AIResponse**: Generated response with model info
- **QualityCheck**: Judge evaluation results with reasoning
- **HumanReview**: Human decision and feedback
- **HITMConfig**: System configuration and parameters

#### 2. AI Response Generator
- Supports Anthropic and OpenAI models
- Configurable temperature and token limits
- Structured prompt handling
- Error handling and retries

#### 3. Quality Checker
- Independent LLM evaluation
- Structured judgment criteria
- JSON-formatted assessment
- Configurable quality thresholds

#### 4. Human Review System
- Interactive CLI prompts
- Three-decision workflow (approve/edit/reject)
- Feedback capture
- Timestamp tracking

#### 5. Feedback Logger
- JSONL logging format
- Complete audit trail
- Statistical aggregation
- Easy data analysis

### Process Flow

```
User Query
    ↓
Generate Response (LLM 1)
    ↓
Quality Check (LLM 2 as Judge)
    ↓
Decision Point
    ├─→ PASSED → Auto-approve → Deliver
    └─→ FLAGGED → Human Review
                      ├─→ Approve → Deliver
                      ├─→ Edit → Deliver edited version
                      └─→ Reject → (Optionally regenerate)
```

## Configuration Options

### HITMConfig Parameters

```python
config = HITMConfig(
    # Generator settings
    generator_provider="anthropic",  # or "openai"
    generator_model="claude-sonnet-4-20250514",
    generator_temperature=0.7,
    generator_max_tokens=1024,
    
    # Judge settings
    judge_provider="openai",  # or "anthropic"
    judge_model="gpt-4o-mini",
    judge_temperature=0.3,
    judge_max_tokens=1024,
    
    # Quality thresholds
    quality_threshold=6.0,  # 0-10 scale
    
    # Logging
    log_file="hitm_dual_llm_feedback.jsonl"
)
```

### Quality Evaluation Criteria

The judge LLM evaluates responses on:
- **Safety**: No harmful, offensive, or inappropriate content
- **Accuracy**: Factually correct information
- **Relevance**: Addresses the user's question
- **Completeness**: Thorough and sufficient answer
- **Clarity**: Well-structured and easy to understand

## Quality Check Example

### Judge Prompt Structure
```
You are a quality assurance judge evaluating AI responses.

Original Query: [user query]
AI Response: [generated response]

Evaluate on these criteria:
1. Safety (no harmful content)
2. Accuracy (factually correct)
3. Relevance (answers the question)
4. Completeness (sufficient detail)
5. Clarity (well-structured)

Provide assessment in JSON format:
{
  "passed": true/false,
  "confidence": 8.5,
  "flags": [],
  "reasoning": "explanation"
}
```

### Sample Judge Response
```json
{
  "passed": true,
  "confidence": 8.5,
  "flags": [],
  "reasoning": "Response is accurate, safe, and clearly explains quantum computing concepts in age-appropriate language. Good use of analogies."
}
```

## Statistics and Monitoring

### View Statistics
During interactive session, type `stats` to see:

```
============================================================
HITM SYSTEM STATISTICS
============================================================
Total Requests:     25
Approved:           20
Edited:             3
Rejected:           2
Flagged by System:  8

Approval Rate:      80.0%
============================================================
```

### Log File Format

Each interaction is logged in JSONL format:

```json
{
  "request": {
    "id": "req_20250116_123456_abc123",
    "user_query": "Explain quantum computing",
    "timestamp": "2025-01-16T12:34:56.789"
  },
  "ai_response": {
    "content": "...",
    "model": "claude-sonnet-4-20250514",
    "provider": "anthropic"
  },
  "quality_check": {
    "passed": true,
    "judge_provider": "openai",
    "judge_model": "gpt-4o-mini",
    "confidence": 8.5,
    "flags": []
  },
  "human_review": {
    "decision": "approve",
    "feedback": "Auto-approved"
  }
}
```

## Programmatic Usage

### Initialize System
```python
from hitm_dual_LLM import HITMSystem, HITMConfig

# Configure
config = HITMConfig(
    generator_provider="anthropic",
    generator_model="claude-sonnet-4-20250514",
    judge_provider="openai",
    judge_model="gpt-4o-mini",
    quality_threshold=7.0
)

# Initialize
hitm = HITMSystem(config=config, validate_api=True)
```

### Process Requests
```python
# Process with automatic review routing
final_response = hitm.process(
    user_query="What is machine learning?",
    require_review=None  # Auto mode
)

# Force human review
final_response = hitm.process(
    user_query="Sensitive medical question",
    require_review=True  # Always review
)

# Get statistics
stats = hitm.get_statistics()
print(f"Approval rate: {stats['approved']/stats['total']*100:.1f}%")
```

### Update Configuration
```python
# Change LLM models during runtime
new_config = HITMConfig(
    generator_provider="openai",
    generator_model="gpt-4o",
    judge_provider="anthropic",
    judge_model="claude-sonnet-4-20250514"
)

hitm.update_config(new_config)
```

## API Key Management Features

The system includes robust API key handling:

### Automatic Detection
- Searches current directory and up to 5 parent directories
- Looks for `keys.env` or `.env` files
- Reports which file was found and loaded

### Validation
- Checks for presence of required keys
- Validates key format (prefixes)
- Provides helpful error messages
- Masks keys in console output for security

### Error Handling
- Clear instructions if keys are missing
- Links to where to obtain API keys
- Validation before system initialization

## Troubleshooting

### Common Issues

**"ANTHROPIC_API_KEY not found"**
- Create a `.env` or `keys.env` file in your directory
- Ensure the key starts with `sk-ant-`
- Check that the file is in the correct location

**"OPENAI_API_KEY not found"**
- Add OPENAI_API_KEY to your environment file
- Ensure the key starts with `sk-`
- Verify you have an active OpenAI account

**"Failed to initialize HITM system"**
- Check your internet connection
- Verify both API keys are valid and active
- Ensure you have API credits/quota remaining
- Check that LangSmith tracing is disabled (set in code)

**Rate Limiting**
- The system makes 2 API calls per query (generator + judge)
- Monitor your API usage in provider dashboards
- Consider adding delays for high-volume usage

### Debug Mode

Enable detailed logging by checking the console output during initialization. The system provides step-by-step status updates.

## Best Practices

### Quality Thresholds
- **High Stakes (7-9)**: Medical, legal, financial advice
- **Medium Stakes (5-7)**: General information, explanations
- **Low Stakes (3-5)**: Creative content, casual queries

### Review Strategies
- **Auto Mode**: Efficient for high-volume, low-risk queries
- **Force Review**: Use for sensitive topics or regulated content
- **Threshold Tuning**: Adjust based on your use case and observed performance

### Model Selection
- **Claude as Generator**: Better for nuanced, contextual responses
- **GPT-4o as Generator**: Faster, cost-effective for straightforward queries
- **Claude as Judge**: Thorough evaluation, catches subtle issues
- **GPT-4o-mini as Judge**: Quick, efficient quality checks

## Logging and Analytics

### Analyzing Logs

Load and analyze the JSONL log file:

```python
import json

# Read logs
with open('hitm_dual_llm_feedback.jsonl', 'r') as f:
    logs = [json.loads(line) for line in f]

# Analyze approval rates by model
anthropic_responses = [l for l in logs if l['ai_response']['provider'] == 'anthropic']
approval_rate = sum(1 for l in anthropic_responses if l['human_review']['decision'] == 'approve') / len(anthropic_responses)

# Find common flags
all_flags = []
for log in logs:
    all_flags.extend(log['quality_check']['flags'])

from collections import Counter
print(Counter(all_flags).most_common(5))
```

## Extension Ideas

### Potential Enhancements

1. **Multi-Judge Ensemble**: Use multiple judges for critical decisions
2. **Adaptive Thresholds**: Automatically adjust based on domain/context
3. **Learning from Feedback**: Train on human review patterns
4. **Web Dashboard**: Graphical interface for reviews and analytics
5. **A/B Testing**: Compare different LLM configurations
6. **Integration Hooks**: Connect to existing customer service platforms
7. **Batch Processing**: Handle multiple queries efficiently
8. **Custom Evaluation Criteria**: Domain-specific quality metrics
9. **Response Caching**: Reduce API calls for common queries
10. **Multi-Language Support**: Evaluate responses in different languages

## Security Considerations

- **API Key Protection**: Never commit `.env` files to version control
- **Sensitive Data**: Be cautious with PII in logged interactions
- **Access Control**: Implement authentication for production deployments
- **Audit Compliance**: Use logs for regulatory requirements
- **Content Filtering**: Additional safety layers for public-facing deployments

## Performance Metrics

Typical performance characteristics:

- **Latency**: 2-5 seconds per query (generator + judge)
- **Accuracy**: Dependent on models and thresholds
- **Cost**: ~2x standard API usage (two LLM calls)
- **Throughput**: Limited by API rate limits

## Contributing

This is a reference implementation demonstrating HITM concepts. Potential improvements:

- Add more LLM providers (Cohere, Mistral, etc.)
- Implement response caching
- Add async processing
- Build web interface
- Create evaluation benchmarks
- Add more sophisticated judge prompts

## License

This implementation is provided as an educational example. Modify and extend as needed for your use case.

## Support and Resources

- **Anthropic Documentation**: https://docs.anthropic.com
- **OpenAI Documentation**: https://platform.openai.com/docs
- **LangChain Documentation**: https://python.langchain.com/docs

## Acknowledgments

Built using:
- **LangChain**: Framework for LLM applications
- **Anthropic Claude**: Advanced language model
- **OpenAI GPT**: Powerful language models
- **Python**: Programming language

---

**Version**: 1.0  
**Last Updated**: November 2025  

For questions, issues, or suggestions, please consult the documentation or modify the code to suit your specific needs.
