# SwarmAgentic AI Agent Demonstration System
![swarmagentic](https://github.com/user-attachments/assets/70c6ee47-b2c3-4589-83f6-165ead8a49a5)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/release/ai-in-pm/SwarmAgentic.svg)](https://github.com/ai-in-pm/SwarmAgentic/releases)
[![GitHub stars](https://img.shields.io/github/stars/ai-in-pm/SwarmAgentic.svg)](https://github.com/ai-in-pm/SwarmAgentic/stargazers)

A comprehensive AI Agent demonstration system featuring **real-time animated visualizations** of swarm intelligence, BERT-based PhD-level reasoning, and interactive SwarmAgentic demonstrations.

![SwarmAgentic Demo](https://img.shields.io/badge/Demo-Live%20Animation-brightgreen)
![AI Level](https://img.shields.io/badge/AI%20Level-PhD-red)
![Visualization](https://img.shields.io/badge/Visualization-Real--time-orange)

## 🎬 Live Demo

https://github.com/user-attachments/assets/8ec464dc-fde4-427c-a6c1-0ba299793d46


Experience SwarmAgentic in action with our **real-time animated demonstrations**:

- **🔄 Swarm Formation**: Watch agents self-organize using flocking algorithms
- **⚡ PSO Optimization**: See particle swarm optimization in real-time
- **🤝 Agent Collaboration**: Observe multi-agent task coordination
- **🌟 Emergent Behavior**: Witness complex patterns from simple rules

## 🚀 Quick Start

### Option 1: Standalone Executable (Recommended)
1. Download the latest release from [Releases](https://github.com/ai-in-pm/SwarmAgentic/releases)
2. Extract and run `SwarmAgentic_Demo.exe`
3. No installation required!

### Option 2: Run from Source
```bash
git clone https://github.com/ai-in-pm/SwarmAgentic.git
cd SwarmAgentic
python -m venv ai_agent_env
ai_agent_env\Scripts\activate  # Windows
# source ai_agent_env/bin/activate  # Linux/macOS
pip install -r requirements.txt
python demo_launcher.py  # For animated demo
# or
python main.py  # For full AI system
```

## ✨ Features

### 🎨 Real-Time Animated Visualizations
- **🔄 Swarm Formation**: Live flocking behavior with separation, alignment, and cohesion
- **⚡ PSO Optimization**: Real-time particle swarm optimization with convergence tracking
- **🤝 Multi-Agent Collaboration**: Dynamic coordinator-worker interactions
- **🌟 Emergent Behavior**: Complex collective intelligence patterns
- **🎮 Interactive Controls**: Speed adjustment, mode switching, real-time parameters

### 🧠 PhD-Level AI Capabilities
- **BERT-Based Reasoning**: Advanced language processing with attention mechanisms
- **Chain-of-Thought Processing**: Visible reasoning steps with confidence scoring
- **Real-Time Scratchpad**: Live visualization of agent's working memory
- **Dynamic Learning**: Adaptive training on SwarmAgentic academic content
- **Knowledge Graph**: Concept relationships and understanding depth

### 📊 Advanced Monitoring & Analytics
- **Output Accuracy Tracking**: Real-time percentage monitoring with trend analysis
- **Expected Discoveries**: Probability metrics for pattern recognition
- **Source Quality Assessment**: Information reliability and freshness scoring
- **Performance Dashboard**: Comprehensive metrics with visual indicators
- **Quality Trends**: Historical analysis and prediction capabilities

### 🖥️ Modern User Interface
- **Animated Visualization Panel**: Real-time SwarmAgentic demonstrations
- **Interactive Control Panel**: Start Training, Demo, Pause, Stop, Reset
- **Progress Visualization**: Live training and demonstration progress
- **Metrics Dashboard**: Real-time performance and quality indicators
- **Professional Styling**: Modern design with intuitive navigation

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10, Linux, macOS | Windows 11 |
| **Python** | 3.8+ | 3.11+ |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 2GB free | 5GB+ |
| **CPU** | Multi-core (i5/Ryzen 5) | i7/Ryzen 7+ |
| **GPU** | Integrated | Dedicated (for faster BERT) |
| **Display** | 1400x900 | 1920x1080+ |

## 📦 Installation

### Option 1: Standalone Executable (Recommended)
```bash
# Download from GitHub Releases
wget https://github.com/ai-in-pm/SwarmAgentic/releases/latest/download/SwarmAgentic_Demo.exe
# Or download manually from: https://github.com/ai-in-pm/SwarmAgentic/releases

# Run immediately - no installation required!
./SwarmAgentic_Demo.exe
```

### Option 2: From Source
```bash
# Clone the repository
git clone https://github.com/ai-in-pm/SwarmAgentic.git
cd SwarmAgentic

# Create and activate virtual environment
python -m venv ai_agent_env

# Windows
ai_agent_env\Scripts\activate

# Linux/macOS
source ai_agent_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Validate installation
python test_installation.py

# Run the application
python demo_launcher.py  # Animated demo
# OR
python main.py           # Full AI system
```

### Option 3: Quick Test
```bash
# Test the animated visualization system
python test_animated_demo.py
```

## Quick Start Guide

### 1. Initial Setup
- Launch the application
- The system will automatically initialize all components
- Wait for the "System Ready" status

### 2. Training the Agent
- Click **"Start Training"** to begin training on the academic paper
- Monitor progress in the training progress bar
- Watch real-time thoughts in the scratchpad
- Training typically takes 2-5 minutes

### 3. Running Demonstrations
- After training completion, **"Start Demo"** becomes enabled
- Click **"Start Demo"** to begin interactive demonstrations
- Monitor demonstration progress and agent responses
- View real-time accuracy and confidence metrics

### 4. Monitoring and Analysis
- **Scratchpad**: View agent's real-time thought processes
- **Working Memory**: Monitor active memory contents
- **Performance Metrics**: Track accuracy, confidence, and processing times
- **Quality Dashboard**: Assess information source reliability

## System Architecture

### Core Components

#### 1. BERT Reasoning Engine (`src/core/bert_engine.py`)
- Advanced language processing with BERT integration
- Chain-of-thought reasoning implementation
- Confidence scoring and attention analysis
- Real-time performance metrics

#### 2. AI Agent Core (`src/core/ai_agent.py`)
- PhD-level cognitive architecture
- Dynamic knowledge acquisition
- Working memory management
- State management and callbacks

#### 3. Learning System (`src/training/learning_system.py`)
- Adaptive learning algorithms
- Knowledge graph construction
- Concept relationship mapping
- Progress validation

#### 4. Quality Metrics (`src/monitoring/quality_metrics.py`)
- Real-time accuracy tracking
- Source quality assessment
- Expected discovery patterns
- Comprehensive monitoring dashboard

#### 5. Interactive Demo (`src/demonstration/interactive_demo.py`)
- Scenario-based demonstrations
- Performance evaluation
- Interactive response generation
- Success metrics calculation

#### 6. GUI Framework (`src/gui/main_window.py`)
- Modern interface design
- Real-time visualization
- Control panel integration
- Progress monitoring

#### 7. Text PSO Synthesis (rule-based, new) (`src/synthesis/`)
- Minimal PSO loop that evolves agent-system specs (roles + workflow) without external LLMs

## How PSO Works in This Project (End‑to‑End)

This section explains the complete PSO workflow and how the system learns, selects tasks, searches for teams, scores them, and optionally uses an LLM for suggestions.

### 1) Training (Domain Understanding)
- The agent loads a domain corpus with this priority:
  1. `TRAINING_TEXT_PATH` (env)
  2. `data/agent_ops_handbook.txt`
  3. `academic_summary.md`
- Using `BERTReasoningEngine` and `AdaptiveLearningSystem`, it mines and validates concepts, builds a knowledge graph in memory, and exports it for the UI.
- The entire corpus is cached to `data/_training_corpus.txt`, enabling the evaluator to extract domain keywords for a “domain_coverage” bonus (see the fitness function below).
- The system also derives “recommended concepts” that can seed PSO tasks (preset: “From Training”).

### 2) Task Selection (What Teams Should Cover)
- Task sets live in `data/tasks.json` (e.g., General, Software Delivery, Research). Select via UI or API `/api/tasks/select`.
- Options: Default (built‑in), From Training, or named categories / explicit lists.
- The UI shows a task‑focused concept list so “Learned Concepts” match your selection (using the knowledge graph export when available).

### 3) Text‑Based PSO (Team Synthesis)
- A team is plain text with:
  - `roles`: list of `{name, responsibilities[], tools[]}`
  - `workflow`: list of step strings
- The engine (`PSOSwarmSynthesizer`) initializes a diverse population, then iteratively mutates candidates and updates personal/global bests. Diversity features include:
  - Unique initial signatures (sorted role names + workflow)
  - Duplicate checks when inserting children
  - Expanded role/workflow pools to improve keyword coverage in domain categories
- Streaming: each iteration streams population metrics, Top‑K, and the current global best. The UI plots coverage (X) and fitness (Y).

### 4) Fitness Function (Scoring Teams)
Implemented in `src/synthesis/eval.py` and matches this weighted sum:

```
coverage        = (# matched task keywords) / (# total keywords)
size_bonus      = 1.0 if 3 ≤ #roles ≤ 6 else 0.7
verify_bonus    = 0.15 if any workflow step contains "verify"|"check"|"audit" else 0.0
redundancy_pen. = 0.2 if duplicate role names exist else 0.0
wf_penalty      = 0.4 if workflow length == 0
                = 0.2 if workflow length > 10
                = 0.0 otherwise
domain_coverage = match against domain keywords (optional, from training cache)
w                = PSO_DOMAIN_WEIGHT (e.g., 0.2)

fitness = clamp(0, 1, 0.5*coverage + 0.2*size_bonus + verify_bonus + w*domain_coverage
                      - redundancy_penalty - wf_penalty)
```

Notes:
- The evaluator scans role names, responsibilities, and workflow text to compute coverage.
- If the training cache is present, domain_coverage provides an additional, tunable bonus.

### 5) Optional LLM Suggestions
- Operators in `src/synthesis/llm_ops.py`:
  - `llm_mutate_system` — 1 improved candidate
  - `llm_mutate_batch` — multiple candidates in one call
- Controls via env vars:
  - `USE_LLM_PSO`, `LLM_PSO_PROB`, `LLM_PSO_MAX_CALLS_PER_ITER`, `LLM_PSO_GBEST_ONLY`, `LLM_PSO_BATCH`, `LLM_PSO_BATCH_COUNT`, `OPENAI_MODEL`
- Circuit breaker & backoff protect against rate limits. All LLM proposals are sanitized for duplicate role names before evaluation.

### Tuning Knobs (UI & Env)
- UI controls (“Viz Controls”):
  - Iteration Pause (s) — updates backend `TEXT_PSO_PAUSE`
  - Reveal ms, Spread, Dot px — visual presentation
- Backend env:
  - `TEXT_PSO_PAUSE`, `TEXT_PSO_POP`, `TEXT_PSO_ITERS`
  - `BERT_MODEL_ID`, `TRAINING_TEXT_PATH`, `PSO_DOMAIN_WEIGHT`, `OPENAI_API_KEY`

### Files to Check
- Agent + streaming: `src/core/ai_agent.py`
- Evaluator: `src/synthesis/eval.py`
- PSO core: `src/synthesis/pso_text.py`
- LLM ops: `src/synthesis/llm_ops.py`
- Frontend: `frontend/src/App.jsx`, `frontend/src/api.js`

- Files:
  - `src/synthesis/agent_spec.py`: schema for roles and workflow
  - `src/synthesis/eval.py`: heuristic fitness (coverage, verification, balance)
  - `src/synthesis/pso_text.py`: `PSOSwarmSynthesizer` running a small population over iterations
- Demo integration: a new scenario “Agent Synthesis via PSO (Rule-based)” runs this loop and shows the best system
- Note: This bridges to the paper’s idea. Full language-driven PSO with prompt-based mutations is out of scope here.

## Configuration

### Application Settings
The system can be configured by modifying the configuration in `main.py`:

```python
config = {
    "bert_model_path": "bert-base-uncased-mrpc/...",
    "academic_paper_path": "academic_summary.md",
    "max_sequence_length": 512,
    "quality_metrics_window": 1000,
    "enable_real_time_monitoring": True,
    "demo_auto_start": False,
    "export_results": True
}
```
docker run --rm -p 8000:8000 -e HF_HOME=/cache/huggingface -e BERT_MODEL_ID=prajjwal1/bert-tiny -v ${PWD}\.cache\huggingface:/cache/huggingface -v ${PWD}\results:/app/results -v ${PWD}\logs:/app/logs swarmagentic-web:latest

### BERT Model Configuration
- Default: Intel optimized BERT base uncased MRPC
- Location: `bert-base-uncased-mrpc/bert-base-uncased-mrpc/huggingface_Intel_bert-base-uncased-mrpc_v1`
- Supports custom BERT models with compatible interfaces

## Usage Examples

### Basic Operation
1. **Training**: Start training → Monitor progress → Wait for completion
2. **Demonstration**: Start demo → Observe scenarios → Review results
3. **Analysis**: Monitor metrics → Export results → Review performance

### Advanced Features
- **Custom Scenarios**: Modify demonstration scenarios in the code
- **Metric Export**: Automatic export of session results
- **Real-time Monitoring**: Continuous quality assessment
- **Performance Tuning**: Adjust learning parameters
- **Agent Synthesis Demo (new)**: Run a compact PSO loop that evolves an agent-system spec and prints the best roles/workflow during the demo

## Troubleshooting

### Common Issues

#### Installation Problems
- **Missing Dependencies**: Ensure all requirements are installed
- **Python Version**: Verify Python 3.8+ is being used
- **Virtual Environment**: Use virtual environment to avoid conflicts

#### Runtime Issues
- **BERT Model Loading**: Verify model path and files exist
- **Memory Issues**: Ensure sufficient RAM (8GB minimum)
- **GUI Problems**: Check display resolution and graphics drivers

#### Performance Issues
- **Slow Training**: Consider GPU acceleration or reduce sequence length
- **High Memory Usage**: Adjust batch sizes and window sizes
- **Unresponsive GUI**: Check for blocking operations in main thread

### Error Messages
- **"BERT model not found"**: Check model path configuration
- **"Academic paper not found"**: Verify `academic_summary.md` exists
- **"GUI initialization failed"**: Check tkinter installation
- **"Import errors"**: Verify all dependencies are installed

## Development

### Building from Source
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Build executable
python build_executable.py
```

### Project Structure
```
ai-agent-demo/
├── src/
│   ├── core/           # Core AI components
│   ├── gui/            # User interface
│   ├── training/       # Learning systems
│   ├── monitoring/     # Quality metrics
│   └── demonstration/  # Demo scenarios
├── bert-base-uncased-mrpc/  # BERT model
├── logs/               # Application logs
├── results/            # Exported results
├── main.py            # Application entry point
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting section above
- Review the logs in the `logs/` directory
- Create an issue on the project repository

## Acknowledgments

- BERT model: Intel optimized BERT base uncased
- Academic content: SwarmAgentic research paper
- GUI framework: tkinter with matplotlib integration
- Machine learning: PyTorch and Transformers libraries
### Text PSO Synthesis Configuration (optional)
- Env `USE_LLM_PSO=1`: Enables LLM-driven mutations in the text-based PSO agent synthesis.
- Env `OPENAI_API_KEY`: Required if `USE_LLM_PSO=1`. Optionally set `OPENAI_MODEL` (default `gpt-4o-mini`) and `OPENAI_BASE_URL` for compatible endpoints.
- When enabled, the "Agent Synthesis" step runs a small PSO over agent specs and the visualization switches to a real population view (mode `text_pso`).
