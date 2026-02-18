# Step-by-Step Instructions: Running the Full Pipeline

**What this does:** Runs 7 AI models against TikTok health transcript data to prove that
multiple LLMs can reliably annotate content without human coders. Produces all the CSV
tables and statistics needed to write and publish a research paper.

**Time estimate:** The annotation step (Step 4) is the longest. With 500 chunks, 7 models,
2 temperatures, and 5 runs each = 210,000 individual inferences. On a single GPU this
could take 12-48+ hours depending on hardware. Everything else combined takes minutes.

---

## BEFORE YOU START: Checklist

Make sure all of these are true before doing anything:

- [ ] You are on the Windows PC where this project lives
- [ ] PostgreSQL is installed and the `tiktok_disorders` database exists with data
- [ ] Ollama is installed (https://ollama.com)
- [ ] Anaconda or Miniconda is installed
- [ ] You have internet access (needed to download AI models the first time)

---

## PART 1: ONE-TIME SETUP (only do this once, ever)

### 1.1 Open a terminal

Press `Win + R`, type `cmd`, press Enter. Or open Anaconda Prompt from the Start Menu.

### 1.2 Navigate to the project folder

```
cd "C:\Users\schol\Documents\Python Projects\tiktok_research"
```

### 1.3 Create the Python environment

This installs all the required Python packages into an isolated environment.

```
conda env create -f environment.yml
```

If it says the environment already exists, that's fine. Skip to 1.4.

### 1.4 Activate the environment

**You must do this every time you open a new terminal window.**

```
conda activate tiktok-research
```

Your prompt should now show `(tiktok-research)` at the beginning.

### 1.5 Download the NLTK tokenizer data

This is a one-time download needed for splitting transcripts into sentences.

```
python -c "import nltk; nltk.download('punkt_tab')"
```

### 1.6 Verify the .env file exists

The file `.env` in the project folder should contain the database password. Open it in
Notepad and confirm it looks like this (with the real password filled in):

```
ANNOTATION_DB_NAME=tiktok_disorders
ANNOTATION_DB_USER=postgres
ANNOTATION_DB_PASSWORD=<the real password>
ANNOTATION_DB_HOST=localhost
ANNOTATION_DB_PORT=5433
OLLAMA_URL=http://localhost:11434
```

If this file doesn't exist, create it with the values above.

### 1.7 Start PostgreSQL

Make sure PostgreSQL is running. You can check by opening pgAdmin or running:

```
psql -U postgres -h localhost -p 5433 -d tiktok_disorders -c "SELECT COUNT(*) FROM videos;"
```

If this returns a number, the database is working. If it fails, start PostgreSQL from
the Windows Services panel (search "Services" in Start Menu, find "postgresql", click Start).

### 1.8 Start Ollama

Open a **separate** terminal window and run:

```
ollama serve
```

Leave this window open. Ollama must be running the entire time the pipeline is running.

If you get "address already in use", Ollama is already running. That's fine.

### 1.9 Download all 7 AI models

This downloads ~100GB total of model weights. It only needs to happen once. Each model
takes 5-30 minutes depending on your internet speed.

Run each of these commands one at a time, waiting for each to finish:

```
ollama pull glm-4.7-flash
ollama pull phi4:latest
ollama pull gpt-oss:20b
ollama pull alibayram/medgemma:27b
ollama pull gemma3:27b
ollama pull qwen3:32b
ollama pull deepseek-r1:32b
```

To verify they all downloaded:

```
ollama list
```

You should see all 7 models in the list.

### 1.10 Verify models work

Back in the project folder terminal (with `tiktok-research` env active):

```
python test_models.py
```

This sends a test prompt to every model and checks the response. You should see
`PASS` for all 7 models. If any show `FAIL`:
- Make sure Ollama is running (`ollama serve` in another terminal)
- Make sure the model was pulled (`ollama pull <model_name>`)
- Try pulling the model again if it shows errors

### 1.11 Run the tests

This verifies the code itself is working correctly:

```
python -m pytest tests/ -v
```

You should see **71 passed**. If any fail, something is wrong with the code — do not proceed.

---

## PART 2: RUNNING THE PIPELINE

### Overview of what happens

The pipeline has 3 phases, run separately:

| Phase | Purpose | Chunks | Split | What you get |
|-------|---------|--------|-------|-------------|
| Phase 1 | Tune & test | 100 | development | Quick check that everything works |
| Phase 2 | Main study | 500 | reliability | The actual data for the paper |
| Phase 3 | Confirm | 200 | holdout | Proof that results generalize |

---

### PHASE 1: Development Run (quick test, ~1-4 hours)

**Purpose:** Make sure everything works end-to-end before committing to the full study.

#### Step 1: Open terminal, activate environment, navigate to project

```
conda activate tiktok-research
cd "C:\Users\schol\Documents\Python Projects\tiktok_research"
```

#### Step 2: Run the pipeline on the development split

```
python run_pipeline.py --name "dev_v1" --chunk-limit 100 --splits development
```

**What this does:**
1. Creates database tables (if they don't exist)
2. Selects creators into dev/reliability/holdout groups
3. Chunks transcripts into pieces
4. Runs every chunk through all 7 models, 2 temperatures, 5 times each
5. Computes stability statistics
6. Derives final labels
7. Validates against existing data
8. Generates CSV report files
9. Runs all extended analyses (cross-model, convergence, significance, etc.)

**While it's running:**
- The terminal will show progress logs
- The annotation step (Step 4/15) takes the longest — it will show completion percentage
- Do NOT close the terminal or shut down the computer
- Do NOT close the Ollama window

**If it crashes or you need to stop it:**
- Note the experiment ID number shown in the logs (e.g., "Experiment ID: 1")
- Press `Ctrl+C` to stop
- To resume later, run:
  ```
  python run_pipeline.py --experiment-id 1 --skip-to annotate
  ```
  (Replace `1` with your actual experiment ID)

#### Step 3: Check if the results look viable

```
python progress_report.py
```

This prints a report card. Look at the **VIABILITY ASSESSMENT** section at the bottom.
You want to see mostly `[PASS]` results.

**If you see `[FAIL]` on most checks:** The prompts or models may need adjustment.
This is normal for a first run. You may need to examine the sample responses and
adjust prompts in `prompts.py`.

**If you see `[PASS]` on most checks:** You're good. Proceed to Phase 2.

#### Step 4: Look at the output files

Open the folder: `outputs\experiment_1\` (the number matches your experiment ID)

You should see CSV files. Open a few in Excel to verify they contain data.

---

### PHASE 2: Main Study (the real run, ~12-48 hours)

**Purpose:** Generate all the data for the paper. This is the big run.

#### Step 1: Start the main study run

```
python run_pipeline.py --name "main_study" --chunk-limit 500 --splits reliability
```

**IMPORTANT:** This will take many hours. Plan to let the computer run overnight or
over a weekend. Make sure:
- The computer won't go to sleep (Settings > System > Power > set sleep to "Never")
- Ollama stays running
- PostgreSQL stays running

#### Step 2: Monitor progress (optional, while it's running)

In a second terminal:

```
conda activate tiktok-research
cd "C:\Users\schol\Documents\Python Projects\tiktok_research"
python progress_report.py
```

This shows how many tasks are completed vs remaining.

#### Step 3: If it crashes mid-annotation

Find your experiment ID from the logs, then resume:

```
python run_pipeline.py --experiment-id <YOUR_ID> --skip-to annotate
```

The pipeline picks up exactly where it left off. No work is lost.

#### Step 4: If annotation finished but later steps crashed

```
python run_pipeline.py --experiment-id <YOUR_ID> --skip-to stability
```

This re-runs stability computation and everything after it.

#### Step 5: After it finishes, check results

```
python progress_report.py
```

Look for:
- Stability rate >= 70% — GOOD
- Krippendorff alpha >= 0.667 — GOOD
- Coverage rate >= 80% — GOOD
- Clarity rate >= 90% — GOOD
- `VERDICT: VIABLE` — you can write the paper

---

### PHASE 3: Holdout Confirmation (~4-12 hours)

**Purpose:** Prove the results from Phase 2 aren't a fluke.

```
python run_pipeline.py --name "holdout_confirm" --chunk-limit 200 --splits holdout
```

After it finishes:

```
python progress_report.py
```

If the holdout metrics are similar to Phase 2 metrics, the results generalize. This is
important for the paper — reviewers will want to see this.

---

## PART 3: COLLECTING THE DATA FOR THE PAPER

After all 3 phases complete, you have everything needed.

### Where are the files?

All output CSVs are in:
```
C:\Users\schol\Documents\Python Projects\tiktok_research\outputs\
```

Inside, there's one folder per experiment:
```
outputs\
  experiment_1\    <-- Phase 1 (development)
  experiment_2\    <-- Phase 2 (main study) *** THIS IS THE MAIN ONE ***
  experiment_3\    <-- Phase 3 (holdout confirmation)
```

### Which files go where in the paper?

Open the **Phase 2** folder (`experiment_2` or whatever your main study ID is).

#### For the RESULTS section:

| Open this file | It becomes | What it shows |
|---------------|------------|---------------|
| `table1_group_reliability.csv` | Table 1 | Main reliability results: alpha, kappa, stability rate (with confidence intervals) for every construct x model x temperature |
| `table2_stability_by_model.csv` | Table 2 | Which models are most reliable? Averaged across constructs |
| `table3_stability_by_temperature.csv` | Table 3 | Does temperature matter? T=0.0 vs T=0.5 |
| `table4_coverage_clarity.csv` | Table 4 | What % of chunks have health content? What % get clear labels? |
| `table5_label_distributions.csv` | Table 5 | Distribution of labels (e.g., how many "past" vs "present" vs "future") |
| `table6_cost_efficiency.csv` | Table 6 | How fast is LLM annotation vs estimated human annotation? |

#### For the VALIDITY section:

| Open this file | What it proves |
|---------------|---------------|
| `cross_model_agreement_matrix.csv` | Every model pair agrees at rate X% (THE headline finding) |
| `cross_model_krippendorff.csv` | Cross-model alpha with confidence intervals |
| `cross_model_consensus.csv` | What % of chunks get unanimous agreement across all models? |
| `construct_correlation_matrix.csv` | Constructs are independent (low correlations = good) |
| `validation_results.csv` | Annotations match existing database values at X% |

#### For the METHODS JUSTIFICATION section:

| Open this file | What it proves |
|---------------|---------------|
| `run_convergence.csv` | R=5 runs is better than R=1 (justifies the multi-run approach) |
| `significance_temperature.csv` | Statistical test: temperature effect (p-value) |
| `significance_model_size.csv` | Statistical test: does model size affect reliability? |
| `significance_constructs.csv` | Statistical test: are some constructs harder than others? |

#### For the ROBUSTNESS section:

| Open this file | What it shows |
|---------------|-------------|
| `stratified_by_content_type.csv` | Reliability broken down by narrative vs informational content |
| `stratified_by_chunk_length.csv` | Reliability broken down by short vs long text chunks |
| `stratified_by_temporal_position.csv` | Reliability broken down by early vs late in creator's timeline |
| `stratified_by_diagnosis.csv` | Reliability broken down by the creator's claimed diagnosis |
| `stratified_by_context_carry.csv` | Reliability with vs without prior context |

#### For the ERROR ANALYSIS section:

| Open this file | What it shows |
|---------------|-------------|
| `confusion_within_model.csv` | When a model is unstable, which labels does it confuse? |
| `confusion_cross_model.csv` | When models disagree, what are the common disagreement patterns? |
| `error_by_chunk_length.csv` | Are longer or shorter chunks more error-prone? |
| `error_by_content_type.csv` | Which content types cause the most errors? |

#### For the APPENDIX:

| Open this file | What it's for |
|---------------|-------------|
| `qualitative_samples.csv` | Example chunks for manual inspection: unanimous, disagreement, none, unclear |
| `full_per_chunk_stability.csv` | Complete raw data export (every chunk, every metric) |
| `figure_stability_threshold_curve.csv` | Data to plot: "if we require higher agreement, how many chunks survive?" |

#### For comparing Phase 2 vs Phase 3 (holdout):

Open the same files from both `experiment_2/` and `experiment_3/` side by side.
The key numbers to compare are:
- `table1_group_reliability.csv` — alpha and stability_rate columns
- `cross_model_krippendorff.csv` — alpha column
If Phase 3 numbers are similar to Phase 2, report this as "results generalize to held-out data."

---

## TROUBLESHOOTING

### "conda is not recognized"
You need to open Anaconda Prompt (from Start Menu), not regular cmd.

### "No module named psycopg2" or "No module named dotenv"
You forgot to activate the environment. Run: `conda activate tiktok-research`

### "connection refused" on database
PostgreSQL isn't running. Open Windows Services, find "postgresql", click Start.

### "connection refused" on Ollama
Ollama isn't running. Open a new terminal and run: `ollama serve`

### Pipeline says "No models available"
You haven't pulled the models yet. Go back to step 1.9 and run the `ollama pull` commands.

### Pipeline crashes during annotation
This is fine. Note the experiment ID from the log output, then resume:
```
python run_pipeline.py --experiment-id <ID> --skip-to annotate
```

### Pipeline crashes AFTER annotation (during stability, reporting, etc.)
Resume from stability — it will redo stability and everything after:
```
python run_pipeline.py --experiment-id <ID> --skip-to stability
```

### "No experiment found with ID X"
You're using the wrong experiment ID. Check which experiments exist:
```
psql -U postgres -h localhost -p 5433 -d tiktok_disorders -c "SELECT experiment_id, experiment_name, status FROM experiment_runs;"
```

### Computer went to sleep / lost power during annotation
Resume it:
```
python run_pipeline.py --experiment-id <ID> --skip-to annotate
```

### Want to re-run just the reports (no re-annotation)
```
python run_pipeline.py --experiment-id <ID> --skip-to reporting
```

### Want to re-run just the extended analyses
```
python run_pipeline.py --experiment-id <ID> --skip-to cross_model
```

### A model is really slow / hanging
You can run with fewer models. For example, to skip the slowest ones:
```
python run_pipeline.py --name "fast_run" --chunk-limit 500 --splits reliability --models glm-4.7-flash phi4:latest qwen3:32b
```
But for the paper, using all 7 models makes the strongest argument.

### Tests fail when running `pytest`
Make sure you activated the conda environment first:
```
conda activate tiktok-research
python -m pytest tests/ -v
```

---

## QUICK REFERENCE: Common Commands

```bash
# Always start with these two lines in any new terminal:
conda activate tiktok-research
cd "C:\Users\schol\Documents\Python Projects\tiktok_research"

# Run everything from scratch:
python run_pipeline.py --name "my_study" --chunk-limit 500

# Resume a crashed run:
python run_pipeline.py --experiment-id 1 --skip-to annotate

# Check progress while running:
python progress_report.py

# Re-generate reports only:
python run_pipeline.py --experiment-id 1 --skip-to reporting

# Run extended analyses only:
python run_pipeline.py --experiment-id 1 --skip-to cross_model

# Run tests:
python -m pytest tests/ -v

# Test that models work:
python test_models.py
```
