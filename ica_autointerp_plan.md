# ICA Dark Matter Autointerp Plan

## Goal
Run automated interpretability (autointerp) on all 100 ICA components extracted from SAE residuals, plus a matched baseline of 100 random SAE features. Produce quantitative interpretability scores for both sets to demonstrate that dark matter contains interpretable features comparable to SAE features.

## Prerequisites
- Saved ICA unmixing matrix (100 x 768) and whitening matrix from the original run
- Access to GPT-2 small + SAELens SAE (blocks.6.hook_resid_pre)
- ~1M tokens of cached activations from OpenWebText (or re-collect)
- Anthropic API key (or OpenAI for GPT-4)
- Budget: ~$5-15 in API calls (200 explain calls + 200 scoring calls)

## Architecture

### Step 1: Collect Rich Activation Examples

For each ICA component AND each baseline SAE feature, collect the **top 20 activating examples** with full context (not just the token).

```python
# For each ICA component i:
# 1. Project residuals onto ICA component direction
# 2. Find top 20 tokens by activation magnitude
# 3. For each token, extract surrounding context window (~50 tokens before, ~20 after)
# 4. Store: (context_text, highlighted_token, activation_value)

# For each SAE feature j (baseline):
# 1. Get SAE encoder activations for feature j
# 2. Same top-20 + context extraction
```

**Context format per example:**
```
...the government faced significant [backlash] from citizens who felt the policy...
                                     ^^^^^^^^^
                                     activation: 3.47
```

**Selecting baseline SAE features:**
- Pick 100 SAE features matched on activation frequency (density) to the ICA components
- This controls for the possibility that ICA components are just "easy" high-frequency features
- Alternatively: pick 100 random active SAE features (simpler, still informative)

### Step 2: Generate Explanations (Explain Pass)

For each component/feature, send the top 10 examples to Claude Sonnet 4 with this prompt:

```
System: You are an interpretability researcher analyzing directions in a neural network's activation space. You will be shown text excerpts where a particular direction activates strongly. The token in [brackets] is where the direction fires. Your job is to identify the common pattern, concept, or feature these examples share.

Respond with:
1. A short label (2-5 words) for what this direction represents
2. A one-sentence explanation
3. A confidence score from 1-5:
   - 5: Clear, monosemantic concept
   - 4: Coherent theme with minor variation
   - 3: Plausible pattern but somewhat noisy
   - 2: Weak or ambiguous pattern
   - 1: No discernible pattern / random

Format your response as JSON:
{"label": "...", "explanation": "...", "confidence": N}

User: Here are the top 10 activating examples for direction #{component_id}:

Example 1 (activation: {val}):
{context with [highlighted] token}

Example 2 (activation: {val}):
{context with [highlighted] token}

...
```

**Key design choices:**
- Use top 10 (not all 20) for explanation — reserve remaining 10 for scoring
- Use Claude Sonnet 4 (claude-sonnet-4-20250514) — good balance of quality and cost
- Temperature 0 for reproducibility
- Parse JSON response, store alongside component metadata

### Step 3: Score Explanations (Detection Pass)

For each component/feature, test whether the explanation generalizes to held-out examples. Use the remaining 10 activating examples + 10 random non-activating examples (20 total, shuffled).

```
System: You are evaluating whether a proposed explanation for a neural network direction is correct. You will be given an explanation and 20 text excerpts. For each excerpt, predict whether this direction would activate strongly on the [bracketed] token (YES or NO).

The proposed explanation is: "{explanation}"

Respond with a JSON array of 20 predictions:
[{"example_id": 1, "prediction": "YES"}, ...]

User: 
Example 1:
{context with [highlighted] token}

Example 2:
{context with [highlighted] token}
...
```

**Scoring metric: Detection balanced accuracy**
```python
# True positives: correctly predicted YES for activating examples
# True negatives: correctly predicted NO for non-activating examples  
# Balanced accuracy = (TP_rate + TN_rate) / 2
# Random baseline = 0.5, perfect = 1.0
```

This follows the Bills et al. / EleutherAI methodology. A score > 0.7 is generally considered interpretable.

### Step 4: Aggregate and Compare

Produce these summary statistics:

**Table 1: Interpretability comparison**
| Metric | ICA Components (n=100) | SAE Features (n=100) |
|--------|----------------------|---------------------|
| Mean confidence (1-5) | ? | ? |
| Median confidence | ? | ? |
| Detection balanced accuracy (mean) | ? | ? |
| Fraction with confidence ≥ 3 | ? | ? |
| Fraction with detection acc > 0.7 | ? | ? |

**Table 2: Breakdown by novelty**
| Metric | Novel ICA (cosine < 0.3, n=19) | Overlapping ICA (cosine ≥ 0.3, n=81) |
|--------|-------------------------------|--------------------------------------|
| Mean detection accuracy | ? | ? |
| Mean confidence | ? | ? |

**Key hypothesis to test:** Novel ICA components (the 19 with cosine < 0.3) have detection accuracy significantly above chance (0.5), demonstrating the dark matter contains genuinely interpretable features.

### Step 5: Qualitative Showcase

For the blog post, present the 5-10 ICA components with highest detection accuracy among the novel set (cosine < 0.3). For each, show:
- The autointerp label and explanation
- Detection accuracy score
- 3 example activations
- Cosine similarity to nearest SAE feature

## File Structure
```
ica_autointerp/
├── collect_examples.py      # Step 1: gather activation examples
├── explain.py               # Step 2: generate explanations via API
├── score.py                 # Step 3: detection scoring via API
├── analyze.py               # Step 4: aggregate stats + tables
├── data/
│   ├── ica_examples.json    # Top activating examples per ICA component
│   ├── sae_examples.json    # Top activating examples per SAE feature  
│   ├── explanations.json    # Autointerp labels + explanations
│   └── scores.json          # Detection accuracy scores
└── results/
    ├── comparison_table.md  # Summary tables for blog post
    └── showcase.md          # Top components for blog post
```

## Runtime & Cost Estimate

| Step | Time | Cost |
|------|------|------|
| Collect examples (GPU) | ~30 min | Free (local) |
| Explain pass (200 calls) | ~15 min | ~$3-5 |
| Score pass (200 calls) | ~15 min | ~$3-5 |
| Analysis | ~5 min | Free |
| **Total** | **~1 hour** | **~$6-10** |

## Implementation Notes

1. **Batching**: Can batch multiple components per API call if context window allows, but single-component calls are simpler and more reliable.

2. **Negative examples for scoring**: Sample random tokens from the dataset that are NOT in the top 1% of activations for that component. Important: sample from content tokens only (exclude `<|endoftext|>`).

3. **Context window**: 50 tokens before + 20 tokens after the highlighted token. This gives enough context for the LLM to understand the semantic role without flooding the prompt.

4. **Deduplication**: If multiple top examples come from the same document or have near-identical context, deduplicate to avoid inflating interpretability scores.

5. **SAE baseline selection**: For the fairest comparison, match SAE features by activation density (fraction of tokens where feature fires above threshold). ICA components are denser, so matching on density avoids comparing against ultra-sparse SAE features that are trivially interpretable.

6. **Alternative baseline**: Also consider running autointerp on 100 random directions (not ICA, not SAE — just random unit vectors projected into residual space). This establishes a "noise floor" — if random directions score 0.5 and ICA scores 0.7+, that's strong evidence.

## Expected Results & What They Mean

**Best case**: Novel ICA components score comparably to SAE features on detection accuracy (both > 0.7). This means the dark matter contains features of similar quality to what SAEs find — SAEs are just missing them due to the sparsity prior.

**Good case**: Novel ICA components score above chance (> 0.6) but below SAE features. This still shows interpretable structure in the dark matter, just somewhat noisier — consistent with the lower kurtosis.

**Null result**: Novel ICA components score near chance (~0.5). This would mean the manual interpretations were cherry-picked or illusory, and the dark matter is indeed unstructured from an interpretability standpoint. (Still publishable as a negative result.)
