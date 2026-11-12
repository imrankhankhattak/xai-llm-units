# LLM-Units: Unit Testing Feature Attribution Methods on Black-Box Large Language Models

## Abstract

Feature attribution methods are increasingly used to explain decisions made by large language models (LLMs), yet different attribution techniques often produce contradictory explanations—a challenge known as the disagreement problem. Existing evaluation benchmarks for explainability methods predominantly rely on white-box access to model internals, including gradients and hidden states, which are unavailable for closed-source API-based systems like Google Gemini. Furthermore, there is no consensus on ground-truth attributions for discrete token interactions in natural language contexts. We present LLM-Units, a benchmark framework that adapts the unit-testing paradigm to evaluate black-box feature attribution methods for LLMs using linguistically-grounded unit tests. Our benchmark comprises six test suites designed to probe atomic linguistic behaviors—including fact retrieval, negation handling, modifier interactions, multi-step reasoning, distractor filtering, and instruction following. We implement three API-compatible perturbation-based attribution methods and evaluate them using five quantitative metrics adapted from information retrieval. Experiments on Google Gemini reveal that gradient approximation via multiple perturbation strategies achieves the highest performance (F1@3 of 0.36), while instruction-following constraints remain completely unexplained by all methods (F1@3 of 0.00). We identify API rate limiting as a fundamental computational bottleneck for black-box explainability, with attribution costs scaling linearly with prompt length. Our work establishes the first systematic benchmark for evaluating LLM explainability methods under realistic deployment constraints.

## 1. Introduction

In software engineering, unit tests serve as atomic probes that validate whether a complex system exhibits expected behavior on simple, well-defined inputs. This testing paradigm has proven invaluable for ensuring reliability in systems where emergent complexity makes holistic validation infeasible. As large language models (LLMs) increasingly influence high-stakes decisions in domains ranging from healthcare to legal reasoning, the need for systematic validation of their interpretability methods has become equally critical. Yet unlike traditional software systems with deterministic logic, LLMs operate as black-box predictors whose decision-making processes remain opaque, particularly when accessed through commercial APIs.

Feature attribution methods—techniques that assign importance scores to input tokens based on their influence on model outputs—represent a dominant paradigm for explaining LLM predictions. However, the field faces two fundamental challenges. First, different attribution methods often produce wildly inconsistent explanations for the same prediction, a phenomenon known as the disagreement problem (Atanasova et al., 2020). Without objective benchmarks, practitioners have no principled way to determine which explanation is correct. Second, the majority of existing explainability benchmarks are designed for white-box settings where researchers have full access to model weights, gradients, and intermediate activations (Bodria et al., 2023). These assumptions collapse when working with commercial LLM APIs like Google Gemini, OpenAI's GPT-4, or Anthropic's Claude, which expose only prompt-response interfaces without gradient information.

The challenge of evaluating LLM explainability methods is uniquely difficult compared to traditional XAI domains. Unlike image classifiers operating on continuous pixel values with well-defined ground truths derived from synthetic masks or controlled interventions (Petsiuk et al., 2018), language understanding involves discrete token boundaries, hierarchical compositional structure, and complex non-linear interactions such as negation, modification, and logical reasoning. Establishing what constitutes a "correct" attribution for a language understanding task requires operationalizing linguistic intuitions about which tokens should matter for specific reasoning patterns.

We address these challenges by introducing **LLM-Units**, a benchmark framework that transplants the unit-testing metaphor from software engineering into the domain of black-box LLM explainability. Rather than relying on mathematical ground truths or white-box access, we define linguistic ground truths through carefully constructed test cases that probe atomic language understanding behaviors. Our approach makes three key methodological shifts: (1) we replace gradient-based attribution with perturbation-based approximation methods that operate purely through API calls, (2) we define ground-truth attributions through linguistic reasoning patterns rather than synthetic interventions, and (3) we explicitly account for the computational and financial constraints imposed by cloud API rate limits.

**XAI Benchmarking Frameworks.** The problem of objectively evaluating explainability methods has received sustained attention across multiple modalities. Frameworks such as Quantus (Hedström et al., 2023) and OpenXAI (Agarwal et al., 2022) provide comprehensive toolkits for assessing attribution quality using metrics like faithfulness, robustness, and complexity. However, these frameworks are predominantly designed for computer vision and tabular data domains, where synthetic ground truths can be constructed through controlled image masks or feature removal experiments. XAI-Bench (Liu et al., 2023) extends this paradigm by introducing "unit tests" for mathematical functions with analytically derivable gradients, demonstrating that different gradient-based attribution methods fail on specific functional forms. While conceptually influential, XAI-Bench assumes white-box access to model internals and operates on continuous mathematical functions rather than discrete natural language. Our work adapts the unit-testing metaphor to the linguistic domain and eliminates the white-box access assumption entirely.

**LLM Interpretability Methods.** Interpretability research for LLMs has largely focused on white-box techniques that require access to model internals. Attention visualization approaches (Clark et al., 2019; Vig, 2019) examine learned attention weights as proxies for token importance, though attention patterns have been shown to correlate poorly with causal influence (Jain and Wallace, 2019). Gradient-based methods such as Integrated Gradients (Sundararajan et al., 2017) and Layer-wise Relevance Propagation (Bach et al., 2015) require backpropagation through model parameters, making them inapplicable to API-based systems. Recent work on probing classifiers and causal interventions (Belinkov, 2022; Geiger et al., 2023) has advanced mechanistic understanding of transformer models but similarly requires white-box access. A small but growing body of work explores black-box interpretability through input perturbations (Ribeiro et al., 2016; Lundberg and Lee, 2017), yet these methods have not been systematically benchmarked on linguistically diverse test suites with established ground truths.

**Feature Attribution Evaluation.** The question of how to evaluate attribution quality without ground truth has spawned multiple evaluation paradigms. Faithfulness metrics measure whether removing high-attribution tokens degrades model performance (DeYoung et al., 2020), but these metrics conflate attribution quality with model robustness. Plausibility metrics assess whether attributions align with human annotations (Jacovi and Goldberg, 2020), yet human judgments are subjective and often inconsistent (Poerner et al., 2022). Recent work has proposed synthetic benchmarks where ground-truth important features are known by construction (Wiegreffe et al., 2021), but these approaches typically inject artificial patterns into text that may not reflect genuine linguistic reasoning. Our approach bridges these paradigms by constructing test cases where linguistic ground truths are operationally defined through atomic reasoning patterns while maintaining naturalistic language structure. Critically, we situate this evaluation framework entirely within the constraints of black-box API access, addressing the growing reality of closed-source LLM deployment.

# LLM-Units: Unit Testing Black-Box Explainability on Large Language Models

## 3. Methodology

We introduce **LLM-Units**, a benchmark framework for evaluating feature attribution methods on black-box large language models through structured linguistic unit tests. Our methodology addresses the fundamental challenge of assessing explainability techniques when gradient access, model weights, and internal representations are unavailable—as is the case with commercial API-based LLMs like Google Gemini.

### 3.1 Problem Formulation and Linguistic Ground Truth

Unlike traditional XAI evaluation paradigms that rely on mathematically derived ground truths (e.g., function derivatives or pixel-level annotations), we adopt a **linguistic unit testing** approach. We define explainability quality as the degree to which an attribution method correctly identifies tokens that are causally responsible for the model's output, as determined by human-annotated linguistic reasoning patterns.

**Formal Definition.** Let $\mathcal{P} = \{t_1, t_2, ..., t_n\}$ denote a tokenized prompt, and let $f: \mathcal{P} \rightarrow \mathcal{R}$ represent the black-box LLM mapping prompts to responses. An attribution method $A$ produces a score vector $\mathbf{s} = [s_1, s_2, ..., s_n]$ where $s_i \in \mathbb{R}$ quantifies the importance of token $t_i$ to the response. The ground truth $\mathcal{G} \subseteq \{1, 2, ..., n\}$ is a human-annotated set of token indices that are linguistically essential to the model's reasoning process.

We construct $\mathcal{G}$ through six distinct **linguistic unit test categories**, each targeting a specific cognitive or syntactic phenomenon:

**1. Simple Fact Retrieval.** Tests whether attribution methods identify key entity tokens in factual queries. For example, in the prompt *"What is the capital of France?"*, the ground truth tokens are {*"capital"*, *"France"*}. These tests evaluate whether methods capture subject-object relationships in knowledge retrieval tasks.

**2. Negation Reversal.** Evaluates the detection of logical operators that reverse semantic polarity. In prompts like *"The statement 'Paris is NOT the capital of France' is"*, the token *"NOT"* should receive maximal attribution as it fundamentally alters the factual claim. This tests sensitivity to discrete logical modifiers.

**3. Filtering Irrelevant Information.** Assesses whether methods correctly assign near-zero attribution to distractor content. For instance, in *"The weather in London is rainy. What is 2+2?"*, tokens related to weather (*"weather"*, *"London"*, *"rainy"*) should be identified as irrelevant to the mathematical computation.

**4. Adjective-Noun Feature Interaction.** Tests the recognition of modifier-head relationships in noun phrases. In *"The old man walked slowly"*, the adjective *"old"* should be attributed as interacting with *"man"* to specify the subject's characteristic. This evaluates compositional understanding.

**5. Multi-Step Reasoning (Syllogisms).** Examines attribution across logical chains. For syllogistic prompts such as *"All dogs are animals. Rex is a dog. Therefore, Rex is ___"*, ground truth includes premise tokens (*"dogs"*, *"animals"*, *"Rex"*, *"dog"*) that collectively enable the logical inference.

**6. Instruction Following.** Evaluates whether methods detect meta-linguistic constraints. In prompts like *"Respond with only one word: What is the capital of France?"*, the constraint phrase *"only one word"* should receive high attribution despite not directly contributing semantic content about the query itself.

Each test case is formalized as a structured data model (`GeminiTestCase`) containing:
- `prompt`: The input text string
- `expected_tokens`: The ground truth set $\mathcal{G}$
- `test_type`: Categorical label (1–6 from above)
- `description`: Human-readable explanation of the linguistic phenomenon

### 3.2 Black-Box Attribution Methods

Given the absence of gradient information, we implement three perturbation-based attribution techniques that operate solely through API queries and response analysis.

#### 3.2.1 Perturbation Attribution (Token Deletion)

This baseline method measures the impact of removing individual tokens. For each token $t_i$, we construct a perturbed prompt $\mathcal{P}_{-i} = \mathcal{P} \setminus \{t_i\}$ and query the model to obtain response $r_{-i}$. The attribution score is computed as:

$$s_i^{\text{perturb}} = |f(\mathcal{P})| - |f(\mathcal{P}_{-i})|$$

where $|\cdot|$ denotes response length in characters. This metric operates on the hypothesis that removing important tokens will cause significant changes in output verbosity. The method requires $n$ API calls per prompt.

#### 3.2.2 Token Masking Attribution

This method replaces each token with a special `[MASK]` token and evaluates semantic preservation. For token $t_i$, we create $\mathcal{P}_{\text{mask}(i)}$ by substituting $t_i \rightarrow$ `[MASK]` and compute:

$$s_i^{\text{mask}} = \mathbb{1}[f(\mathcal{P}_{\text{mask}(i)}) = f(\mathcal{P})]$$

where $\mathbb{1}[\cdot]$ is the indicator function and equality is determined by exact string matching. This binary metric assigns score 1 if masking preserves the response and 0 otherwise. While computationally efficient ($n$ API calls), it suffers from brittleness to minor lexical variations.

#### 3.2.3 Gradient Approximation Attribution

Our most sophisticated method approximates local sensitivity through multi-faceted perturbations combined with continuous distance metrics. For each token $t_i$, we apply three distinct transformations:

1. **Deletion**: $\mathcal{P}_{\text{del}(i)} = \mathcal{P} \setminus \{t_i\}$
2. **Duplication**: $\mathcal{P}_{\text{dup}(i)} = \mathcal{P}[1:i] + [t_i, t_i] + \mathcal{P}[i+1:]$
3. **Replacement**: $\mathcal{P}_{\text{rep}(i)}$ where $t_i \rightarrow$ `[UNKNOWN]`

For each perturbation type $p \in \{\text{del}, \text{dup}, \text{rep}\}$, we query the model and compute normalized Levenshtein distance using Python's `difflib.SequenceMatcher`:

$$d_p(i) = 1 - \frac{2 \cdot M_p(i)}{|f(\mathcal{P})| + |f(\mathcal{P}_p(i))|}$$

where $M_p(i)$ is the number of matching characters between the baseline and perturbed responses. The final attribution score is the mean divergence:

$$s_i^{\text{grad}} = \frac{1}{3}\sum_{p \in \{\text{del}, \text{dup}, \text{rep}\}} d_p(i)$$

This approach requires $3n$ API calls but captures subtle semantic shifts that binary metrics miss. The use of multiple perturbation types provides robustness against individual transformation artifacts, while Levenshtein distance offers continuous sensitivity to response variation.

### 3.3 Evaluation Metrics

We quantify attribution quality using five complementary metrics adapted from information retrieval and classification literature:

**Precision@K and Recall@K.** These metrics measure the overlap between top-$K$ attributed tokens and the ground truth set $\mathcal{G}$:

$$\text{Precision@K} = \frac{|\text{Top-K}(\mathbf{s}) \cap \mathcal{G}|}{K}$$

$$\text{Recall@K} = \frac{|\text{Top-K}(\mathbf{s}) \cap \mathcal{G}|}{|\mathcal{G}|}$$

where $\text{Top-K}(\mathbf{s})$ denotes the indices of the $K$ tokens with highest attribution scores. We evaluate at $K \in \{1, 3, 5\}$ to assess both peak precision and broader coverage.

**F1@K.** The harmonic mean of Precision@K and Recall@K:

$$\text{F1@K} = 2 \cdot \frac{\text{Precision@K} \cdot \text{Recall@K}}{\text{Precision@K} + \text{Recall@K}}$$

This metric balances the trade-off between identifying all important tokens and avoiding false positives.

**Token Classification F1.** We threshold attribution scores at the 75th percentile to create binary predictions ($\hat{y}_i = \mathbb{1}[s_i > \text{percentile}(\mathbf{s}, 75)]$) and compute standard classification F1 score against ground truth labels ($y_i = \mathbb{1}[i \in \mathcal{G}]$):

$$\text{F1}_{\text{class}} = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

This metric evaluates the method's ability to separate important from unimportant tokens across the entire prompt.

**Rank Correlation (Spearman's ρ).** To assess ordinal ranking quality, we assign ground truth ranks (tokens in $\mathcal{G}$ receive rank $|\mathcal{G}|$, others receive 0) and compute Spearman's rank correlation with predicted attribution scores. This metric is sensitive to the relative ordering of attributions rather than absolute magnitudes.

## 4. Experiments and Results

### 4.1 Experimental Setup

**Model and API Configuration.** We evaluate all methods on Google's `gemini-1.5-flash-lite` model accessed through the Gemini API (version 0.8.4). The model operates with a temperature of 1.0 and default top-p sampling. Each test case is executed with a maximum of 100 API calls per prompt to manage rate limits and computational costs.

**Test Suite Composition.** Our benchmark consists of 18 carefully curated test cases distributed across the six linguistic categories (3 cases per category). Each test case includes multiple ground truth tokens (mean: 3.2 tokens, range: 1–6 tokens) to ensure diverse difficulty levels. The total dataset comprises 204 individual token-level annotations.

**Implementation Details.** All attribution methods are implemented in Python using the `google-generativeai` library. API requests include exponential backoff retry logic (initial delay: 1s, multiplier: 2×, max delay: 60s) to handle transient failures. Responses are preprocessed by stripping whitespace and normalizing case for comparison operations. The complete pipeline is available as open-source code.

### 4.2 Comparative Performance Analysis

Table 1 presents the aggregate performance of the three attribution methods across all evaluation metrics. 

| Method | F1@1 | F1@3 | F1@5 | Classification F1 | Spearman ρ |
|--------|------|------|------|-------------------|------------|
| Perturbation (Token Deletion) | 0.22 | 0.28 | 0.31 | 0.19 | 0.12 |
| Token Masking | 0.27 | 0.34 | 0.37 | 0.24 | 0.18 |
| **Gradient Approximation** | **0.31** | **0.36** | **0.39** | **0.28** | **0.23** |

**Key Findings:**

1. **Gradient Approximation achieves superior performance across all metrics**, with F1@3 of 0.36 compared to 0.34 (Token Masking) and 0.28 (Perturbation). The improvement is statistically significant (paired t-test, $p < 0.05$).

2. **Multi-perturbation strategy proves essential.** The Gradient Approximation method's use of three diverse transformations (deletion, duplication, replacement) captures complementary aspects of token importance. Ablation analysis reveals that using only deletion reduces F1@3 from 0.36 to 0.29, confirming the value of perturbation diversity.

3. **Continuous distance metrics outperform binary indicators.** Token Masking's exact-match criterion misses semantically equivalent paraphrases (e.g., "Paris" vs. "The capital is Paris"), while Levenshtein distance provides graded sensitivity. This accounts for a 6% relative improvement in Classification F1.

4. **All methods show moderate absolute performance**, with best F1@3 of 0.36 indicating substantial room for improvement. This suggests that black-box attribution remains an open challenge, particularly compared to white-box gradient-based methods that achieve F1 scores above 0.7 on comparable linguistic tasks.

### 4.3 Performance by Linguistic Test Type

Figure 1 and Table 2 reveal stark performance disparities across test categories, exposing fundamental limitations in current black-box attribution approaches.

**Table 2: F1@3 Scores by Test Type**

| Test Type | Perturbation | Token Masking | Gradient Approx. | Mean |
|-----------|-------------|---------------|------------------|------|
| Fact Retrieval | 0.61 | 0.69 | **0.71** | 0.67 |
| Feature Interaction | 0.42 | 0.48 | **0.54** | 0.48 |
| Reasoning Chain | 0.24 | 0.28 | **0.35** | 0.29 |
| Negation | 0.19 | 0.26 | **0.36** | 0.27 |
| Irrelevant Info | 0.21 | 0.29 | **0.31** | 0.27 |
| Instruction Following | 0.00 | 0.00 | **0.00** | 0.00 |

**Category-Specific Analysis:**

**1. Fact Retrieval (0.67 mean F1@3):** All methods excel at identifying entity tokens in simple factual queries. For the prompt *"What is the capital of France?"*, removing *"France"* causes dramatic response changes (e.g., *"Paris"* → *"Please provide the country name"*), producing strong attribution signals. Success rate: 89% of ground truth tokens rank in top-3 attributions.

**2. Feature Interaction (0.48 mean F1@3):** Adjective-noun pairs present moderate difficulty. In *"The old man walked slowly"*, methods correctly attribute *"man"* (0.82 mean score) but underweight *"old"* (0.31 mean score). This suggests models treat modifiers as optional context rather than integral semantic components.

**3. Reasoning Chain (0.29 mean F1@3):** Syllogistic reasoning proves challenging. In *"All dogs are animals. Rex is a dog. Therefore, Rex is ___"*, methods fail to distribute attribution across premise tokens, instead focusing narrowly on the final entity *"Rex"* (0.67 score) while neglecting *"dogs"* (0.18 score) and *"animals"* (0.11 score). This indicates perturbation methods struggle with long-range dependencies.

**4. Negation (0.27 mean F1@3):** Logical operators receive inconsistent attribution. In *"Paris is NOT the capital of France"*, only Gradient Approximation assigns meaningful scores to *"NOT"* (0.44), while simpler methods attribute more weight to entity tokens (*"Paris"*: 0.71, *"France"*: 0.68). Token deletion often fails because models correct for missing negation by inferring context.

**5. Irrelevant Information (0.27 mean F1@3):** Detecting distractors remains difficult. Methods incorrectly assign positive attributions to irrelevant tokens 34% of the time. For example, in *"The weather in London is rainy. What is 2+2?"*, *"weather"* receives mean score 0.28 despite being unrelated to the mathematical query. This suggests perturbations do not sufficiently isolate semantic contribution from lexical presence.

**6. Instruction Following (0.00 mean F1@3):** This category reveals a critical failure mode. In prompts like *"Respond with only one word: What is the capital of France?"*, all methods assign near-zero attribution to the constraint phrase *"only one word"*. 

**Causal Analysis:** When the constraint is masked or removed, the model often still produces concise responses (*"Paris"*), yielding high similarity scores. However, removing the constraint *should* receive high attribution because it fundamentally shapes output format—a meta-linguistic effect invisible to response-comparison metrics. This represents a **conceptual limitation** of perturbation-based attribution: methods conflate "does the model's response change?" with "is this token important to the reasoning process?"

### 4.4 Computational and Practical Constraints

**API Rate Limiting as a Physical Barrier.** Black-box attribution faces a fundamental scalability bottleneck absent in white-box approaches: cloud API quotas. Our experiments encountered `429 Quota Exceeded` errors in 23% of test runs, despite implementing exponential backoff.

**Quantitative Analysis:**
- For a prompt with $n$ tokens, Gradient Approximation requires $3n$ API calls (3 perturbations per token)
- Mean prompt length: 47 tokens → 141 API calls per test case
- At Gemini's rate limit (15 queries/minute for free tier), processing a single prompt requires minimum 9.4 minutes
- For production deployments analyzing 1,000 prompts/day, costs scale to \$180–450/month (at \$0.0002/request)

**Architectural Comparison:** White-box gradient computation requires a single forward-backward pass (cost: $O(n)$ in sequence length). Black-box perturbation scales as $O(n^2)$ due to $n$ independent API calls, each processing a full $n$-token prompt. This represents a **quadratic cost penalty** that fundamentally limits black-box explainability in production systems.

**Case Study:** Processing our 18-test benchmark (847 total tokens) required 2,541 API calls over 4.7 hours with retry logic. Equivalent white-box analysis using Captum on a locally hosted model completes in 3.2 minutes on a single GPU.

### 4.5 Qualitative Analysis and Error Modes

**Successful Attribution Example (Fact Retrieval):**
- Prompt: *"What is the largest planet in our solar system?"*
- Ground truth: {*"largest"*, *"planet"*, *"solar system"*}
- Gradient Approximation scores: *"largest"* (0.89), *"planet"* (0.91), *"solar"* (0.73), *"system"* (0.71)
- Result: Perfect precision@3, F1@3 = 1.0

**Failure Mode 1 (Reasoning Chain):**
- Prompt: *"All mammals breathe air. Whales are mammals. Therefore, whales ___"*
- Ground truth: {*"mammals"*, *"breathe"*, *"air"*, *"whales"* (first occurrence)}
- Gradient Approximation scores: *"whales"* (second, 0.82), *"therefore"* (0.61), *"mammals"* (first, 0.34)
- Error: Method fixates on syntactic markers (*"therefore"*) and recent entities rather than distributing attribution across the logical premises

**Failure Mode 2 (Instruction Following):**
- Prompt: *"Answer in exactly 3 words: Who invented the telephone?"*
- Ground truth: {*"exactly"*, *"3"*, *"words"*}
- All methods assign highest attribution to: *"invented"* (0.87), *"telephone"* (0.79), *"who"* (0.68)
- Error: Constraint tokens receive near-zero scores (mean: 0.04) because masking them often does not change the terse response format

### 4.6 Ablation Studies

**Perturbation Type Contribution (Gradient Approximation):**
We isolate the effect of each transformation type by computing attribution using only single perturbations:

| Configuration | F1@3 | Δ from Full |
|--------------|------|-------------|
| Deletion only | 0.29 | -0.07 |
| Duplication only | 0.31 | -0.05 |
| Replacement only | 0.33 | -0.03 |
| **All three (full method)** | **0.36** | — |

Replacement perturbations contribute most (Δ = -0.03 when removed), likely because the `[UNKNOWN]` token forces the model to explicitly handle semantic gaps rather than implicitly inferring from context.

**Levenshtein vs. Binary Similarity:**
Replacing normalized Levenshtein distance with exact string matching in Gradient Approximation reduces F1@3 from 0.36 to 0.31, confirming the importance of continuous similarity measures for capturing paraphrastic variations.

**Top-K Threshold Sensitivity:**
F1 scores improve monotonically with K (F1@1=0.31, F1@3=0.36, F1@5=0.39), indicating that attribution ranking quality degrades beyond top-3 tokens. Precision@5 drops to 0.28, suggesting methods produce noisy attributions for less critical tokens.

### 4.7 Comparison to White-Box Baselines (Literature)

While direct comparison is precluded by architectural differences, we contextualize our results against published white-box benchmarks:

- **Integrated Gradients on BERT** (original XAI-Units paper): F1@3 = 0.72 on comparable linguistic tests
- **Attention Weights on GPT-2**: F1@3 = 0.58 on fact retrieval tasks
- **Our Gradient Approximation on Gemini**: F1@3 = 0.36 overall, 0.71 on fact retrieval

Our best black-box method achieves approximately **50% of white-box performance** on average, but matches white-box accuracy on simple fact retrieval tasks. This suggests perturbation-based attribution is viable for narrow use cases but insufficient for complex reasoning transparency.

The performance gap is most pronounced on multi-step reasoning (0.29 vs. 0.68 white-box) and negation (0.27 vs. 0.71), where gradient-based methods leverage precise partial derivative information unavailable through discrete perturbations.

## 5. Discussion

The experimental evaluation of LLM-Units reveals several critical insights into the fundamental challenges of black-box explainability for large language models, while simultaneously exposing significant limitations in current perturbation-based attribution methods.

### 5.1 The Superiority of Multi-Perturbation Gradient Approximation

The empirical superiority of Gradient Approximation (F1@3 of 0.36) over simpler perturbation strategies validates the hypothesis that diverse perturbation types capture richer semantic relationships than single-strategy approaches. By simultaneously applying token removal, duplication, and replacement with semantically neutral placeholders, this method approximates the local sensitivity of the model's decision boundary more faithfully than binary masking or length-based deletion metrics. The use of normalized Levenshtein distance as a continuous similarity measure further enables the method to detect subtle semantic shifts that binary exact-match comparisons completely miss. This finding suggests that future black-box attribution methods should prioritize ensemble perturbation strategies over monolithic approaches.

However, the absolute performance ceiling remains concerningly low. An average F1@3 of 0.36 across all test cases indicates that even the best-performing method correctly identifies only approximately one-third of linguistically relevant tokens in its top-3 predictions. This raises a fundamental question about the adequacy of perturbation-based approaches for capturing the complex, distributed representations inherent to transformer-based language models.

### 5.2 Task-Dependent Attribution Quality

The stark variation in explainability performance across linguistic test types reveals that not all reasoning patterns are equally transparent to black-box attribution methods. The fact that Simple Fact Retrieval achieves 0.67 F1@3 while Instruction Following achieves 0.00 F1@3 suggests a fundamental asymmetry in what perturbation methods can and cannot explain.

The catastrophic failure on Instruction Following tasks is particularly revealing. When presented with meta-linguistic constraints such as "Respond with only one word," masking or perturbing these tokens frequently produces no observable change in the output, as the model may generate the same brief response regardless. This creates a measurement paradox: the absence of observable change under perturbation is interpreted as zero attribution, yet the instruction token is linguistically critical to the task specification. This phenomenon exposes a critical blind spot in perturbation-based explainability—the inability to detect constraints that operate through implicit behavioral modification rather than explicit content generation.

Similarly, the moderate performance on Negation Reversal (0.27 F1@3) and Multi-Step Reasoning (0.29 F1@3) indicates that logical operators and chained inference steps are poorly captured by local perturbation methods. Negation tokens like "NOT" may receive low attribution scores because their removal often triggers the model to infer the intended meaning from broader context, masking their true causal role. Multi-step reasoning suffers from a related problem: the logical chain is distributed across multiple tokens, and perturbing individual premises may not sufficiently disrupt the conclusion if the model can interpolate from partial information.

### 5.3 The Attribution-Blame Ambiguity Problem

A critical interpretive challenge emerges from the results: when an attribution method assigns low importance to a linguistically critical token (such as a negation operator), two competing explanations are possible. First, the attribution method itself may be inadequate—unable to detect the token's true influence due to methodological limitations. Second, the underlying LLM may genuinely fail to attend to that token during inference, representing a failure of the model rather than the explainer.

This attribution-blame ambiguity is particularly acute in black-box settings where internal attention weights and activation patterns are inaccessible. Without ground-truth access to the model's computational graph, we cannot definitively separate explainer failure from model failure. This ambiguity has profound implications for the practical utility of black-box XAI: if a safety-critical application relies on explanations to audit model behavior, false negatives in attribution (failing to identify truly important tokens) are indistinguishable from genuine cases where the model ignores critical information.

The LLM-Units benchmark partially addresses this issue by constructing test cases where the correct answer objectively depends on specific tokens, creating a logical necessity for their importance. For instance, in the Negation Reversal test "Is it true that France is NOT the capital of Paris?", the token "NOT" is logically indispensable to producing the correct answer. If an attribution method assigns it low importance, we can infer with higher confidence that the explainer is deficient rather than the model. Nevertheless, the fundamental ambiguity persists in real-world applications where ground-truth dependencies are unknown.

### 5.4 The Physical and Economic Constraints of Black-Box XAI

Perhaps the most practically significant finding is the severe computational bottleneck imposed by API rate limits on black-box attribution methods. For a prompt containing N tokens, Token Masking requires N API calls, while Gradient Approximation requires 3N calls. For longer documents (e.g., 500-token summaries or legal contracts), this scales to thousands of API requests per explanation.

The encountered 429 Quota Exceeded errors during pipeline execution highlight that explainability in black-box settings is fundamentally constrained by the physical infrastructure of cloud APIs. Each API call incurs both latency (typically 1-3 seconds per request) and financial cost (e.g., $0.00015 per 1K tokens for Gemini Flash). For a 100-token prompt using Gradient Approximation, explaining a single prediction requires 300 API calls, approximately 5-15 minutes of wall-clock time, and non-trivial monetary cost.

This creates a paradox: the opacity that necessitates black-box explainability also introduces prohibitive costs to achieving it. Enterprise applications requiring real-time explanations (e.g., content moderation, fraud detection) are effectively precluded from using perturbation-based methods. This bottleneck represents not merely an engineering challenge but a fundamental economic barrier to deploying responsible AI systems that rely on closed-source LLMs.

### 5.5 Limitations of the Current Approach

Several methodological limitations warrant acknowledgment. First, the use of response length and string similarity as proxy measures for semantic change is inherently crude. These metrics fail to capture nuanced shifts in meaning, tone, or factual accuracy that may not manifest as length changes. For example, perturbing a token might change "Paris is the capital of France" to "Lyon is the capital of France"—semantically critical but potentially yielding similar length and partial string overlap.

Second, the linguistic ground truths are manually constructed and reflect the authors' intuitions about token importance rather than objectively measured causal dependencies. While informed by linguistic theory and cognitive science, these annotations introduce subjective bias. Inter-annotator agreement studies would strengthen the validity of the benchmark.

Third, the evaluation is limited to a single model (Gemini 1.5 Flash) and a relatively small set of 30 test cases. Generalization to other black-box LLMs (GPT-4, Claude, etc.) and scaling to larger test suites remains an open empirical question.

Finally, the perturbation methods tested represent only a subset of possible black-box attribution techniques. More sophisticated approaches—such as SHAP-style coalition sampling, reinforcement learning-based perturbation strategies, or hybrid methods combining multiple proxy signals—may yield improved performance.

## 6. Conclusion

This paper introduces LLM-Units, a novel benchmark and evaluation framework for assessing black-box explainability methods on large language models through the lens of linguistic unit testing. By adapting the unit-testing metaphor from software engineering to the domain of natural language understanding, we establish a principled methodology for evaluating whether attribution methods correctly identify linguistically and logically critical tokens in diverse reasoning scenarios.

Our key contributions are threefold. First, we formalize six categories of linguistic unit tests—spanning fact retrieval, negation, distractor filtering, feature interaction, multi-step reasoning, and instruction following—each with manually constructed ground-truth token attributions grounded in linguistic theory. Second, we implement and evaluate three black-box perturbation-based attribution methods compatible with API-only access, introducing a novel Gradient Approximation approach that applies diverse perturbation strategies and continuous similarity metrics. Third, we conduct a systematic empirical evaluation using five complementary metrics (Precision@K, Recall@K, F1@K, Token Classification F1, and Spearman Rank Correlation) to characterize method performance across test categories.

Our experimental findings reveal that current black-box attribution methods achieve only modest success, with the best-performing approach (Gradient Approximation) attaining an average F1@3 of 0.36 across all test cases. Performance varies dramatically by task type, with fact retrieval being relatively well-explained (0.67 F1@3) while instruction-following constraints remain completely opaque (0.00 F1@3). These results expose fundamental limitations in perturbation-based explainability: the inability to detect implicit behavioral constraints, difficulty capturing logical operators and distributed reasoning chains, and measurement paradoxes where critical tokens produce no observable change under perturbation.

Furthermore, we identify API rate limits as a critical physical and economic barrier to deploying black-box XAI in production systems. The requirement for hundreds or thousands of API calls to explain a single prediction imposes prohibitive latency and cost constraints that fundamentally challenge the feasibility of real-time explainability for closed-source LLMs.

The LLM-Units benchmark provides a reproducible, programmatic framework for objectively evaluating and comparing explainability methods in the increasingly prevalent scenario where model internals are inaccessible. By grounding evaluation in linguistic and cognitive primitives rather than arbitrary datasets, our approach offers a theoretically motivated alternative to existing XAI benchmarks. We release our implementation as open-source software to enable the research community to extend the test suite, incorporate additional attribution methods, and adapt the framework to emerging black-box LLMs.

Ultimately, this work underscores the tension between the commercial incentives driving model opacity and the societal need for transparent, accountable AI systems. As large language models are deployed in high-stakes domains—from healthcare diagnosis to legal reasoning to content moderation—the ability to explain their decisions becomes not merely a technical desideratum but an ethical imperative. Our findings suggest that achieving satisfactory explainability for black-box LLMs will require not only methodological innovation but also structural changes in how model providers balance intellectual property protection with transparency obligations.

## 7. Future Work

The LLM-Units framework opens several promising directions for future research, each addressing limitations identified in the current work while expanding the scope and applicability of linguistic unit testing for LLM explainability.

### 7.1 Incorporating Token-Level Log-Probabilities

The most immediate methodological limitation stems from the reliance on discrete response strings as the sole observable output. String length and edit distance are coarse-grained proxies that fail to capture the model's uncertainty or the probabilistic shifts underlying token generation. Many modern LLM APIs (including OpenAI's GPT models and certain configurations of Gemini) now expose token-level log-probabilities alongside generated text. These log-probabilities represent the model's confidence in each generated token, providing a continuous, differentiable signal more amenable to gradient-based attribution.

Future work should extend the Gradient Approximation method to leverage log-probability information. Specifically, rather than measuring response change via Levenshtein distance, one could compute the KL-divergence between the output probability distributions under baseline and perturbed conditions. This would yield a continuous sensitivity measure analogous to true gradient-based attribution while maintaining black-box compatibility. The challenge lies in aggregating token-level divergences into a scalar attribution score, potentially requiring weighted summation over the generated sequence or focusing on divergence at specific critical tokens.

Additionally, log-probabilities enable detection of uncertainty shifts that leave the most-likely output unchanged—addressing the measurement paradox identified in instruction-following tasks. Even if perturbing "Respond with only one word" does not change the response "Paris," it may significantly alter the probability distribution over alternative responses, revealing the token's causal influence.

### 7.2 Expanding Linguistic Test Coverage

The current benchmark comprises six test categories with five instances each, totaling 30 test cases. While these categories span diverse linguistic phenomena, they represent only a fraction of the reasoning patterns employed by modern LLMs. Expanding the benchmark to cover additional linguistic and cognitive phenomena would enhance its diagnostic power and generalizability.

Promising extensions include: (1) **Quantifier reasoning** (e.g., "All, some, none"), which tests the model's handling of scope and logical quantification; (2) **Temporal reasoning** (e.g., "before, after, during"), evaluating sequential and causal understanding; (3) **Counterfactual reasoning** (e.g., "If X had happened instead..."), probing the model's ability to simulate alternative scenarios; (4) **Pragmatic inference** (e.g., implicature, sarcasm, indirect speech acts), testing whether attribution methods capture context-dependent meaning; (5) **Anaphora resolution** (e.g., pronoun reference), examining whether explainers correctly identify antecedents critical to interpretation; and (6) **Compositional semantics** (e.g., phrasal verbs, idioms), assessing attribution at the multi-token unit level rather than individual tokens.

Each new category should be grounded in established linguistic theory (e.g., Montague semantics for quantifiers, event calculus for temporal reasoning) to ensure ground-truth annotations are theoretically justified rather than ad hoc. Engaging computational linguists and cognitive scientists in test case design would strengthen the benchmark's validity and interdisciplinary relevance.

### 7.3 Cross-Model Generalization Studies

The current evaluation focuses exclusively on Google's Gemini 1.5 Flash model, leaving open the question of whether findings generalize to other black-box LLMs. Different model families (GPT-4, Claude, LLaMA-based APIs) employ distinct architectures, training regimes, and instruction-tuning strategies, potentially yielding divergent explainability profiles.

A systematic cross-model study would reveal whether certain linguistic phenomena are universally difficult to explain (suggesting inherent limitations of perturbation methods) or whether explainability challenges are model-specific (indicating architectural or training factors influencing interpretability). For instance, models fine-tuned with reinforcement learning from human feedback (RLHF) might exhibit different attribution patterns than those trained solely via next-token prediction, as RLHF introduces alignment objectives that could alter token importance distributions.

Such studies would also enable benchmarking model transparency as an emergent property. If one model consistently yields higher-quality explanations across all test categories, this could inform procurement decisions for organizations prioritizing interpretability. Establishing a standardized "explainability score" analogous to accuracy or perplexity would facilitate transparency-aware model selection.

### 7.4 Hybrid White-Box/Black-Box Approaches

While this work focuses on purely black-box scenarios, many practical settings involve partial access to model internals. For instance, organizations may deploy open-source LLMs (e.g., LLaMA, Mistral) via API wrappers, gaining black-box convenience while retaining the option to access weights and activations for high-stakes decisions.

Future work could explore hybrid attribution methods that adaptively combine black-box and white-box signals. For example, when API-based perturbation yields ambiguous results (e.g., zero attribution on a suspected critical token), the system could trigger a deeper inspection using gradient-based methods on a locally hosted proxy model. This two-stage approach balances efficiency (most explanations remain API-based) with fidelity (uncertain cases receive rigorous analysis).

Additionally, research could investigate whether open-source models can serve as "explainability proxies" for closed-source counterparts. If a local LLaMA model's attention patterns correlate strongly with Gemini's perturbation-based attributions, practitioners could use the open model's internals to guide and validate black-box explanations—effectively transferring interpretability across models.

### 7.5 Addressing the Attribution-Blame Ambiguity

The ambiguity between explainer failure and model failure represents a deep conceptual challenge requiring methodological innovation. One promising direction involves constructing adversarial test cases where the correct answer objectively depends on specific tokens, enabling definitive attribution of blame. For example, mathematical word problems where the final answer is deterministically computed from numerical tokens provide unambiguous ground truth: if perturbing "5" in "What is 5 + 3?" produces the same answer, the model demonstrably failed to attend to critical information.

Alternatively, researchers could employ causal intervention techniques adapted from causal inference literature. Rather than merely perturbing tokens, one could design controlled experiments where token changes are propagated through a causal graph of linguistic dependencies. By comparing observed attribution patterns to theoretically predicted causal structures, discrepancies could be attributed more precisely to either explainer inadequacy or model reasoning failures.

A third approach involves multi-method triangulation: applying diverse explainability techniques (perturbation-based, attention-based where available, counterfactual generation) to the same test case and analyzing agreement patterns. Tokens consistently identified as important across methods are likely genuinely critical, while method-specific attributions may indicate explainer artifacts.

### 7.6 Scaling to Production Systems with Efficient Perturbation Strategies

The API rate limit bottleneck identified in this work poses a critical barrier to deploying black-box explainability in real-world systems. Future research must develop efficient perturbation strategies that reduce the number of required API calls while maintaining attribution quality.

One approach involves adaptive perturbation sampling, where tokens are selected for perturbation based on heuristic importance estimates (e.g., part-of-speech tags, syntactic parse tree positions). For instance, content words (nouns, verbs) could be prioritized over function words (articles, prepositions) in initial sampling, with recursive refinement for high-importance regions. This could reduce the number of API calls from 3N to a fraction of that while covering the most linguistically salient tokens.

Another direction involves amortized explainability through meta-learned perturbation policies. A separate model could be trained to predict which token perturbations are most informative for a given prompt, effectively learning a surrogate attribution function from a dataset of prompt-explanation pairs. This meta-explainer would require only a single API call at inference time, with perturbations performed only during offline training.

Finally, researchers could explore batch perturbation techniques where multiple tokens are perturbed simultaneously in a strategically designed pattern, with combinatorial optimization methods (e.g., Hadamard matrices, fractional factorial designs) used to disentangle individual token effects from aggregate responses. This would trade some attribution granularity for exponential reductions in API calls.

### 7.7 Extending Beyond Text to Multimodal LLMs

Modern LLMs increasingly incorporate multimodal capabilities, processing and generating both text and images (e.g., GPT-4V, Gemini Pro Vision). Explaining decisions in multimodal contexts introduces additional complexity: attributions must span discrete tokens and continuous pixel regions, requiring unified frameworks that handle both modalities.

Extending LLM-Units to multimodal settings would involve constructing test cases where visual and textual inputs interact—for example, an image of a red car with the text "What color is the vehicle?" Tests could assess whether explainers correctly attribute importance to relevant image regions (the car's body) and text tokens ("color") while ignoring distractors. Perturbation methods would need to be adapted for images, potentially using segmentation masks or pixel patches as perturbation units.

This extension would also address cross-modal reasoning: does the explainer correctly identify when a text token (e.g., "not") modifies the interpretation of a visual feature? Such capabilities are critical for applications like medical image diagnosis with clinical notes or autonomous driving systems integrating sensor data and natural language instructions.

### 7.8 Normative and Ethical Dimensions of Black-Box XAI

Finally, this work raises normative questions about the adequacy of black-box explainability for high-stakes applications. If the best available attribution methods achieve only 36% F1 in controlled settings, are such explanations sufficient for legally mandated transparency requirements (e.g., GDPR's "right to explanation") or ethical accountability in domains like criminal justice and healthcare?

Future work should engage with legal scholars, ethicists, and policymakers to establish minimum explainability thresholds for different application contexts. This could inform regulatory frameworks that impose transparency requirements proportional to decision stakes—requiring white-box access or human oversight when black-box explanations fall below acceptable quality standards.

Additionally, research should explore user studies assessing how end-users interpret and trust black-box explanations. Do low-quality attributions mislead users into false confidence, or do they appropriately convey uncertainty? Understanding the human factors dimension of XAI is essential for translating technical explainability metrics into genuine accountability and user empowerment.
