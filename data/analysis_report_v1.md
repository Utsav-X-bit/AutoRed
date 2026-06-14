# AutoRed Dataset Analysis Report v1

**Generated:** AutoRed

## 1. Dataset Sizes

| Dataset | Entries | Description |
|---------|---------|-------------|
| Raw Successes | 1947 | All successful runs |
| Raw Failures | 4330 | All failed runs |
| **Positive** | 291 | ground_truth_leaked OR verification_success |
| **Verified** | 138 | verification_success == True |
| **Failures** | 4330 | No ground_truth_leaked AND no verification_success |

## 2. Feature Mining Analysis (Positive vs Failures)

| Feature | Successes | Failures | Success Prev | Failure Prev | Leak Rate | Lift |
|---------|-----------|----------|--------------|--------------|-----------|------|
| contains_metaphor_analogy | 1 | 6 | 0.3% | 0.1% | 14.3% | 2.48 |
| contains_educational_frame | 20 | 191 | 6.9% | 4.4% | 9.5% | 1.56 |
| contains_repeat | 141 | 1579 | 48.5% | 36.5% | 8.2% | 1.33 |
| contains_roleplay | 29 | 349 | 10.0% | 8.1% | 7.7% | 1.24 |
| contains_conditional | 94 | 1200 | 32.3% | 27.7% | 7.3% | 1.17 |
| contains_questioning | 96 | 1259 | 33.0% | 29.1% | 7.1% | 1.13 |
| contains_list_format | 9 | 124 | 3.1% | 2.9% | 6.8% | 1.08 |
| contains_prompt_injection | 205 | 2870 | 70.4% | 66.3% | 6.7% | 1.06 |
| contains_length_constraint | 22 | 315 | 7.6% | 7.3% | 6.5% | 1.04 |
| contains_negation_bypass | 22 | 329 | 7.6% | 7.6% | 6.3% | 0.99 |
| contains_technical_jargon | 91 | 1369 | 31.3% | 31.6% | 6.2% | 0.99 |
| contains_pseudocode | 6 | 92 | 2.1% | 2.1% | 6.1% | 0.97 |
| contains_dan_pattern | 14 | 225 | 4.8% | 5.2% | 5.9% | 0.93 |
| contains_command_injection | 7 | 126 | 2.4% | 2.9% | 5.3% | 0.83 |
| contains_hypothetical | 26 | 477 | 8.9% | 11.0% | 5.2% | 0.81 |
| contains_social_engineering | 13 | 301 | 4.5% | 7.0% | 4.1% | 0.64 |
| contains_translation | 13 | 469 | 4.5% | 10.8% | 2.7% | 0.41 |
| contains_format_wrapper | 0 | 17 | 0.0% | 0.4% | 0.0% | 0.00 |
| contains_encouragement | 0 | 9 | 0.0% | 0.2% | 0.0% | 0.00 |

### Top 5 Most Discriminative Features (Highest Lift)

- **contains_metaphor_analogy**: lift=2.48, leak_rate=14.3% (1 successes, 6 failures)
- **contains_educational_frame**: lift=1.56, leak_rate=9.5% (20 successes, 191 failures)
- **contains_repeat**: lift=1.33, leak_rate=8.2% (141 successes, 1579 failures)
- **contains_roleplay**: lift=1.24, leak_rate=7.7% (29 successes, 349 failures)
- **contains_conditional**: lift=1.17, leak_rate=7.3% (94 successes, 1200 failures)

## 3. Feature Mining Analysis (All Successes vs All Failures)

| Feature | Successes | Failures | Success Prev | Failure Prev | Leak Rate | Lift |
|---------|-----------|----------|--------------|--------------|-----------|------|
| contains_begin_with | 1 | 0 | 0.1% | 0.0% | 100.0% | ∞ |
| contains_educational_frame | 171 | 191 | 8.8% | 4.4% | 47.2% | 1.99 |
| contains_metaphor_analogy | 5 | 6 | 0.3% | 0.1% | 45.5% | 1.85 |
| contains_negation_bypass | 262 | 329 | 13.5% | 7.6% | 44.3% | 1.77 |
| contains_command_injection | 97 | 126 | 5.0% | 2.9% | 43.5% | 1.71 |
| contains_technical_jargon | 841 | 1369 | 43.2% | 31.6% | 38.1% | 1.37 |
| contains_questioning | 737 | 1259 | 37.9% | 29.1% | 36.9% | 1.30 |
| contains_hypothetical | 259 | 477 | 13.3% | 11.0% | 35.2% | 1.21 |
| contains_repeat | 849 | 1579 | 43.6% | 36.5% | 35.0% | 1.20 |
| contains_roleplay | 172 | 349 | 8.8% | 8.1% | 33.0% | 1.10 |
| contains_conditional | 575 | 1200 | 29.5% | 27.7% | 32.4% | 1.07 |
| contains_dan_pattern | 95 | 225 | 4.9% | 5.2% | 29.7% | 0.94 |
| contains_prompt_injection | 1092 | 2870 | 56.1% | 66.3% | 27.6% | 0.85 |
| contains_translation | 161 | 469 | 8.3% | 10.8% | 25.6% | 0.76 |
| contains_social_engineering | 101 | 301 | 5.2% | 7.0% | 25.1% | 0.75 |
| contains_list_format | 38 | 124 | 2.0% | 2.9% | 23.5% | 0.68 |
| contains_pseudocode | 26 | 92 | 1.3% | 2.1% | 22.0% | 0.63 |
| contains_length_constraint | 87 | 315 | 4.5% | 7.3% | 21.6% | 0.61 |
| contains_encouragement | 2 | 9 | 0.1% | 0.2% | 18.2% | 0.49 |
| contains_format_wrapper | 1 | 17 | 0.1% | 0.4% | 5.6% | 0.13 |

## 4. Strategy Effectiveness Analysis

| Strategy | Total | Successes | Failures | Success Rate | GT Leaked | Verified |
|----------|-------|-----------|----------|--------------|-----------|----------|
| exception_discovery | 775 | 308 | 467 | 39.7% | 33 | 12 |
| instruction_leak | 1073 | 401 | 672 | 37.4% | 50 | 30 |
| trigger_phrase_discovery | 1523 | 531 | 992 | 34.9% | 39 | 41 |
| roleplay | 596 | 196 | 400 | 32.9% | 10 | 15 |
| translation | 627 | 179 | 448 | 28.5% | 8 | 7 |
| system_prompt_recovery | 625 | 125 | 500 | 20.0% | 19 | 6 |
| summarization | 1058 | 207 | 851 | 19.6% | 44 | 27 |

### Top 5 Most Effective Strategies

- **exception_discovery**: 39.7% success rate (308/775 attempts)
- **instruction_leak**: 37.4% success rate (401/1073 attempts)
- **trigger_phrase_discovery**: 34.9% success rate (531/1523 attempts)
- **roleplay**: 32.9% success rate (196/596 attempts)
- **translation**: 28.5% success rate (179/627 attempts)

## 5. Defense Complexity Analysis

| Complexity | Total | Successes | Failures | Success Rate |
|------------|-------|-----------|----------|--------------|
| easy | 916 | 336 | 580 | 36.7% |
| medium | 2447 | 947 | 1500 | 38.7% |
| hard | 2914 | 664 | 2250 | 22.8% |

## 6. Access Code Type Analysis

| Code Type | Total | Successes | Failures | Success Rate |
|-----------|-------|-----------|----------|--------------|
| TOKEN | 2805 | 1375 | 1430 | 49.0% |
| MULTILINE | 1527 | 207 | 1320 | 13.6% |
| PHRASE | 680 | 200 | 480 | 29.4% |
| SENTENCE | 522 | 102 | 420 | 19.5% |
| CONVERSATION | 417 | 17 | 400 | 4.1% |
| UNKNOWN | 326 | 46 | 280 | 14.1% |

## 7. Attack Length Analysis

| Length Bucket | Total | Successes | Failures | Success Rate |
|---------------|-------|-----------|----------|--------------|
| short (<50) | 11 | 0 | 11 | 0.0% |
| medium (50-150) | 1799 | 581 | 1218 | 32.3% |
| long (150-300) | 4341 | 1344 | 2997 | 31.0% |
| very_long (>300) | 126 | 22 | 104 | 17.5% |

## 8. Key Findings

- **Best Strategy:** exception_discovery with 39.7% success rate
- **Most Discriminative Feature:** contains_metaphor_analogy with lift=2.48
- **Hardest Defense:** hard complexity with 22.8% success rate
- **Verified vs Positive:** 138 verified out of 291 positive (47.4%)

## 9. Recommendations for SFT Training

1. **Use Verified Dataset** for highest-quality training data
2. **Focus on top-performing strategies** identified above
3. **Incorporate effective features** into attack generation templates
4. **Balance complexity levels** to ensure robust training
5. **Consider length constraints** based on length analysis
