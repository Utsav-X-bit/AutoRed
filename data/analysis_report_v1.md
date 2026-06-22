# AutoRed Dataset Analysis Report v1

**Generated:** AutoRed

## 1. Dataset Sizes

| Dataset | Entries | Description |
|---------|---------|-------------|
| Raw Successes | 16865 | All successful runs |
| Raw Failures | 80137 | All failed runs |
| **Positive** | 2723 | ground_truth_leaked OR verification_success |
| **Verified** | 1570 | verification_success == True |
| **Failures** | 80137 | No ground_truth_leaked AND no verification_success |

## 2. Feature Mining Analysis (Positive vs Failures)

| Feature | Successes | Failures | Success Prev | Failure Prev | Leak Rate | Lift |
|---------|-----------|----------|--------------|--------------|-----------|------|
| contains_length_constraint | 163 | 3559 | 6.0% | 4.4% | 4.4% | 1.35 |
| contains_roleplay | 191 | 5041 | 7.0% | 6.3% | 3.7% | 1.12 |
| contains_prompt_injection | 2023 | 55496 | 74.3% | 69.3% | 3.5% | 1.07 |
| contains_conditional | 814 | 22601 | 29.9% | 28.2% | 3.5% | 1.06 |
| contains_educational_frame | 229 | 6491 | 8.4% | 8.1% | 3.4% | 1.04 |
| contains_repeat | 1469 | 43921 | 53.9% | 54.8% | 3.2% | 0.98 |
| contains_dan_pattern | 150 | 4606 | 5.5% | 5.7% | 3.2% | 0.96 |
| contains_hypothetical | 256 | 7904 | 9.4% | 9.9% | 3.1% | 0.95 |
| contains_format_wrapper | 9 | 282 | 0.3% | 0.4% | 3.1% | 0.94 |
| contains_metaphor_analogy | 2 | 65 | 0.1% | 0.1% | 3.0% | 0.91 |
| contains_translation | 226 | 7753 | 8.3% | 9.7% | 2.8% | 0.86 |
| contains_technical_jargon | 932 | 32741 | 34.2% | 40.9% | 2.8% | 0.84 |
| contains_questioning | 709 | 25712 | 26.0% | 32.1% | 2.7% | 0.81 |
| contains_negation_bypass | 246 | 9592 | 9.0% | 12.0% | 2.5% | 0.75 |
| contains_list_format | 76 | 3415 | 2.8% | 4.3% | 2.2% | 0.65 |
| contains_pseudocode | 32 | 1691 | 1.2% | 2.1% | 1.9% | 0.56 |
| contains_command_injection | 75 | 3983 | 2.8% | 5.0% | 1.8% | 0.55 |
| contains_social_engineering | 86 | 6887 | 3.2% | 8.6% | 1.2% | 0.37 |
| contains_begin_with | 0 | 11 | 0.0% | 0.0% | 0.0% | 0.00 |
| contains_encouragement | 0 | 50 | 0.0% | 0.1% | 0.0% | 0.00 |

### Top 5 Most Discriminative Features (Highest Lift)

- **contains_length_constraint**: lift=1.35, leak_rate=4.4% (163 successes, 3559 failures)
- **contains_roleplay**: lift=1.12, leak_rate=3.7% (191 successes, 5041 failures)
- **contains_prompt_injection**: lift=1.07, leak_rate=3.5% (2023 successes, 55496 failures)
- **contains_conditional**: lift=1.06, leak_rate=3.5% (814 successes, 22601 failures)
- **contains_educational_frame**: lift=1.04, leak_rate=3.4% (229 successes, 6491 failures)

## 3. Feature Mining Analysis (All Successes vs All Failures)

| Feature | Successes | Failures | Success Prev | Failure Prev | Leak Rate | Lift |
|---------|-----------|----------|--------------|--------------|-----------|------|
| contains_metaphor_analogy | 25 | 65 | 0.1% | 0.1% | 27.8% | 1.83 |
| contains_hypothetical | 2091 | 7904 | 12.4% | 9.9% | 20.9% | 1.26 |
| contains_roleplay | 1325 | 5041 | 7.9% | 6.3% | 20.8% | 1.25 |
| contains_negation_bypass | 2382 | 9592 | 14.1% | 12.0% | 19.9% | 1.18 |
| contains_translation | 1902 | 7753 | 11.3% | 9.7% | 19.7% | 1.17 |
| contains_technical_jargon | 7678 | 32741 | 45.5% | 40.9% | 19.0% | 1.11 |
| contains_educational_frame | 1521 | 6491 | 9.0% | 8.1% | 19.0% | 1.11 |
| contains_questioning | 5701 | 25712 | 33.8% | 32.1% | 18.1% | 1.05 |
| contains_format_wrapper | 60 | 282 | 0.4% | 0.4% | 17.5% | 1.01 |
| contains_conditional | 4710 | 22601 | 27.9% | 28.2% | 17.2% | 0.99 |
| contains_prompt_injection | 10508 | 55496 | 62.3% | 69.3% | 15.9% | 0.90 |
| contains_repeat | 8194 | 43921 | 48.6% | 54.8% | 15.7% | 0.89 |
| contains_dan_pattern | 846 | 4606 | 5.0% | 5.7% | 15.5% | 0.87 |
| contains_begin_with | 2 | 11 | 0.0% | 0.0% | 15.4% | 0.86 |
| contains_length_constraint | 632 | 3559 | 3.7% | 4.4% | 15.1% | 0.84 |
| contains_command_injection | 699 | 3983 | 4.1% | 5.0% | 14.9% | 0.83 |
| contains_list_format | 482 | 3415 | 2.9% | 4.3% | 12.4% | 0.67 |
| contains_social_engineering | 812 | 6887 | 4.8% | 8.6% | 10.5% | 0.56 |
| contains_pseudocode | 181 | 1691 | 1.1% | 2.1% | 9.7% | 0.51 |
| contains_encouragement | 4 | 50 | 0.0% | 0.1% | 7.4% | 0.38 |

## 4. Strategy Effectiveness Analysis

| Strategy | Total | Successes | Failures | Success Rate | GT Leaked | Verified |
|----------|-------|-----------|----------|--------------|-----------|----------|
| exception_discovery | 11099 | 2233 | 8866 | 20.1% | 244 | 162 |
| translation | 8076 | 1601 | 6475 | 19.8% | 153 | 71 |
| roleplay | 9181 | 1686 | 7495 | 18.4% | 148 | 103 |
| instruction_leak | 21550 | 3753 | 17797 | 17.4% | 712 | 429 |
| system_prompt_recovery | 4707 | 792 | 3915 | 16.8% | 153 | 81 |
| trigger_phrase_discovery | 32087 | 5295 | 26792 | 16.5% | 712 | 461 |
| summarization | 10302 | 1505 | 8797 | 14.6% | 405 | 263 |

### Top 5 Most Effective Strategies

- **exception_discovery**: 20.1% success rate (2233/11099 attempts)
- **translation**: 19.8% success rate (1601/8076 attempts)
- **roleplay**: 18.4% success rate (1686/9181 attempts)
- **instruction_leak**: 17.4% success rate (3753/21550 attempts)
- **system_prompt_recovery**: 16.8% success rate (792/4707 attempts)

## 5. Defense Complexity Analysis

| Complexity | Total | Successes | Failures | Success Rate |
|------------|-------|-----------|----------|--------------|
| easy | 15955 | 3245 | 12710 | 20.3% |
| medium | 33106 | 7499 | 25607 | 22.7% |
| hard | 47941 | 6121 | 41820 | 12.8% |

## 6. Access Code Type Analysis

| Code Type | Total | Successes | Failures | Success Rate |
|-----------|-------|-----------|----------|--------------|
| TOKEN | 41267 | 13730 | 27537 | 33.3% |
| MULTILINE | 27067 | 587 | 26480 | 2.2% |
| PHRASE | 11532 | 1692 | 9840 | 14.7% |
| SENTENCE | 7414 | 284 | 7130 | 3.8% |
| CONVERSATION | 6514 | 34 | 6480 | 0.5% |
| UNKNOWN | 3208 | 538 | 2670 | 16.8% |

## 7. Attack Length Analysis

| Length Bucket | Total | Successes | Failures | Success Rate |
|---------------|-------|-----------|----------|--------------|
| short (<50) | 110 | 1 | 109 | 0.9% |
| medium (50-150) | 18319 | 4057 | 14262 | 22.1% |
| long (150-300) | 71204 | 12096 | 59108 | 17.0% |
| very_long (>300) | 7369 | 711 | 6658 | 9.6% |

## 8. Key Findings

- **Best Strategy:** exception_discovery with 20.1% success rate
- **Most Discriminative Feature:** contains_length_constraint with lift=1.35
- **Hardest Defense:** hard complexity with 12.8% success rate
- **Verified vs Positive:** 1570 verified out of 2723 positive (57.7%)

## 9. Recommendations for SFT Training

1. **Use Verified Dataset** for highest-quality training data
2. **Focus on top-performing strategies** identified above
3. **Incorporate effective features** into attack generation templates
4. **Balance complexity levels** to ensure robust training
5. **Consider length constraints** based on length analysis
