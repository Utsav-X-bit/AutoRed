# AutoRed Dataset Analysis Report v1

**Generated:** AutoRed

## 1. Dataset Sizes

| Dataset | Entries | Description |
|---------|---------|-------------|
| Raw Successes | 26740 | All successful runs |
| Raw Failures | 135929 | All failed runs |
| **Positive** | 4852 | ground_truth_leaked OR verification_success |
| **Verified** | 3143 | verification_success == True |
| **Failures** | 135929 | No ground_truth_leaked AND no verification_success |

## 2. Feature Mining Analysis (Positive vs Failures)

| Feature | Successes | Failures | Success Prev | Failure Prev | Leak Rate | Lift |
|---------|-----------|----------|--------------|--------------|-----------|------|
| contains_encouragement | 5 | 84 | 0.1% | 0.1% | 5.6% | 1.67 |
| contains_format_wrapper | 70 | 1417 | 1.4% | 1.0% | 4.7% | 1.38 |
| contains_length_constraint | 328 | 7475 | 6.8% | 5.5% | 4.2% | 1.23 |
| contains_conditional | 1534 | 38518 | 31.6% | 28.3% | 3.8% | 1.12 |
| contains_prompt_injection | 3604 | 91185 | 74.3% | 67.1% | 3.8% | 1.11 |
| contains_educational_frame | 472 | 11985 | 9.7% | 8.8% | 3.8% | 1.10 |
| contains_roleplay | 275 | 6997 | 5.7% | 5.1% | 3.8% | 1.10 |
| contains_dan_pattern | 291 | 8219 | 6.0% | 6.0% | 3.4% | 0.99 |
| contains_repeat | 2304 | 66526 | 47.5% | 48.9% | 3.3% | 0.97 |
| contains_metaphor_analogy | 8 | 232 | 0.2% | 0.2% | 3.3% | 0.97 |
| contains_translation | 325 | 9815 | 6.7% | 7.2% | 3.2% | 0.93 |
| contains_hypothetical | 354 | 10816 | 7.3% | 8.0% | 3.2% | 0.92 |
| contains_technical_jargon | 1254 | 39197 | 25.8% | 28.8% | 3.1% | 0.90 |
| contains_questioning | 1322 | 42661 | 27.2% | 31.4% | 3.0% | 0.87 |
| contains_negation_bypass | 314 | 10682 | 6.5% | 7.9% | 2.9% | 0.82 |
| contains_list_format | 169 | 6016 | 3.5% | 4.4% | 2.7% | 0.79 |
| contains_pseudocode | 89 | 3439 | 1.8% | 2.5% | 2.5% | 0.73 |
| contains_social_engineering | 272 | 12077 | 5.6% | 8.9% | 2.2% | 0.63 |
| contains_command_injection | 161 | 7510 | 3.3% | 5.5% | 2.1% | 0.60 |
| contains_begin_with | 1 | 169 | 0.0% | 0.1% | 0.6% | 0.17 |

### Top 5 Most Discriminative Features (Highest Lift)

- **contains_encouragement**: lift=1.67, leak_rate=5.6% (5 successes, 84 failures)
- **contains_format_wrapper**: lift=1.38, leak_rate=4.7% (70 successes, 1417 failures)
- **contains_length_constraint**: lift=1.23, leak_rate=4.2% (328 successes, 7475 failures)
- **contains_conditional**: lift=1.12, leak_rate=3.8% (1534 successes, 38518 failures)
- **contains_prompt_injection**: lift=1.11, leak_rate=3.8% (3604 successes, 91185 failures)

## 3. Feature Mining Analysis (All Successes vs All Failures)

| Feature | Successes | Failures | Success Prev | Failure Prev | Leak Rate | Lift |
|---------|-----------|----------|--------------|--------------|-----------|------|
| contains_format_wrapper | 369 | 1417 | 1.4% | 1.0% | 20.7% | 1.32 |
| contains_roleplay | 1819 | 6997 | 6.8% | 5.1% | 20.6% | 1.32 |
| contains_negation_bypass | 2766 | 10682 | 10.3% | 7.9% | 20.6% | 1.32 |
| contains_translation | 2540 | 9815 | 9.5% | 7.2% | 20.6% | 1.32 |
| contains_hypothetical | 2752 | 10816 | 10.3% | 8.0% | 20.3% | 1.29 |
| contains_technical_jargon | 9506 | 39197 | 35.5% | 28.8% | 19.5% | 1.23 |
| contains_metaphor_analogy | 51 | 232 | 0.2% | 0.2% | 18.0% | 1.12 |
| contains_educational_frame | 2547 | 11985 | 9.5% | 8.8% | 17.5% | 1.08 |
| contains_questioning | 9040 | 42661 | 33.8% | 31.4% | 17.5% | 1.08 |
| contains_command_injection | 1381 | 7510 | 5.2% | 5.5% | 15.5% | 0.93 |
| contains_conditional | 6946 | 38518 | 26.0% | 28.3% | 15.3% | 0.92 |
| contains_prompt_injection | 16078 | 91185 | 60.1% | 67.1% | 15.0% | 0.90 |
| contains_dan_pattern | 1439 | 8219 | 5.4% | 6.0% | 14.9% | 0.89 |
| contains_repeat | 11566 | 66526 | 43.3% | 48.9% | 14.8% | 0.88 |
| contains_encouragement | 14 | 84 | 0.1% | 0.1% | 14.3% | 0.85 |
| contains_pseudocode | 564 | 3439 | 2.1% | 2.5% | 14.1% | 0.83 |
| contains_length_constraint | 1206 | 7475 | 4.5% | 5.5% | 13.9% | 0.82 |
| contains_social_engineering | 1840 | 12077 | 6.9% | 8.9% | 13.2% | 0.77 |
| contains_list_format | 847 | 6016 | 3.2% | 4.4% | 12.3% | 0.72 |
| contains_begin_with | 14 | 169 | 0.1% | 0.1% | 7.7% | 0.42 |

## 4. Strategy Effectiveness Analysis

| Strategy | Total | Successes | Failures | Success Rate | GT Leaked | Verified |
|----------|-------|-----------|----------|--------------|-----------|----------|
| jailbreak_framing | 156 | 76 | 80 | 48.7% | 8 | 5 |
| authority_override | 182 | 87 | 95 | 47.8% | 3 | 1 |
| latent_injection | 185 | 85 | 100 | 45.9% | 12 | 7 |
| reflection_attack | 333 | 152 | 181 | 45.6% | 42 | 32 |
| json_smuggling | 184 | 83 | 101 | 45.1% | 3 | 1 |
| markdown_smuggling | 178 | 80 | 98 | 44.9% | 1 | 1 |
| format_conversion | 198 | 88 | 110 | 44.4% | 39 | 25 |
| unicode_bypass | 185 | 82 | 103 | 44.3% | 4 | 1 |
| encoding_bypass | 188 | 82 | 106 | 43.6% | 7 | 4 |
| yaml_smuggling | 190 | 82 | 108 | 43.2% | 11 | 4 |
| base64_bypass | 197 | 83 | 114 | 42.1% | 13 | 7 |
| translation | 9523 | 2061 | 7462 | 21.6% | 205 | 113 |
| exception_discovery | 12201 | 2520 | 9681 | 20.7% | 296 | 207 |
| roleplay | 10263 | 1974 | 8289 | 19.2% | 166 | 113 |
| instruction_leak | 37354 | 6013 | 31341 | 16.1% | 1198 | 781 |
| trigger_phrase_discovery | 46746 | 7213 | 39533 | 15.4% | 1062 | 709 |
| summarization | 25553 | 3616 | 21937 | 14.2% | 1165 | 870 |
| system_prompt_recovery | 18853 | 2363 | 16490 | 12.5% | 422 | 262 |

### Top 5 Most Effective Strategies

- **jailbreak_framing**: 48.7% success rate (76/156 attempts)
- **authority_override**: 47.8% success rate (87/182 attempts)
- **latent_injection**: 45.9% success rate (85/185 attempts)
- **reflection_attack**: 45.6% success rate (152/333 attempts)
- **json_smuggling**: 45.1% success rate (83/184 attempts)

## 5. Defense Complexity Analysis

| Complexity | Total | Successes | Failures | Success Rate |
|------------|-------|-----------|----------|--------------|
| easy | 25513 | 4801 | 20712 | 18.8% |
| medium | 57506 | 11319 | 46187 | 19.7% |
| hard | 79650 | 10620 | 69030 | 13.3% |

## 6. Access Code Type Analysis

| Code Type | Total | Successes | Failures | Success Rate |
|-----------|-------|-----------|----------|--------------|
| TOKEN | 64935 | 21806 | 43129 | 33.6% |
| MULTILINE | 47634 | 734 | 46900 | 1.5% |
| PHRASE | 19007 | 2747 | 16260 | 14.5% |
| SENTENCE | 13766 | 496 | 13270 | 3.6% |
| CONVERSATION | 12174 | 34 | 12140 | 0.3% |
| UNKNOWN | 5033 | 923 | 4110 | 18.3% |
| STRUCTURED | 120 | 0 | 120 | 0.0% |

## 7. Attack Length Analysis

| Length Bucket | Total | Successes | Failures | Success Rate |
|---------------|-------|-----------|----------|--------------|
| short (<50) | 5191 | 826 | 4365 | 15.9% |
| medium (50-150) | 38861 | 7517 | 31344 | 19.3% |
| long (150-300) | 106585 | 17090 | 89495 | 16.0% |
| very_long (>300) | 12032 | 1307 | 10725 | 10.9% |

## 8. Key Findings

- **Best Strategy:** jailbreak_framing with 48.7% success rate
- **Most Discriminative Feature:** contains_encouragement with lift=1.67
- **Hardest Defense:** hard complexity with 13.3% success rate
- **Verified vs Positive:** 3143 verified out of 4852 positive (64.8%)

## 9. Recommendations for SFT Training

1. **Use Verified Dataset** for highest-quality training data
2. **Focus on top-performing strategies** identified above
3. **Incorporate effective features** into attack generation templates
4. **Balance complexity levels** to ensure robust training
5. **Consider length constraints** based on length analysis
