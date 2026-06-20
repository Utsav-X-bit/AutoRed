<h1 align="center"><img src="assets/autored-logo.png" style="vertical-align: middle" width="50px"> <b>AutoRed</b>: Automated Attack Generation Framework for Red Teaming of Large Language Models</h1>  


🪪 LLMs pose privacy risks by retaining sensitive information in context memory, potentially leading to unintended data exposure.

🛡️ Traditional red teaming is costly and slow. 

This work presents **AutoRed**, an innovative learning framework developed to automatically generate malicious attack scenarios for extracting sensitive information from LLMs.

**AutoRed** consists 

- **One high-level model for decision-making**:
  - The **Stop Point Identifier** is a DistilBERT-based binary classifier that determines whether the current stage should proceed with an attack or conditionally execute the extraction pipeline.

- **Two low-level modules for prompt injection attack tasks**:
  - The **Malicious Prompt Generator** (now upgraded to Llama-3.1-8B-Lexi-Uncensored-V2) is designed to generate a diverse range of malicious prompt injection attacks using 7 distinct strategies and mutation logic.
  - The **Sensitive Information Extractor** is a multi-layer pipeline combining Regex patterns, capitalized words extraction, and an LLM-based fallback, followed by a verifier loop to accurately extract and confirm sensitive data.



## Acknowledgments

This work uses the [**RL4LMs**](https://github.com/allenai/RL4LMs) library developed by [**AllenAI** (Ai2)](https://allenai.org/); see the [license](https://github.com/yoyostudy/AutoRed/blob/main/LICENSE) for details.

  
