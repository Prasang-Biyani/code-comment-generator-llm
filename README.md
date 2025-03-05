# Personalized Code Comment Generator

A Generative AI tool that leverages Large Language Models (LLMs) to automatically generate meaningful, context-aware code comments tailored to your coding style. Built with Hugging Face Transformers and fine-tuned on real-world code-comment pairs, this project aims to improve code readability and save developers time.

---

## Features
- **Context-Aware Comments**: Generates comments based on code syntax, logic, and variable names.
- **Personalization**: Adapts to user preferences (e.g., concise or beginner-friendly comments).
- **Scalable**: Fine-tuned on open-source datasets with the option to add your own codebases.
- **Powered by LLMs**: Uses Transformer-based models like CodeBERT for high-quality generation.

### Example
**Input Code:**
```python
def calc_avg(numbers):
    total = sum(numbers)
    return total / len(numbers)
```
**Output Code:**
```python
# Computes the average of a list of numbers.
def calc_avg(numbers):
    total = sum(numbers)
    return total / len(numbers)
```
