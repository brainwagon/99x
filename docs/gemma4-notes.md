Gemma 4's E2B and E4B (Edge 2 Billion and 4 Billion effective parameter) models are excellent choices for local, on-device agentic workflows. They natively support function calling and feature a configurable "thinking" mode that allows the model to reason step-by-step before acting.

Here is how you can implement and control these features in your harness, along with the best resources to guide your development.

## Implementing Thinking Mode

Gemma 4 introduces explicit control over its reasoning process. To integrate this into your harness, you need to manage specific control tokens and parse the output correctly:

* **Enable Thinking:** Add the `<|think|>` token at the very beginning of your system prompt. To disable it, simply omit the token.
* **Parse the Output:** When thinking is enabled, the model will output its internal reasoning before the final answer, wrapped in specific channel tags. Your harness should be programmed to parse and hide (or log) this section:
```text
<|channel>thought
[Internal reasoning]
<channel|>
[Final answer]

```


* **Prune Conversation History:** In multi-turn agentic loops, you must strip the thinking content from the historical model output. Only the final response should be fed back into the context window for the next turn, otherwise the model's reasoning loop may degrade.

## Optimizing Tool Calling

If you are experiencing failing tool calls or poor instruction following, the issue is almost always the chat template.

* **Update the Chat Template:** Older Gemma templates will break tool calling in Gemma 4. Gemma 4 now uses standard `system`, `assistant`, and `user` roles. Ensure your harness (whether you are using Transformers, llama.cpp, or another backend) is applying the new native system prompt template correctly.
* **Schema Complexity:** While both models support function calling, the E4B model is significantly more reliable when dealing with complex tool schemas (10+ functions), overlapping intents, or ambiguous inputs. If your agent requires complex reasoning to select the right tool, default to E4B.

## Official Resources and Documentation

To get the exact chat templates, configuration parameters, and code snippets for your specific backend, you should consult these primary resources:

1. **Google AI for Developers (Gemma 4 Docs):** The official documentation provides comprehensive architectural overviews, memory planning guides, and the exact control tokens needed for the edge models.
2. **Hugging Face Model Cards (`google/gemma-4-E2B` and `E4B`):** The model cards contain ready-to-use Python snippets for the `Transformers` library, including how to structure the `messages` array for complex prompts and multimodal tool calling.
3. **Ollama & Unsloth Documentation:** If your harness relies on local inference engines like Ollama or llama.cpp, their respective Gemma 4 documentation pages provide the exact CLI arguments (like `--chat-template-kwargs '{"enable_thinking":false}'`) and pre-configured templates to make tool calling work out of the box.

---

[Gemma 4: Free AI Agent in 5 Minutes](https://www.youtube.com/watch?v=luynuetQxV4)
This video provides a quick five-minute walkthrough on getting a local Gemma 4 agent running, which can help you see the tool calling and agentic capabilities in action.
