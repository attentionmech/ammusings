# Random walks by LLMs and weird behaviour of gemma2:9b model

This is a simple experiment of asking LLMs do a random walk. The test was done with open source llama3.1/2 and gemma2 series. My general expectation was that as temperature will grow the random walk will keep growing more. But somehow the gemma2:9b model is behaving weirdly. That is what I am investigating. But nonetheless it's cool to look at LLMs visually, and not just in loss graphs / tokens. The table below has graphs arranged in matrix of temperature and model.

The setup is very simple - We give the LLM details about the experiment and ask it to do a random walk on the grid. Right now we don't pass it previous context but just that what is the time T. The LLM is asked to reply with either of the four directions.


# What's weird

The weird behaviour is that gemma2:9b is just not considering the UP, DOWN dimensions despite being asked to; whereas it was trivial with other LLMS. And it is consistently doing this across all temperature values whereas other LLMs smaller than it are doing much different

# Test setup

1. Ollama with LiteLLM
2. Mac M2/ 16 GB Ram
3. Context is not continued and every interaction is new per turn per walk
4. Walks with the same configuration of model+temperature are color coded (for sims did 5)


# LLM interaction

```python
def random_walk_step_llm(t, current_position, model_name, temperature, grid_size):
    """Updated to include grid size information in the prompt"""
    answer = llm_unstructured_query(
        f"You are a random walker in a {grid_size}x{grid_size} grid centered at (0,0). "
        f"At t=0 you started at the center (0,0). Currently at t={t}, your position is {current_position}. "
        f"Reply with either UP, DOWN, LEFT, or RIGHT to move in that direction. "
        f"If you could not comply to prompt, you will stay at the same place.",
        model=model_name,
        temperature=temperature
    )

    if "UP" in answer:
        dx, dy = 0, 1
    elif "DOWN" in answer:
        dx, dy = 0, -1
    elif "LEFT" in answer:
        dx, dy = -1, 0
    elif "RIGHT" in answer:
        dx, dy = 1, 0
    else:
        dx, dy = 0, 0

    return dx, dy
```

So setup is pretty basic, just tell LLM about what is going on and ask it to choose a direction. Ideally if the LLM knows about random walks and have been trained on a shit tonn of data - it would be piece of cake for it to simulate one. And most LLMs do it. There are two question though-

1. Even with a explicit instruction why can't a LLM produce a random walk at temperature = 0 => This is most likey due to context not being passed around and LLMs especially smaller ones not able to do the work just based on time and position argument itself. Which behaviour is better though?
3. What the f is happening with gemma2:9b

llama3:8b also seems to be doing the most `aesthetic/randomlike walks` . Nonetheless enjoy the visuals for others in the following tables:

## Table 1: Temperatures 0.0, 0.1, 0.3, 0.5

| Model                  | 0.0                                     | 0.1                                     | 0.3                                     | 0.5                                     |
|------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|
| `ollama/llama3.2:1b`   | <img src="assets/images/random_walk_ollama_llama3.2:1b_temp_0.0.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.2:1b_temp_0.1.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.2:1b_temp_0.3.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.2:1b_temp_0.5.png" width="300"> |
| `ollama/llama3.1:8b`   | <img src="assets/images/random_walk_ollama_llama3.1:8b_temp_0.0.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.1:8b_temp_0.1.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.1:8b_temp_0.3.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.1:8b_temp_0.5.png" width="300"> |
| `ollama/llama3.2:3b`   | <img src="assets/images/random_walk_ollama_llama3.2:3b_temp_0.0.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.2:3b_temp_0.1.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.2:3b_temp_0.3.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.2:3b_temp_0.5.png" width="300"> |
| `ollama/gemma2:2b`     | <img src="assets/images/random_walk_ollama_gemma2:2b_temp_0.0.png" width="300"> | <img src="assets/images/random_walk_ollama_gemma2:2b_temp_0.1.png" width="300"> | <img src="assets/images/random_walk_ollama_gemma2:2b_temp_0.3.png" width="300"> | <img src="assets/images/random_walk_ollama_gemma2:2b_temp_0.5.png" width="300"> |
| `ollama/gemma2:9b`     | <img src="assets/images/random_walk_ollama_gemma2:9b_temp_0.0.png" width="300"> | <img src="assets/images/random_walk_ollama_gemma2:9b_temp_0.1.png" width="300"> | <img src="assets/images/random_walk_ollama_gemma2:9b_temp_0.3.png" width="300"> | <img src="assets/images/random_walk_ollama_gemma2:9b_temp_0.5.png" width="300"> |

---

## Table 2: Temperatures 0.7, 0.9, 1.0

| Model                  | 0.7                                     | 0.9                                     | 1.0                                     |
|------------------------|------------------------------------------|------------------------------------------|------------------------------------------|
| `ollama/llama3.2:1b`   | <img src="assets/images/random_walk_ollama_llama3.2:1b_temp_0.7.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.2:1b_temp_0.9.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.2:1b_temp_1.0.png" width="300"> |
| `ollama/llama3.1:8b`   | <img src="assets/images/random_walk_ollama_llama3.1:8b_temp_0.7.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.1:8b_temp_0.9.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.1:8b_temp_1.0.png" width="300"> |
| `ollama/llama3.2:3b`   | <img src="assets/images/random_walk_ollama_llama3.2:3b_temp_0.7.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.2:3b_temp_0.9.png" width="300"> | <img src="assets/images/random_walk_ollama_llama3.2:3b_temp_1.0.png" width="300"> |
| `ollama/gemma2:2b`     | <img src="assets/images/random_walk_ollama_gemma2:2b_temp_0.7.png" width="300"> | <img src="assets/images/random_walk_ollama_gemma2:2b_temp_0.9.png" width="300"> | <img src="assets/images/random_walk_ollama_gemma2:2b_temp_1.0.png" width="300"> |
| `ollama/gemma2:9b`     | <img src="assets/images/random_walk_ollama_gemma2:9b_temp_0.7.png" width="300"> | <img src="assets/images/random_walk_ollama_gemma2:9b_temp_0.9.png" width="300"> | <img src="assets/images/random_walk_ollama_gemma2:9b_temp_1.0.png" width="300"> |


full code: [code](assets/A00002-A.py)

few animated frames for llama models random walk: [video](https://x.com/attentionmech/status/1870850915605021169)

for a video of all runs(too small visibility tbh): [all video](assets/output_grid.mp4)

For discussion: 
- [hackernews](https://news.ycombinator.com/item?id=42486923)
- [twitter thread](https://x.com/attentionmech/status/1870850915605021169)
