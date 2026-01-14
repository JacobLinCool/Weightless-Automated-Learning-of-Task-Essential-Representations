# Zero-Shot Prompt Optimization via Feature Search

**Objective:** Automatically discover the optimal set of high level representations (audio/visual features) to include in the context window to maximize target model's zero-shot performance.

**Algorithm:**

1. **Initialization:** Initialize the input context feature set S with a baseline representation (e.g., `{ raw audio }`).

2. **Zero-Shot Evaluation:**
   * **Inference:** For a batch of samples, construct a prompt containing all representations in S and query the model for predictions.
   * **Selection:** Identify the "Hard Negatives"—samples where the model confidently predicted the wrong answer.

3. **Reflective Feedback (The "Analyst" Step):**
   * Provide the model with the **failed sample**, its **wrong prediction**, and the **ground truth**.
   * **Prompt:** *"You answered X, but the truth is Y. Looking at the current features provided, what specific visual or numerical representation (e.g., a Pitch Contour, Onset Strength Envelope, CQT spectrogram) is missing that would have highlighted the difference?"*

4. **Feature Implementation:**
   * Pass the suggested feature description to the Coding Agent.
   * The Agent implements the extractor, verifies it runs without errors, and adds the new representation to S.

5. **Pruning & Optimization:**
   * If  (context limit or cost constraints):
   * Ask the model to identify the least informative representation in the current set and remove it.
   * **Constraint:** Do not remove **Protected Features**. A feature becomes protected if it being added repeatedly (e.g., 3 times) by the model. (So we know it's important.)

6. **Termination:** Repeat from **Step 2** until the zero-shot accuracy stops improving for  consecutive steps.

---

> Note: The high-level features here refer to pre-computed audio/visual representations (e.g., spectrograms, pitch contours, object detections, music source separation) that can be included in the model's context window to aid its understanding and prediction.
