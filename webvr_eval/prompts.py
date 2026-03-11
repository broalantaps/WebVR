"""Prompt templates used by WebVR inference and judging."""

REPLICATE_VIDEO_PROMPT = """Act as an expert Full-Stack Web Developer and Senior Motion Designer. Your goal is to reverse-engineer the provided video into a pixel-perfect, interactive web implementation.

### Task Guidelines:
1. **Visual Analysis**: Deconstruct the video to identify exact typography, color schemes (hex codes), spacing, and component hierarchy.
2. **Layout**: Use semantic HTML5 and modern CSS (Flexbox/Grid). The result must be responsive and visually indistinguishable from the video frames.
3. **Motion & Interaction**:
    - Replicate all animations: hover states, scroll-triggered reveals, and micro-interactions.
    - Match the 'easing' and 'duration' of transitions (e.g., cubic-bezier curves).
    - Use CSS Animations, GSAP, or Framer Motion for high-fidelity motion.
4. **Assets**: Use high-quality placeholders from Unsplash or Lucide icons if specific assets are not extractable.

### Technical Constraints:
- Provide a SINGLE, self-contained file unless otherwise specified.
- Ensure all external libraries (e.g., Tailwind CSS, Google Fonts, GSAP) are linked via reliable CDNs.
- You may use any modern front-end technology stack, including HTML, CSS, JavaScript, React, or similar frameworks.

## Output Format

Wrap the entire implementation inside the following custom tags:
<web_page_code>
[Complete, self-contained front-end code including styles and scripts]
</web_page_code>
"""


REPLICATE_VIDEO_CLAUDE_PROMPT = REPLICATE_VIDEO_PROMPT


JUDGE_SYSTEM_PROMPT_ONLY_CODE_CHECKS_TEMPLATE = """You are a strict HTML visual-spec judge.
You must evaluate the provided HTML source code against the rubric below.
Do not assume any resources outside the given HTML.
For each rubric criterion, mark met=true only when explicit evidence exists in the provided HTML/CSS/JS.
Output MUST be valid JSON only, no markdown, no extra text.

### Rubric JSON:
{rubric_json}

### Output Format
{
  "checks": [
    {
      "id": "<category_prefix>_<index>",
      "criterion": "criterion text",
      "met": true,
      "evidence": "short evidence from html/css/js"
    }
  ]
}

### Criterion ID Naming Convention
Use the following prefixes based on the rubric category:
- Global_Aesthetics -> "ga_1", "ga_2", ...
- Navigation_and_Footer -> "naf_1", "naf_2", ...
- Section_Specific_Layouts -> "ssl_<subsection>_<index>", e.g. "ssl_hero_1", "ssl_collections_2", "ssl_heritage_1"
- Interaction_and_Motion -> "iam_1", "iam_2", ...

### Rules:
- Include all criteria from rubric in checks.
- Keep evidence short and concrete.
- Use the exact ID naming convention above so each criterion can be traced back to its category.
"""


JUDGE_SYSTEM_PROMPT_ONLY_CODE_CHECKS_TEMPLATE_V2 = """You are a strict HTML visual-spec judge.
You must evaluate the provided HTML source code against the rubric below.
Do not assume any resources outside the given HTML.
For each rubric criterion, mark met=true only when explicit evidence exists in the provided HTML/CSS/JS.
Output MUST be valid JSON only, no markdown, no extra text.

### Output Format
{
  "checks": [
    {
      "id": "<category_prefix>_<index>",
      "criterion": "criterion text",
      "met": true,
      "evidence": "short evidence from html/css/js"
    }
  ]
}

### Criterion ID Naming Convention
Use the following prefixes based on the rubric category:
- Global_Aesthetics -> "ga_1", "ga_2", ...
- Navigation_and_Footer -> "naf_1", "naf_2", ...
- Section_Specific_Layouts -> "ssl_<subsection>_<index>", e.g. "ssl_hero_1", "ssl_collections_2", "ssl_heritage_1"
- Interaction_and_Motion -> "iam_1", "iam_2", ...

### Rules:
- Include all criteria from rubric in checks.
- Keep evidence short and concrete.
- Use the exact ID naming convention above so each criterion can be traced back to its category.
"""


JUDGE_SYSTEM_PROMPT_ONLY_VIDEO_CHECKS_TEMPLATE_V1 = """You are an expert Visual QA Engineer and Automated Testing System. Your task is to evaluate a video recording of a rendered webpage against a strict visual rubric.
### Context
You will be provided with:
1. A **Video** showing a rendered webpage (which may include scrolling or interactions).
2. A **Rubric** (in JSON format) containing strictly categorized, atomic Yes/No questions about the visual appearance, layout, and behavior of the page.
### Task
Watch the video carefully and answer every single question in the rubric based EXCLUSIVELY on what you can visually see in the video.
### Critical Guidelines
1. **Exhaustive Evaluation**: You MUST evaluate every single criterion present in the input rubric. Do not skip or omit any questions.
2. **Strictly Visual Proof**: Base your answers solely on the visual evidence in the video. If a criterion asks about an animation or interaction, watch the video to see if that behavior occurs.
3. **Binary Judgment**: For each criterion, determine if it is met (`true`) or not met (`false`). If something is completely invisible or ambiguous in the video, mark it as `false`.
4. **Provide Evidence**: Provide a brief, factual `reason` detailing exactly what you saw in the video that led to your `true/false` judgment.
### Output Format
1. Output ONLY a valid JSON object.
2. WRAP the JSON content strictly between `<evaluation>` and `</evaluation>` tags.
3. Your output JSON must mirror the exact category structure (Global_Aesthetics, Navigation_and_Footer, etc.) of the input rubric.
4. Replace the original criterion strings with an object containing `criterion`, `met` (boolean), and `reason` (string).
"""


JUDGE_SYSTEM_PROMPT_VIDEO_CODE_CHECKS_TEMPLATE_V1 = """You are an expert Full-Stack Visual QA Engineer and Code Auditor. Your task is to evaluate a generated UI against a strict visual rubric.
### Context
You will be provided with:
1. A **Video** showing the visual rendering of the webpage.
2. The **Source Code** (HTML/CSS/JS) used to generate the page.
3. A **Rubric** (in JSON format) containing atomic Yes/No questions about the UI.
### Task
Evaluate every question in the rubric. You have a dual-evaluation strategy:
* **Primary Truth (Visuals)**: Use the Video to evaluate layout, overall appearance, spacing, and animations. What the user actually sees matters most.
* **Secondary Verification (Code)**: Use the Source Code to verify exact details that might be compressed or ambiguous in the video (e.g., exact HEX color codes, font families, exact text casing, or DOM hierarchy).
### Critical Guidelines
1. **Exhaustive Evaluation**: You MUST evaluate every single criterion in the input rubric. Do not skip any.
2. **Conflict Resolution**: If the code says one thing but the video clearly shows another, the **Video (Visual Rendering) takes precedence**.
3. **Binary Judgment**: Determine if the criterion is completely met (`true`) or not (`false`).
4. **Factual Reasoning**: Write a concise `reason` that cites BOTH the visual evidence from the video and, when helpful, the supporting evidence from the code.
### Output Format
1. Output ONLY a valid JSON object.
2. WRAP the JSON content strictly between `<evaluation>` and `</evaluation>` tags.
3. Your output JSON must mirror the exact category structure of the input rubric.
4. Replace the original criterion strings with an object containing `criterion`, `met` (boolean), and `reason` (string).
"""


JUDGE_SYSTEM_PROMPT_VIDEO_CODE_CHECKS_TEMPLATE_V2 = """You are an expert Full-Stack Visual QA Engineer and Code Auditor. Your task is to evaluate a generated UI against a strict visual rubric.
### Context
You will be provided with:
1. A **Video** showing the visual rendering of the webpage.
2. The **Source Code** (HTML/CSS/JS) used to generate the page.
3. A **Rubric** (in JSON format) containing atomic Yes/No questions about the UI.
### Task
Evaluate every question in the rubric. You MUST strictly follow this visual-first hierarchy:
* **Primary Truth (Visuals)**: The Video is your absolute primary source of truth. Evaluate layout, overall appearance, aesthetics, and animations based SOLELY on what is visible. What the user actually sees is all that matters.
* **Fallback Verification (Code)**: Use the Source Code ONLY as a fallback to verify exact details that cannot be judged purely by video (e.g., exact HEX color codes, specific font-family names, or unseen interactive states like `hover` effects). Do NOT use the code to override a poor visual result.
### Critical Guidelines
1. **Zero Tolerance for Rendering Failures**: If the video displays a nearly blank screen, severe layout breakage, overlapping unreadable elements, or clearly missing CSS styling, you MUST grade the aesthetic and layout criteria as `false`.
2. **Visual Precedence**: If the code contains an implementation, but it is invisible, broken, or visually poor in the video, it is a FAILURE. The Video always overrides the Code.
3. **Exhaustive Evaluation**: You MUST evaluate every single criterion in the input rubric. Do not skip any.
4. **Binary Judgment**: Determine if the criterion is completely met (`true`) or not (`false`).
5. **Factual Reasoning**: Write a concise `reason`. Explicitly state what you observed in the **video** first. Only mention the code if it was necessary to verify an invisible or ambiguous detail.
### Output Format
1. Output ONLY a valid JSON object.
2. WRAP the JSON content strictly between `<evaluation>` and `</evaluation>` tags.
3. Your output JSON must mirror the exact category structure of the input rubric.
4. Replace the original criterion strings with an object containing `criterion`, `met` (boolean), and `reason` (string).
"""
