from flask import Flask, request, jsonify
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
import os
import json
import random
import openai
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DIFFICULTIES = ["Easy", "Medium", "Hard", "Very Hard"]
CATEGORIES = [
    "General",
    "History",
    "Fun Fact",
    "Records/Statistics",
]

@app.route("/quiz", methods=["POST"])
def quiz():
    if "image" not in request.files:
        return jsonify({"error": "image required"}), 400

    image = request.files["image"]
    difficulty = request.form.get("difficulty", "Medium").strip().title()
    category = request.form.get("category", "General")

    if difficulty not in DIFFICULTIES or category not in CATEGORIES:
        return jsonify({"error": "invalid difficulty or category"}), 400

    # 1. Gemini describes the exact object
    image_bytes = image.read()
    gemini_response = gemini_client.models.generate_content(
    model="gemini-2.5-pro",
    contents=[
        genai_types.Part.from_bytes(data=image_bytes, mime_type=image.mimetype),
        """
Describe the main subject of the image in a simple, grounded, visual way.

GOALS:
- Works for ANY image (object, food, tech, animals, vehicles, packaging, clothing, furniture, scenes, etc.)
- Keep descriptions short (1–2 sentences).
- Only describe what is visually obvious.
- Allow brand/model guessing ONLY when clearly visible or iconic.
- Avoid hallucinating specifics.

RULES:
- If no clear brand/model/year is visible or iconic, use:
  "unknown brand", "unknown model", "unknown year".
- Context should be simple and visual (e.g., "on a table", "outdoors").
- Category_general should be a broad type (e.g., "food", "vehicle", "animal", "tool", "electronics", "furniture").
- Materials should include only the most obvious ones.

OUTPUT ONLY valid JSON with this structure:

{
  "description": "1–2 sentence simple description of the main subject.",
  "brand": "Visible or iconic brand, or 'unknown brand'.",
  "model": "Visible or iconic model, or 'unknown model'.",
  "year": "Visible year, rough era if truly obvious, or 'unknown year'.",
  "color": "Main visible colors.",
  "condition": "Basic visible condition.",
  "style": "Simple style descriptor.",
  "category_general": "Broad category like 'food', 'vehicle', 'tool', 'electronics', etc.",
  "material": "Main visible materials.",
  "context": "Short visual context like 'on a table', 'in a kitchen', 'outdoors'.",
  "size": "Simple size descriptor like 'small', 'medium', 'large'.",
  "notable_features": "Key features that stand out visually."
}

No markdown. No commentary. JSON only.
"""
    ]
)
    desc = gemini_response.text

    try:
        # handle possible ```json ... ``` wrapping
        cleaned = desc.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # naive split to remove language marker if present
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        obj = json.loads(cleaned)
    except Exception:
        obj = {"description": desc}

    # Save Gemini description to description.txt
    with open("description.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False))

    # 2. GPT creates quiz + hint
    # GPT will ALWAYS put the correct answer as the FIRST option.
    # We'll shuffle them afterwards and compute the new correct_index.
    gpt_prompt = f"""
You are a professional quiz creator for a production mobile app.

You are given:
1. A structured object description generated from the actual image (this is ground truth).
2. A selected Category.
3. A selected Difficulty level.

Your job: Create ONE high-quality quiz tied directly to the object in the image.

============================================================
IMAGE BINDING RULES (STRICT)
============================================================
• The question MUST be about the exact object described.
• If a brand or model is identified:
    - Background MUST reference it.
    - The question MUST be about something meaningfully tied to that brand/model’s
      design, engineering, history, surprising facts, or cultural significance.
• DO NOT produce simple guess-who, guess-what, or name-based questions
  (for example, “Who is this shoe named after?”) for Hard or Very Hard.
• Only if BOTH brand AND model are unknown may the question be based on a
  category-level topic instead of a specific product line.

============================================================
CATEGORY NAME BAN
============================================================
The question MUST NOT include the category name in the text:
• Do NOT use words like: "fun fact", "history", "historically", "record", "statistical record", or similar.
• The category should influence the *type of insight*, NOT appear as wording.

============================================================
BACKGROUND REQUIREMENTS
============================================================
• Start with 1–2 sentences of background.
• Background MUST:
  - Reference specific visual traits in the object description
    (color, materials, condition, shape, logos, text on the object, etc.).
  - Reference the brand/model if known and not marked as unknown.
  - Naturally lead into the question in a smooth, story-like way.
• No filler. No generic “sneakers are popular” trivia if details about the specific object exist.

============================================================
CATEGORY INTERPRETATION RULES
============================================================
General
→ Any trivia question. Can be a combination of other categories or anyting about the object/thing.

History
→ Origins, evolution, milestones, or development of THIS brand/model/type.

Fun Fact
→ Unexpected, quirky, surprising, or lesser-known insights about THIS brand/model/type.

Records/Statistics
→ Achievements, breakthroughs, notable comparisons, firsts, or extremes tied to THIS
   brand/model/type.

============================================================
DIFFICULTY SYSTEM (EXTREMELY STRICT)
============================================================

-------------------
EASY
-------------------
• Question: straightforward; can be answered using the background alone.
• Distractors: Plausible but clearly weaker or less fitting than the correct answer.
• No deep knowledge required.

-------------------
MEDIUM
-------------------
• Question: requires combining background clues with general knowledge.
• Distractors: Plausible alternatives that share clear similarities with the correct answer.
• No specialist-level knowledge needed, but not answerable by random guessing.

-------------------
HARD
-------------------
• Question: nuanced and tied to deeper design, engineering, or historical details
  of the brand/model/type or its collaboration context.
• Question MUST NOT be something visible directly in the image (logo text, color, etc.).
• Avoid simple “who is this named after?” style questions.
• Distractors: Very close in theme and details; none can be obviously wrong.
• Must require knowledgeable reasoning, careful reading of the background, or
  real-world context beyond the obvious.

-------------------
VERY HARD
-------------------
• Question: feels expert-level and subtle.
• Must involve deeper design reasoning, innovation history, niche engineering details,
  or conceptual product lineage decisions related to THIS brand/model/type.
• MUST NOT be directly answerable from the image alone.
• Avoid superficial or widely known facts (e.g., “which athlete?” if the name is in the product).
• Distractors: Nearly indistinguishable without strong domain knowledge.
• Only highly knowledgeable users should be confident.

============================================================
HINT RULES (NON-NEGOTIABLE)
============================================================
The hint MUST:
• NOT contain any keywords from the correct answer.
• NOT mention brand names, model names, athlete names, product-line names, or dates.
• NOT reveal the category.
• MUST scale with difficulty:

EASY → a clear but not giveaway nudge.  
MEDIUM → indirect but still practically helpful.  
HARD → abstract or conceptual, no direct identifiers.  
VERY HARD → one short sentence that is cryptic, metaphorical, or philosophical.

Examples of valid Very Hard hints:
• "The shift happens before the form takes shape."
• "Consider the intention behind the refinement."
• "Look to what is added by removing."

============================================================
DISTRACTOR RULES (IMPORTANT)
============================================================
ALL wrong answers MUST:
• Be plausible and of the same *type* as the correct answer
  (if the correct answer is a fun-fact choice, all options must be fun-fact choices; 
   if it’s a collaboration detail, all options must be collaboration details, etc.).
• Scale with difficulty:
  - EASY → somewhat believable but clearly weaker.
  - MEDIUM → closely related and reasonable alternatives.
  - HARD → subtle variations on the same theme, hard to eliminate without nuance.
  - VERY HARD → conceptual differences that only experts or careful readers can parse.
• Never be ridiculous or obviously false.

============================================================
OUTPUT FORMAT
============================================================
Return ONLY valid JSON in this structure:

{{
  "question": "Full text: background + question.",
  "options": [
    "Correct answer FIRST",
    "Plausible wrong answer",
    "Plausible wrong answer",
    "Plausible wrong answer"
  ],
  "hint": "Difficulty-scaled hint.",
  "explanation": "Short explanation of why the correct answer is correct."
  "title": "Very short title for object/question"
}}

============================================================
INPUTS
============================================================
Object description (ground truth): {json.dumps(obj, ensure_ascii=False)}
Category: {category}
Difficulty: {difficulty}

Generate the quiz now.
"""

    gpt_response = openai_client.chat.completions.create(
        model="gpt-4o",
        temperature=0.8,
        messages=[
            {"role": "system", "content": "You are a precise but friendly quiz generator. Always reply with valid JSON only."},
            {"role": "user", "content": gpt_prompt},
        ],
    )

    quiz_text = gpt_response.choices[0].message.content.strip()

    # Strip ```json ... ``` if GPT wraps it
    if quiz_text.startswith("```"):
        quiz_text = quiz_text.strip("`")
        # remove leading language marker if present
        if quiz_text.lower().startswith("json"):
            quiz_text = quiz_text[4:].strip()

    quiz = json.loads(quiz_text)

    # Save GPT quiz output to quiz.txt
    with open("quiz.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(quiz, indent=2, ensure_ascii=False))

    # 3. Shuffle options while tracking the correct answer index
    options = quiz.get("options", [])
    if not options or len(options) < 2:
        return jsonify({"error": "LLM did not return enough options"}), 500

    # We assume the FIRST option is correct before shuffling
    correct_answer = options[0]

    # Create a shuffled copy of the options
    shuffled_options = options[:]
    random.shuffle(shuffled_options)

    # New index of the correct answer after shuffling
    try:
        correct_index = shuffled_options.index(correct_answer)
    except ValueError:
        # Safety fallback: if for some reason the correct answer vanished,
        # just force the first option to be correct.
        shuffled_options[0] = correct_answer
        correct_index = 0

    # Build final quiz with shuffled answers and correct answer index
    final_quiz = {
        "question": quiz.get("question", ""),
        "options": shuffled_options,
        "correct_index": correct_index,
        "hint": quiz.get("hint", ""),
        "explanation": quiz.get("explanation", ""),
        "title": quiz.get("title", ""),
        "difficulty": difficulty,
        "category": category
    }

    # Save quiz to quiz.json (after shuffling, including answer index)
    with open("quiz.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(final_quiz, indent=2, ensure_ascii=False))

    return jsonify(final_quiz)

if __name__ == "__main__":
    app.run(port=5000)
