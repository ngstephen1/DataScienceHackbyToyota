import google.generativeai as genai
print("=== PROGRAM STARTED ===")

genai.configure(api_key="AIzaSyA635IfolASfClWwQ7QFWEkArh2EczNegk")

model = genai.GenerativeModel("gemini-2.0-flash")

with open("/Users/tranminhtue/Documents/GitHub/DataScienceHackbyToyota/src/Screenshot 2025-11-23 at 04.39.03.png"):

    mime_type = "image/png"

response = model.generate_content(
    f"""Your job:

1. **High-level situation read**
   - In plain language, what story do these numbers tell about the car and race?
   - Where are we strong / weak (pace, consistency, sectors, tyre life, traffic, etc.)?

2. **Perfect pit stop window**
   - Based on stint evolution, lap-time falloff, and pit-lane loss:
     - When would you recommend pitting in a normal green-flag race?
     - How many stops, and roughly which lap ranges?
     - Explain the tradeoffs (track position vs. tyre advantage vs. fuel).

3. **Caution / safety-car reaction**
   - Given these patterns, how should the tool react if a caution appears:
     - **Early race** (before ideal window)
     - **In the ideal window**
     - **Late race**
   - Describe simple rules or thresholds the tool could implement.
   - Call out any “must box now” situations vs “stay out” situations.

4. **What to build into the tool**
   - List 5–8 concrete features or signals the tool should monitor in real time
     (e.g. rolling average lap delta vs class leader, predicted stint length,
     green-flag pit loss, caution-risk index, etc.).
   - For each, describe *how the tool would use it* to recommend or veto a pit stop.

Keep the output focused, actionable, and oriented toward building a
real-time race-engineer copilot.""", file,
)

print(response.text)
