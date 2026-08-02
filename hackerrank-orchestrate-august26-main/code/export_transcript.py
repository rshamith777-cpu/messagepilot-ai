import os
import json
import re

def export_full_transcript():
    log_dir = r"C:\Users\SUMITH R\.gemini\antigravity-ide\brain\6ce21736-c7d4-4957-bc62-8ef67e47858f\.system_generated\logs"
    jsonl_path = os.path.join(log_dir, "transcript_full.jsonl")
    if not os.path.exists(jsonl_path):
        jsonl_path = os.path.join(log_dir, "transcript.jsonl")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai_chat_transcript.txt")

    if not os.path.exists(jsonl_path):
        print(f"Log file not found: {jsonl_path}")
        return

    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass

    transcript_lines = [
        "==========================================================================",
        "       MESSAGEPILOT AI — COMPLETE AI DEVELOPMENT & CHAT TRANSCRIPT       ",
        "==========================================================================",
        "Project: MessagePilot AI (WhatsApp Notification Router)",
        "Repository: https://github.com/rshamith777-cpu/messagepilot-ai.git",
        "System: Antigravity AI Pair Programming Session",
        "Conversation ID: 6ce21736-c7d4-4957-bc62-8ef67e47858f",
        "--------------------------------------------------------------------------\n"
    ]

    step_num = 1

    for e in entries:
        src = e.get("source")
        t = e.get("type")
        content = e.get("content")

        if t == "USER_INPUT" and content:
            if isinstance(content, dict):
                text = content.get("text", "")
            else:
                text = str(content)

            if "<USER_REQUEST>" in text:
                m = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", text, re.DOTALL)
                if m:
                    text = m.group(1).strip()
            
            # Clean up prompt wrappers if present
            if text.startswith("{{ CHECKPOINT"):
                m = re.search(r"# User Requests\s*(.*?)(?=# Previous Session Summary|\Z)", text, re.DOTALL)
                if m:
                    text = m.group(1).strip()

            text = text.strip()
            if text and not text.startswith("[SYSTEM_MESSAGE]"):
                transcript_lines.append(f"[STEP {step_num}] USER PROMPT:")
                transcript_lines.append(text)
                transcript_lines.append("\n" + "-" * 70 + "\n")
                step_num += 1

        elif src == "MODEL" and t == "PLANNER_RESPONSE" and content:
            if isinstance(content, dict):
                text = content.get("text", "")
            else:
                text = str(content)

            # Filter out raw internal JSON tool call dumps, keeping human markdown explanation
            text = re.sub(r"```json\s*\{.*?\}\s*```", "", text, flags=re.DOTALL)
            text = text.strip()
            if text:
                transcript_lines.append(f"[STEP {step_num}] ANTIGRAVITY AI RESPONSE:")
                transcript_lines.append(text)
                transcript_lines.append("\n" + "=" * 75 + "\n")
                step_num += 1

    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(transcript_lines))

    print(f"Successfully exported full AI chat transcript to: {out_path}")
    print(f"Transcript size: {os.path.getsize(out_path) / 1024:.1f} KB | Total conversation turns: {step_num - 1}")

if __name__ == "__main__":
    export_full_transcript()
