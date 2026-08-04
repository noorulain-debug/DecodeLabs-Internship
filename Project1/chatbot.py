import random
import difflib

# 1. KNOWLEDGE BASE
knowledge_base = {
    "greeting": {
        "keywords": ["hello", "hi", "hey"],
        "responses": ["Hi there! How are you?", "Hey! What's up?", "Hello! Great to see you."]
    },
    "how_are_you": {
        "keywords": ["how are you", "how you doing"],
        "responses": ["I'm just code, but I'm doing great! How about you?",
                       "Feeling logically sound today! And you?"]
    },
    "identity": {
        "keywords": ["your name", "who are you"],
        "responses": ["I'm Byte, your Project 1 chatbot.", "Call me Byte — nice to meet you."]
    },
    "capability": {
        "keywords": ["what can you do"],
        "responses": ["I can chat about basics — try 'hello', 'joke', or 'weather'!"]
    },
    "weather": {
        "keywords": ["weather"],
        "responses": ["I can't check real weather yet — I'm rule-based, not connected to the internet!"]
    },
    "joke": {
        "keywords": ["joke", "make me laugh"],
        "responses": ["Why do programmers prefer dark mode? Because light attracts bugs.",
                       "I'd tell you a UDP joke, but you might not get it."]
    },
    "thanks": {
        "keywords": ["thanks", "thank you"],
        "responses": ["You're welcome!", "Anytime!"]
    }
}

exit_commands = ["bye", "exit", "quit", "goodbye", "see you"]
exit_responses = ["Goodbye! Have a great day.", "See you soon!", "Bye! Take care."]

# 2. MOOD WORDS — used only for context/follow-up handling
mood_words = {
    "good": "Glad to hear that!",
    "great": "Awesome, that's great!",
    "fine": "Good to know you're doing fine.",
    "okay": "Alright, thanks for sharing.",
    "ok": "Alright, thanks for sharing.",
    "bad": "Sorry to hear that. I hope things get better.",
    "not great": "Sorry to hear that. I hope things get better."
}

conversation_log = []
last_intent = None   # remembers what the bot last talked about

# 3. HELP MENU
def show_help():
    print("Bot: Here's what I can talk about:")
    for intent_name in knowledge_base:
        example = knowledge_base[intent_name]["keywords"][0]
        print(f"   - {intent_name} (try: '{example}')")
    print("   - Type 'bye', 'exit', or 'quit' anytime to leave.")

# 4. TYPO-TOLERANT KEYWORD MATCHING
def find_response(user_text):
    global last_intent

    # First pass: exact substring match (same as before)
    for intent_name, intent_data in knowledge_base.items():
        for keyword in intent_data["keywords"]:
            if keyword in user_text:
                last_intent = intent_name
                return random.choice(intent_data["responses"])

    # Second pass: fuzzy/typo matching on individual words
    all_keywords = []
    for intent_data in knowledge_base.values():
        all_keywords.extend(intent_data["keywords"])

    for word in user_text.split():
        close_matches = difflib.get_close_matches(word, all_keywords, n=1, cutoff=0.75)
        if close_matches:
            matched_keyword = close_matches[0]
            for intent_name, intent_data in knowledge_base.items():
                if matched_keyword in intent_data["keywords"]:
                    last_intent = intent_name
                    return random.choice(intent_data["responses"])

    # Third pass: context/follow-up check
    if last_intent == "how_are_you" and user_text in mood_words:
        return mood_words[user_text]

    return "I do not understand that yet. Type 'help' to see what I can do."

# 5. MAIN LOOP
print("Bot: Hi! I'm Byte. Type 'help' for options or 'bye' to leave.")

while True:
    raw_input_text = input("You: ")
    clean_input = raw_input_text.lower().strip()

    if clean_input in ["help", "menu"]:
        show_help()
        continue   # NEW: skip the rest of the loop, go straight back to asking for input

    if any(cmd in clean_input for cmd in exit_commands):
        farewell = random.choice(exit_responses)
        print("Bot:", farewell)
        conversation_log.append(("You", raw_input_text))
        conversation_log.append(("Bot", farewell))
        break

    reply = find_response(clean_input)
    print("Bot:", reply)

    conversation_log.append(("You", raw_input_text))
    conversation_log.append(("Bot", reply))

# 6. SUMMARY
print("\n--- Conversation Summary ---")
for speaker, text in conversation_log:
    print(f"{speaker}: {text}")
print(f"Total exchanges: {len(conversation_log) // 2}")