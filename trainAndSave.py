import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

# --- Data ---
# Labels: 0 = No Response, 1 = AI Response, 2 = Human Response
texts = [
    # ---- NO RESPONSE NEEDED (0) ----
    "Ok sounds good",
    "Sounds good!",
    "Works for me!",
    "Perfect, see you then!",
    "Got it!",
    "Never mind",
    "Nvm forget it",
    "Don't worry about it",
    "All good!",
    "No worries!",
    "Just saw this, sorry for the late reply",
    "Oops just saw this!",
    "Sorry, just now seeing this",
    "My bad, late reply",
    "🙏",
    "👍",
    "❤️",
    "😂",
    "lol",
    "Haha",
    "Same lol",
    "Lol same",
    "Haha exactly",
    "Right??",
    "Ikr!",
    "kk",
    "k",
    "omg same",
    "Totally",
    "Exactly",
    "Fair enough",
    "True true",
    "Yeah for sure",
    "Absolutely",
    "For sure!",
    "Cool!",
    "Nice!",
    "Awesome!",
    "Sweet!",
    "Perfect!",

    # ---- AI RESPONSE NEEDED (1) ----
    # Greetings
    "Hey!",
    "Heyyyy",
    "Good morning!",
    "Happy Friday!",
    "Hope you have a great day!",
    "Just wanted to say hi!",
    "Thinking of you!",
    "Miss you!",
    "Hey stranger!",
    "Long time no talk!",
    "How are you doing?",
    "How's it going?",
    "What's up?",
    "How's life?",
    "Hope you're well!",
    "How have you been?",
    "How's the family?",
    "Hope your week is going well!",
    "How's everything?",
    "You doing okay?",
    # Weather / casual observations
    "Can you believe this weather?",
    "Beautiful day out, right?",
    "This heat is insane today.",
    "It's freezing out there today!",
    "Looks like rain again.",
    "Finally feels like summer!",
    "The weather has been so weird lately.",
    "Such a gloomy day.",
    "Loving this sunny weather!",
    "Snow day!!",
    # Casual venting / life comments
    "Ugh, Monday again.",
    "Is it Friday yet??",
    "This week has been a lot.",
    "Traffic was a nightmare this morning.",
    "I could really go for some pizza rn.",
    "Dude, I just had the best burger.",
    "Bro the tacos from last night were so good.",
    "I'm so tired today lol",
    "I'm exhausted today.",
    "I could nap for 10 hours right now.",
    "Can this week be over already?",
    "I can't function without my morning coffee.",
    "Monday again already, huh.",
    "Thank goodness it's Friday.",
    "Almost the weekend!",
    "This week is flying by.",
    # Sports / pop culture
    "Did you see the game last night?!",
    "Can you believe that ending??",
    "Dude, that game was insane last night.",
    "That plot twist was insane.",
    "I've been binge-watching that show all weekend.",
    "Have you seen that new show everyone's talking about?",
    "Have you listened to that new album yet?",
    "Did you hear about that movie coming out?",
    "Have you tried that new restaurant downtown?",
    "Have you been to that new coffee shop?",
    # Compliments / affection
    "Hey, you looked great the other day!",
    "Dude your Instagram post was so good.",
    "You've been crushing it lately!",
    "Saw your post, you look amazing!",
    "That thing you said the other day really stuck with me.",
    "I love you!",
    "Love you!",
    "I love you son.",
    "I love you mom.",
    "Miss you so much!",
    "You're the best!",
    "You always make me smile.",
    "I'm so lucky to have you.",
    "You mean so much to me.",
    "I appreciate you so much.",
    # Social gratitude
    "Aw thanks, you're the best!",
    "Thanks for being such a good friend!",
    "Thanks for checking in!",
    "Aww thank you, that means a lot!",
    "You always know how to make me smile.",
    # Goodbyes / sign-offs
    "Talk soon!",
    "Catch you later!",
    "Have a good one!",
    "Enjoy your evening!",
    "Take care!",
    "Stay warm out there!",
    "Have a great weekend!",
    "Hope your week gets better!",
    "Don't work too hard!",
    "See you later!",
    "Good night!",
    "Sleep well!",
    "Sweet dreams!",
    "Morning!",
    "Rise and shine!",

    # ---- HUMAN RESPONSE NEEDED (2) ----
    # Favor / task requests
    "Can you help me move this Saturday?",
    "Hey, can you send me that file?",
    "Can you cover my shift on Friday?",
    "Can you pick me up from the airport?",
    "Hey, can you watch my dog this weekend?",
    "Can you proofread something for me real quick?",
    "Can you send me the address?",
    "Hey, can you Venmo me back when you get a chance?",
    "Can you grab milk on your way home?",
    "Can you forward me that email?",
    "Can you give me a ride tomorrow?",
    "Can you help me with something?",
    "Hey, I need a favor.",
    "Quick favor to ask, you free?",
    "Can you watch the kids Saturday night?",
    # Questions needing a real answer
    "What time does the party start?",
    "Hey, do you know a good plumber?",
    "Do you have Sarah's number?",
    "What's the address for tonight?",
    "Do you know what time the gym opens?",
    "What did the doctor say?",
    "Do you remember what we paid last time?",
    "Hey, what's the WiFi password at your place?",
    "Do you know a good mechanic?",
    "What time are you getting there?",
    "Do you have the login info?",
    "What's the plan for tomorrow?",
    "Do you know if he's coming?",
    # Scheduling / logistics
    "Are you free Tuesday evening?",
    "When are you back in town?",
    "Can we reschedule for Thursday?",
    "What time works for you?",
    "Are you still coming tomorrow?",
    "Hey, can we move our meeting?",
    "When do you land?",
    "What time should I be there?",
    "Are you still on for tonight?",
    "Can we push it back an hour?",
    "Hey, when's a good time to call?",
    "Are you free later today?",
    "Can we meet earlier?",
    "Is tomorrow still good for you?",
    "When do you need it by?",
    # Advice / opinion requests
    "Hey, can I get your opinion on something?",
    "What do you think I should do?",
    "Does this look okay?",
    "Should I take the job offer?",
    "What would you do in my situation?",
    "Do you think I'm overreacting?",
    "Can I run something by you?",
    "Would you reach out to him or just leave it?",
    "Should I say something or just let it go?",
    "What do you think about this email I'm about to send?",
    "Do you think it's worth it?",
    # Work / professional
    "Hey, quick question about the project.",
    "Did you get a chance to look at the doc I sent?",
    "Can we jump on a call today?",
    "Did you see my email?",
    "Hey, what's the status on that?",
    "Do you have the report ready?",
    "Can you review this before I send it?",
    "Did the client get back to you?",
    "Hey, are you joining the meeting?",
    "Can you handle that for me while I'm out?",
    "Did you finish the slides?",
    "Hey, I need that by end of day.",
    "What's your availability this week?",
    "Did you see the Slack message?",
    # Serious / emotional requiring human
    "Hey, my dad is in the hospital.",
    "I need to tell you something important.",
    "Something happened and I need to talk.",
    "I just got back from the hospital.",
    "I think we need to talk.",
    "Hey, are you around? It's urgent.",
    "Can we talk later? Something came up.",
    "Hey I need your advice on something serious.",
    "I have a problem and I don't know what to do.",
    "I just got diagnosed with cancer.",
    "I'm going through a really hard time.",
    "I lost my job today.",
    "My mom passed away.",
    "I'm getting a divorce.",
    "I've been struggling with my mental health.",
    "I don't know what to do anymore.",
    # Significant life events
    "I just got engaged!",
    "We're having a baby!",
    "I got into the program!",
    "We got the house!",
    "I got the job!!",
    # Follow-ups
    "Hey, any update on that?",
    "Just checking — did you get my last message?",
    "Did you ever figure that out?",
    "Hey, still waiting to hear back from you.",
    "Hey, did you make a decision yet?",
    "Any news on your end?",
    # Invitations requiring real RSVP
    "You coming to the party Saturday?",
    "Are you in for the trip?",
    "Want to grab dinner this week?",
    "You down to hang out tomorrow?",
    "Are you joining us for lunch?",
    "Want to come over this weekend?",
    "You free for a call later?",
    "Want to go to the game with us?",
    "Are you coming to the wedding?",
    "Want to grab coffee and catch up?",
    # Explicit human needed
    "We need to talk.",
    "Can we talk?",
    "We should talk soon.",
    "I need to talk to you.",
    "Call me when you can.",
    "Can you call me?",
    "Give me a call when you're free.",
    "Let me know what you think!",
    "Let me know if that works.",
    "Let me know when you're free.",
    "Let me know if you need anything.",
    "Just let me know!",
]

labels = [0]*40 + [1]*85 + [2]*123

print(f"Texts: {len(texts)}")
print(f"Labels: {len(labels)}")

# --- Embed ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X = embedder.encode(texts, show_progress_bar=True)

# --- Cross-validate ---
model = SVC(kernel='rbf', probability=True, C=1.0)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_f1 = cross_val_score(model, X, labels, cv=cv, scoring='f1_weighted')
scores_acc = cross_val_score(model, X, labels, cv=cv, scoring='accuracy')
print(f"F1:       {scores_f1.mean():.2f} ± {scores_f1.std():.2f}")
print(f"Accuracy: {scores_acc.mean():.2f} ± {scores_acc.std():.2f}")

# --- Train on all data ---
model.fit(X, labels)

# --- Classify ---
def classify_message(text):
    vec = embedder.encode([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred]
    labels_map = {0: "No Response Needed", 1: "AI Response", 2: "Human Response"}
    return pred, f"{labels_map[pred]} ({prob:.0%} confidence)"

async def handle_message(text):
    pred, label = classify_message(text)
    print(f"[{label}] '{text}'")

    if pred == 1:
        # AI drafts a reply
        response = await get_ai_response(text)
        print(f"AI reply: {response}")
    elif pred == 2:
        # Notify human to step in
        notify_human(text)

async def get_ai_response(text):
    # Plug into your existing AI pipeline here
    # e.g. OpenAI, Anthropic, etc.
    return f"[AI draft for: '{text}']"

def notify_human(text):
    # Plug into your notification system here
    # e.g. push notification, flag in UI, etc.
    print(f"[HUMAN NEEDED] Flag this message: '{text}'")

# --- Save ---
joblib.dump(model, 'message_classifier.pkl')

# --- Test ---
test_messages = [
    "Hey!",
    "I love you son.",
    "Can you call me?",
    "We're having a baby!",
    "Ok sounds good",
    "I just got diagnosed with cancer.",
    "Did you see the game last night?",
    "Want to grab dinner this week?",
    "lol",
    "We need to talk.",
    "What is your name?",
    "Who am I",
    "I'm so bored",
    "I farted on a pickle",
    "I miss you"
]

print("\n--- Test Results ---")
for msg in test_messages:
    _, result = classify_message(msg)
    print(f"'{msg}' → {result}")
