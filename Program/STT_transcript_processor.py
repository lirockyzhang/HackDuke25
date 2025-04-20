import re
from collections import deque

from BRAIN_organizer import think_about_what_to_say, interrupt

def split_clauses(text: str) -> list[str]:
    return [s for s in re.split(r'(?<=[\.\?\!])\s+', text.strip()) if s]

def update_window(text: str, window: deque[str]) -> str:
    for clause in split_clauses(text):
        window.append(clause)
    return " ".join(window)

from collections import deque

def transcript_processor(history):
    # context_window = deque(maxlen=5)
    # history = ""

    # while True:
    #     new_text = queue.get()  # ⏳ blocks until new item is added to queue
    #     if new_text is None:
    #         break  # 🔚 clean exit signal

    #     context = update_window(new_text, context_window)
    #     history += " " + new_text
    #     if_interrupt = interrupt(context)

    #     print("Interrupt:" + if_interrupt)

    #     if if_interrupt != "False":
    #         print("Thinking about what to say!")
    #         think_about_what_to_say(history, if_interrupt)
    think_about_what_to_say(history, "True")