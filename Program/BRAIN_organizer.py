from BRAIN_llm_init import llama_slow as llm
from BRAIN_langchain import combined_memory, memory_chain

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from BRAIN_llm_init import llama_slow as interrupt_llm
import tts_engine


def think_about_what_to_say(history, interrupt_way):

    if interrupt_way != "True":
        prompt = (
            "Provide one appropriate English interjection (one word) to respond to the following conversation history:\n\n"
            f"{history}\n"
            "Output only that word with no extra text."
        )
        interjection = llm.predict(prompt).strip()
        tts_engine.tts_engine.speak_text(interjection)
        return

    # Otherwise, proceed with full response
    respond(history)


def respond(history):

    user_input = history
    
    # 1) retrieve memory
    mem_vars = combined_memory.load_memory_variables({"input": user_input})

    # 2) generate response via memory_chain
    response_text = memory_chain.predict(input=user_input, **mem_vars)

    # 3) play audio and get the final string
    tts_engine.tts_engine.speak_immediately(response_text)

    actual_output = response_text

    # 4) update memory with separate input/output
    combined_memory.save_context(
        {"input": user_input},
        {"output": actual_output}
    )

    return 0


# Define the prompt
interrupt_prompt = PromptTemplate.from_template(
    "Given the following spoken context:\n\n\"{context}\"\n\n"
    "Should you interrupt right now?\n"
    "Reply with one of: True, False, or Interjection.\n"
)

# Wrap in an LLMChain
interrupt_chain = LLMChain(llm=interrupt_llm, prompt=interrupt_prompt)

def interrupt(context: str):
    context = context.strip()
    
    if context == "" or context.endswith("<silence> <silence>"):
        return "True"
    
    try:
        response = interrupt_chain.run(context=context).strip()
    except Exception as e:
        return "False"

    if response not in {"True", "False", "Interjection"}:
        return "False"  # fallback for malformed response

    return response