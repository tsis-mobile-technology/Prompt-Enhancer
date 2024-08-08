from langchain_community.chat_models import ChatOpenAI
import google.generativeai as genai
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
import os 
from src.prompts import templates


def load_model(model_name):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=model_name)
    return llm 


def convert_newlines(prompt):
    prompt = prompt.replace("\n", "  \n")
    return prompt


def apply_skill(llm, skill, prompt, order_num, lang_eng=False):
    system_message = templates["system"]
    if lang_eng and order_num == 1:
        system_message += '\n' + templates["lang_eng"]
    elif not lang_eng:
        system_message += '\n' + templates["lang_default"]

    template = templates[skill]
    prompt_template = PromptTemplate.from_template(template)
    formatted_input = prompt_template.format(prompt=prompt)
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=formatted_input),
    ]

    response = llm.invoke(messages)

    return response.content


def apply_skills(llm, skills_to_apply, prompt, lang_eng=False):
    system_message = templates["system_multiple"]
    if lang_eng:
        system_message += '\n' + templates["lang_eng"]
    else:
        system_message += '\n' + templates["lang_default"]

    skills = [skill for skill, toggled in skills_to_apply.items() if toggled]
    integrated_templates = "[Prompt Engineering Techniques to Apply]\n"

    for idx, skill in enumerate(skills):
        template = templates[f"{skill}_simpler"]
        integrated_templates += f"{idx+1}. {skill}: {template}\n"
    integrated_templates += "Based on [Prompt engineering techniques to apply], refine the prompt provided below. Ensure that each technique is fully incorporated to achieve a clear and effective improvement:\n\n[original]\n{prompt}\n[improved]\n"

    prompt_template = PromptTemplate.from_template(integrated_templates)
    formatted_input = prompt_template.format(prompt=prompt)

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=formatted_input),
    ]

    response = llm.invoke(messages)

    return response.content

# src/utils.py
# ... (다른 유틸리티 함수들)

def load_gemini_model(model_name, system_instruction):
    """
    Gemini 모델을 로드하는 함수 (임시 구현)
    """
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    # Create the model
    generation_config = {
        "temperature": 0.85,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        system_instruction=system_instruction,
    )
    
    print("Gemini 모델 로드 (임시)")
    return model

def apply_skill_gemini(model_name, skill, prompt, order_num, lang_eng=False):
    """
    Gemini 모델에 스킬을 적용하는 함수 (임시 구현)
    """
    system_message = templates["system"]
    if lang_eng and order_num == 1:
        system_message += '\n' + templates["lang_eng"]
    elif not lang_eng:
        system_message += '\n' + templates["lang_default"]

    template = templates[skill]
    prompt_template = PromptTemplate.from_template(template)
    formatted_input = prompt_template.format(prompt=prompt)
    
    # messages = [
    #     SystemMessage(content=system_message),
    #     HumanMessage(content=formatted_input),
    # ]

    response = load_gemini_model(model_name, system_message).start_chat().send_message(formatted_input)

    print(f"Gemini 스킬 적용: {skill} (임시) {system_message, formatted_input}")
    
    return response.text

def apply_skills_gemini(model_name, skills_to_apply, prompt, lang_eng):
    """
    여러 Gemini 스킬을 한 번에 적용하는 함수 (임시 구현)
    """
    system_message = templates["system_multiple"]
    if lang_eng:
        system_message += '\n' + templates["lang_eng"]
    else:
        system_message += '\n' + templates["lang_default"]

    skills = [skill for skill, toggled in skills_to_apply.items() if toggled]
    integrated_templates = "[Prompt Engineering Techniques to Apply]\n"

    for idx, skill in enumerate(skills):
        template = templates[f"{skill}_simpler"]
        integrated_templates += f"{idx+1}. {skill}: {template}\n"
    integrated_templates += "Based on [Prompt engineering techniques to apply], refine the prompt provided below. Ensure that each technique is fully incorporated to achieve a clear and effective improvement:\n\n[original]\n{prompt}\n[improved]\n"

    prompt_template = PromptTemplate.from_template(integrated_templates)
    formatted_input = prompt_template.format(prompt=prompt)

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=formatted_input),
    ]

    response = load_gemini_model(model_name, system_message).start_chat().send_message(formatted_input)

    print(f"여러 Gemini 스킬 적용 (임시), {messages}")
    return response.text
