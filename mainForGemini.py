import streamlit as st
import os
from google.generativeai import GenerativeModel
from src.utils import load_gemini_model, apply_skill_gemini, apply_skills_gemini, convert_newlines
from src.prompts import insert

st.set_page_config(page_title="Prompt Enhancer", page_icon=":rocket:", initial_sidebar_state="collapsed", layout="wide")
st.title(":rocket: Prompt Enhancer")
st.markdown("""
##### Prompt Engineering at Your Fingertips!
""")
st.text("")
st.text("")


with st.popover("**:blue[Enter your Google Cloud Project ID]**"):
    GEMINI_PROJECT_ID = st.text_input("Google Cloud Project ID", type="password", value=os.environ.get("PROJECT_ID", ""))
    if not GEMINI_PROJECT_ID:
        os.environ["GEMINI_PROJECT_ID"] = GEMINI_PROJECT_ID

with st.popover("**:blue[Enter your Google Gemini API Key]**"):
    GEMINI_API_KEY = st.text_input("Google Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    if not GEMINI_API_KEY:
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
        st.info("""
        Instructions to obtain your Google Gemini API key will be added here once the API is publicly available.
        """)

if not GEMINI_API_KEY or not GEMINI_PROJECT_ID:
    st.error("Please enter both your Google Cloud Project ID and Gemini API key")
st.text("")
st.text("")
col1, buff, col2 = st.columns([0.3, 0.05, 0.65])

with col1:
    st.markdown("""### Select Prompt Engineering Skills""")
    with st.popover("**:blue[Learn more about this section]**"):
        st.info("For detailed information on each prompt engineering skill, click [here](https://arxiv.org/pdf/2312.16171.pdf)")
        st.warning("**Warning**: Some prompt engineering skills can conflict with each other and may not be applied effectively.  \ne.g. \"Impertavie Task\" and \"Penalty Warning\"")
    
    toggle_all = st.toggle("**Select/Unselect all skills**", key="master_toggle")
    
    with st.expander("**Prompt Structure and Clarity**", expanded=True):
        audience_integration = st.toggle("**Audience Integration**", value=toggle_all)
        st.markdown("Integrate the intended audience in the prompt.")
        affirmative_sentencing = st.toggle("**Affirmative Sentencing**", value=toggle_all)
        st.markdown("Employ affirmative directives such as 'do,' while steering clear of negative language like 'don’t'.")
        output_primers = st.toggle("**Output Primers**", value=toggle_all)
        st.markdown("Use output primers, which involve concluding the prompt with the beginning of the desired output.")
        delimiters = st.toggle("**Delimiters**", value=toggle_all) 
        st.markdown("Use delimiters to distinguish specific segments of text within the prompt.")
        formatted_prompt = st.toggle("**Formatted Prompt**", value=toggle_all) 
        st.markdown("Use Formatted prompt to allow the model to understand the requirements structurally.")

    with st.expander("**Specificity and Information**", expanded=True):
        fewshot_prompting = st.toggle("**Few-Shot Prompting**", value=toggle_all)
        st.markdown("Implement example-driven prompting.")
        guideline_indicators = st.toggle("**Guideline Indicators**", value=toggle_all)
        st.markdown("Clarify the requirements using keywords, regulations, hints, or instructions.")

    with st.expander("**Content and Language Style**", expanded=True): 
        no_politeness = st.toggle("**No Politeness**", value=toggle_all)
        st.markdown("If you prefer more concise answers, remove any unnecessary polite or indirect phrases in the prompt.")
        imperative_task = st.toggle("**Imperative Task**", value=toggle_all)
        st.markdown("Incorporate the following phrases: \"Your task is\" and \"You MUST\".")
        penalty_warning = st.toggle("**Penalty Warning**", value=toggle_all) 
        st.markdown("Incorporate the following phrases: \"You will be penalized\".")
        role_assignment = st.toggle("**Role Assignment**", value=toggle_all)
        st.markdown("Assign a specific role or persona to the model within the prompt.")
        echo_directive = st.toggle("**Echo Directive**", value=toggle_all)
        st.markdown("Repeat a specific word or phrase multiple times within a prompt.")
    
    with st.expander("**Complex Tasks**", expanded=True): 
        task_decomposition = st.toggle("**Task Decomposition**", value=toggle_all)
        st.markdown("Break down complex tasks into a sequence of simpler prompts in an interactive conversation.")
        # cot_with_fewshot = st.toggle("**CoT with Few-Shot**", value=toggle_all)
        # st.markdown("Combine Cot with few-shot prompts.")

    # skills order need to be optimized 
    skills_to_apply = {
        "no_politeness": no_politeness, 
        "affirmative_sentencing": affirmative_sentencing,  
        "audience_integration": audience_integration,
        "role_assignment": role_assignment,
        "penalty_warning": penalty_warning,
        "imperative_task": imperative_task,
        "guideline_indicators": guideline_indicators,
        "task_decomposition": task_decomposition,
        "fewshot_prompting": fewshot_prompting,
        # "cot_with_fewshot": cot_with_fewshot,  
        "echo_directive": echo_directive,
        "delimiters": delimiters,
        "formatted_prompt": formatted_prompt,
        "output_primers": output_primers,
    }


with col2:
    st.markdown("""### Enhance your prompt easily!""")
    st.text("")
    prompt = st.text_area("**Original Prompt**", placeholder="Enter your prompt here to receive an enhanced version.", height=300)

    phrases_to_insert = {} 
    st.text("")
    st.markdown("**How about including these magical phrases or words in your prompt?**\n\n**(이런 마법같은 문구나 단어를 프롬프트에 포함시키는 건 어떨까요?)**")
    step_by_step = st.checkbox("Take a deep breath and work on this step by step.")
    tipping = st.checkbox("I’m going to tip $200 for a better solution!")
    important_to_career = st.checkbox("This is very important to my career.")
    explain_beginner = st.checkbox("Explain to me as if I'm a beginner.")
    detailed_writing = st.checkbox("Write a detailed text for me by adding all the information necessary.")
    human_like_response = st.checkbox("Answer in a natural, human-like manner.")
    unbiased_response = st.checkbox("Ensure that your answer is unbiased and avoids relying on stereotypes.")
    
    phrases_to_insert = {
        "step_by_step": step_by_step,
        "tipping": tipping,
        "important_to_career": important_to_career,
        "explain_beginner": explain_beginner,
        "detailed_writing": detailed_writing,
        "human_like_response": human_like_response,
        "unbiased_response": unbiased_response,
    }

    st.text("")

    model_name = st.selectbox("**Select the model**", 
                              ("gemini-1.5-flash", "gemini-1.0-pro", "gemini-1.5-pro", "gemini-1.5-pro-exp-0801", "gemma-2-2b-it", "gemma-2-9b-it", "gemma-2-27b-it"), 
                              index=0, 
                              placeholder="Select the model...")

    st.text("") 

    lang_eng = st.checkbox("**Provide the enhanced prompt in English.** (If the original input is in another language but you need the prompt in English.)")
    simplified = st.checkbox("**Enhance the prompt with simplified instructions that save processing time and tokens.** (Best results with GPT-4 models)")
    st.text("")
    st.text("")

    enhance_btn = st.button("**:blue[Enhance!]**")

    if enhance_btn:
        if not prompt:
            st.toast("Please enter your prompt.")
        if not GEMINI_API_KEY or not GEMINI_PROJECT_ID:
            st.toast("Please enter both your Google Cloud Project ID and Gemini API Key.")
        
        order_num = 1
        with st.spinner("Processing..."):
            # llm = load_gemini_model(model_name)  # Load Gemini model

            if simplified:
                prompt = apply_skills_gemini(model_name, skills_to_apply, prompt, lang_eng)
                
                for phrase, checked in phrases_to_insert.items():
                    if checked:
                        prompt = insert[phrase] + '\n' + prompt
            else:
                for skill, toggled in skills_to_apply.items():
                    if toggled:
                        prompt = apply_skill_gemini(model_name, skill, prompt, order_num, lang_eng) 
                        st.markdown(f":zap: {order_num}. Your prompt has been enhanced with **\"{skill}\"**!")
                        container = st.container(border=True)
                        container.markdown(convert_newlines(prompt))
                        order_num += 1

                for phrase, checked in phrases_to_insert.items():
                    if checked:
                        st.markdown(f":zap: {order_num}. Your prompt has been enhanced with **\"{phrase}\"**!")
                        prompt = insert[phrase] + '\n' + prompt
                        container = st.container(border=True)
                        container.markdown(convert_newlines(prompt))
                        order_num += 1

        st.toast("Prompt Enhancement Success!")
        st.markdown("### Enhanced Prompt")
        container = st.container(border=True)
        container.title(":crystal_ball:")
        container.markdown(convert_newlines(prompt))

st.text("")
st.text("")
st.markdown("<p style='text-align: center;'>Github: @tsis-mobile-technology, (feat: gainworld@gmail.com)</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Github: @lim-hyo-jeong</p>(feat: @tsis-mobile-technology/Prompt-Enhancer)", unsafe_allow_html=True)