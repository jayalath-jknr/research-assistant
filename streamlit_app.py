import validators, streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import os
import time
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, ChatVertexAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


from trulens_eval import Feedback, Huggingface, Tru, TruChain
from trulens_eval.feedback.provider.hugs import Huggingface

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

hugs = Huggingface()
tru = Tru()

f_lang_match = Feedback(hugs.language_match).on_input_output()
feedback_nontoxic = Feedback(huggingface_provider.not_toxic).on_output()
f_pii_detection = Feedback(hugs.pii_detection).on_input()
feedback_positive = Feedback(huggingface_provider.positive_sentiment).on_output()

# TruLens chain recorder
chain_recorder = TruChain(
    conversation,
    app_id="contextual-chatbot",
    feedbacks=[f_lang_match, feedback_nontoxic, f_pii_detection, feedback_positive],
)
# Streamlit app
st.subheader('Summarize URL')

# Get OpenAI API key and URL to be summarized
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", value="", type="password")
    st.caption("*If you don't have an OpenAI API key, get it [here](https://platform.openai.com/account/api-keys).*")
    model = st.selectbox("OpenAI chat model", ("gpt-3.5-turbo", "gpt-3.5-turbo-16k"))
    st.caption("*If the article is long, choose gpt-3.5-turbo-16k.*")
url = st.text_input("URL", label_visibility="collapsed")

# If 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not openai_api_key.strip() or not url.strip():
        st.error("Please provide the missing fields.")
    elif not validators.url(url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Please wait..."):
                # Load URL data
                loader = UnstructuredURLLoader(urls=[url])
                data = loader.load()
                
                # Initialize the ChatOpenAI module, load and run the summarize chain
                llm = ChatOpenAI(temperature=0, model=model, openai_api_key=openai_api_key)
               
                prompt_template = """Write a summary of the following in 250-300 words:
                
                    {text}

                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(data)

                st.success(summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
