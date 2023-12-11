from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from streamlit_js_eval import streamlit_js_eval
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import streamlit as st
import openai
import streamlit.components.v1 as components
import re
import plotly.graph_objects as go
import os



st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="💬")
st.title("HR interview Simulator - Alpha Release")

if 'report_button_clicked' not in st.session_state:
    st.session_state.report_button_clicked = False


def create_pie_chart(score):
    # Calculate the remaining part of the chart
    remaining = 10 - score

    fig = go.Figure(data=[go.Pie(labels=["Your Score", "Full Score"],
                                 values=[score, remaining],
                                 hole=.7,
                                 marker_colors=["#56bfc7", "#1c4657"])])
    
    # Update chart appearance
    fig.update_layout(showlegend=False, 
                      annotations=[dict(text=f'{score}/10', x=0.5, y=0.5, font_size=20, showarrow=False)])
    
    return fig

def generate_report():
    st.session_state.report_button_clicked = True
    with st.spinner('Generating report... Please wait'):
        memory_str = str(memory)
        memory_str2 = str(msgs.messages)

            # Initialize another OpenAI instance for feedback generation
        llm2 = OpenAI(temperature=0.5, openai_api_key=openai.api_key, model_name="gpt-4-1106-preview")

            # Prepare the prompt for feedback generation
        prompt_text = f"""You are a helpful tool that evaluates the performance of an interviewee based on the given context. Write a feedback. Be honest even if it means to be critical. Also be very critical about short or yes and no answers. Recommend how to improve. Give an overall score and write the score at the end. Also at the end rate the interviewee on the following criteria from 1 to 10 - Content Relevance and Depth, Professionalism, Communication Skills, Critical Thinking and Problem-Solving. Use only whole numbers for the rating. Use the following template at the end so that this text can be extracted by programmers: Overall Score: <your score> Content Relevance and Depth: <your score>, Professionalism: <your score>, Communication Skills: <your score>, Critical Thinking and Problem-Solving: <your score> do not write anything after the scores. This is the script: {memory_str2}"""

            # Generate feedback
        feedback = llm2.predict(prompt_text)

            #search for the overall score
        match = re.search(r"Overall Score: (\d+)", feedback)
        if match:
            score = int(match.group(1))
        else:
            score = 0  # Default score if not found

            # Store the feedback in session state to persist it
        st.session_state['feedback'] = feedback
        st.session_state['score'] = score
        st.session_state['show_progress'] = True


"""
✔️ Experiment with the tool and explore its current capabilities.

✔️ Provide us with feedback about your experience, suggestions for improvements, and any issues you encounter.

✔️ Keep in mind the developmental nature of this Alpha release as we work towards a more robust and feature-rich version.


Thank you for being a part of this exciting journey in technology. Your participation and feedback are instrumental in shaping the future of HR interviews! 🚀🤖👥
"""
components.html("""<hr style="height:3px;border:none;color:#ffffff;background-color:#fff;" /> """)

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)

if "tech_messages" not in st.session_state:
    st.session_state.tech_messages = []

# Get an OpenAI API Key before continuing
openai.api_key = st.secrets["OPEN_AI_KEY"]
openai_api_key = openai.api_key



# Set up the LLMChain, passing in memory
template = """ You are an HR executive for the company Amazon you are interviewing the user for the position Junior Data Scientist as an HR. Ask one question at a time! And keep the interview going, do not end the interview. You can also ask questions relevant to the position the candidate is applying for. Do not write things like System:, or AI: in front of your response. Use the following structure for the interview: start with simple HR questions, then ask behavioral questions, and finally theoretical questions. This questions were used in previous interviews: How do you handle working with stress?, What Is Hypothesis Testing?.
{history}
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
llm_chain = LLMChain(llm=OpenAI(openai_api_key=openai_api_key, model_name="gpt-4-1106-preview"), prompt=prompt, memory=memory)


max_user_inputs = 6

if "num_user_inputs" not in st.session_state:
    st.session_state.num_user_inputs = len(st.session_state['tech_messages']) // 2


if "initialized" not in st.session_state or not st.session_state.initialized:
    msgs.add_ai_message("Hello, and welcome to this interview. I'm excited to learn more about you and your qualifications. To start, could you please introduce yourself and give us a brief overview of your background and experience?")
    st.session_state.initialized = True

# Display the messages
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Check if the number of messages is less than the max_messages and the interview hasn't ended
if st.session_state.num_user_inputs < max_user_inputs:
    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)

        # Update the messages in the state
        st.session_state['tech_messages'].append({'type': 'human', 'content': prompt})

        # Increment the number of user inputs
        st.session_state.num_user_inputs += 1

        # If it's not the last user input, generate and display AI response
        if st.session_state.num_user_inputs < max_user_inputs:
            response = llm_chain.run(prompt)
            st.chat_message("ai").write(response)
            st.session_state['tech_messages'].append({'type': 'ai', 'content': response})


# If the max number of messages has been reached and the interview has not been marked as ended
if st.session_state.num_user_inputs >= max_user_inputs and not st.session_state.get('interview_ended', False):
    with st.spinner('Ending the interview... Please wait'):
                response = llm_chain.run(prompt)
    st.chat_message("ai").write("Thank you for the interview. We will get back to you soon.")
    st.session_state['interview_ended'] = True

# Display an info message after the interview has ended
if st.session_state.get('interview_ended', False):
    st.info("The interview has ended. Thank you for participating.")


# Display feedback button and handle its click
if st.session_state.get('interview_ended', False) and not st.session_state.report_button_clicked:
    if st.button("Get report", key='report', on_click=generate_report):
        # The button will call the generate_report function when clicked
        pass  # No additional actions needed here

# Check if feedback has been generated and display it
if 'feedback' in st.session_state:
    
    match = re.search(r"Overall Score: (\d+)", st.session_state['feedback'])
    if match:
        score = int(match.group(1))
    else:
        score = 0  # Default score if not found
    
    pie_chart = create_pie_chart(score)
    st.plotly_chart(pie_chart, use_container_width=True)
    
    st.write("Interview Report:")
    st.write(st.session_state['feedback'])
    
    st.text("")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        st.text("")

    with col2:
        if st.button("Redo Interview", type="primary"):
            streamlit_js_eval(js_expressions="parent.window.location.reload()")

    with col3:
        st.link_button("Give us Feedback", "https://forms.gle/xLf5RtkkX9nHzWir5")
    
    with col4:
        st.text("")


