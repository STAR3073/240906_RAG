import os
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from rag import rag_setup, create_rag_chain



# 프로젝트 이름 설정
st.title("PDF 기반 질의응답 챗봇")



# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


# 대화 기록이 없다면, chat_history 라는 키로 빈 대화를 저장하는 list 를 생성
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# chain 을 초기화
if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None

# 대화 기록에 채팅을 추가
def add_history(role, message):
    st.session_state["chat_history"].append(ChatMessage(role=role, content=message))

def print_history():
    for chat_message in st.session_state["chat_history"]:
        # 메시지 출력(role: 누가 말한 메시지 인가?) .write(content: 메시지 내용)
        st.chat_message(chat_message.role).write(chat_message.content)




# PDF 파일 업로드 URL 생성
with st.sidebar:
    uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    retriever = rag_setup(file_path, chunk_size=300, chunk_overlap=50)

    return retriever


# 파일 업로드 되었을 때
if uploaded_file:
    retriever = embed_file(uploaded_file)
    st.session_state["rag_chain"] = create_rag_chain(retriever)

# 이전까지의 대화를 출력
print_history()


user_input = st.chat_input("궁금한 내용을 입력해 주세요")

if user_input:
    # 파일이 업로드가 된 이후
    if st.session_state["rag_chain"]:
        rag_chain = st.session_state["rag_chain"]

        # 사용자의 질문을 출력
        st.chat_message("user").write(user_input)

        # AI의 질문을 출력
        with st.chat_message("ai"):
            # 답변을 출력할 빈 공간을 만든다.
            chat_container = st.empty()

            # 사용자가 질문을 입력하면, 체인에 질문을 넣고 실행합니다.
            answer = rag_chain.stream(user_input)

            # 스트리밍 출력
            ai_answer = ""
            for token in answer:
                ai_answer += token
                chat_container.markdown(ai_answer)

        # 대화 기록에 추가
        add_history("user", user_input)
        add_history("ai", ai_answer)

    else:
        # 파일이 업로드 되지 않았을 때
        st.warning("PDF 파일을 업로드 해주세요.")


