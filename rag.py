import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def main():
    st.set_page_config(layout="wide")
    st.subheader("Système de Question Réponse Intelligent ", divider="rainbow")

    with st.sidebar:
        st.sidebar.title("Data Loader")
        st.image("rag.png", width=500)

        pdf_docs = st.file_uploader("Upload Your PDFs", accept_multiple_files=True)

        if st.button("Submit"):
            with st.spinner("Loading..."):
                pdf_content = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        pdf_content += page.extract_text() or ""

                # Split content into chunks
                splitter = CharacterTextSplitter(
                    separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
                )
                chunks = splitter.split_text(pdf_content)
                st.write(chunks)

                # Initialize Gemini embeddings
                GEMINI_API_KEY = "AIzaSyDB6KFmhslzsSJCpZL2F0UGV3IhuK0L_pQ"
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

                # Create vector store
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)

                # LLM model
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

                # Prompt template
                
                prompt = ChatPromptTemplate.from_template("""
                Vous êtes un assistant d'IA utile, précis et honnête.

                <instructions>
                -vous devrez ecrire une phrase pour encourager l'utilisateur à poser une question.par exemple :bon question voila la reponse , D'accord je veux repondre a ta question ?  
                - Répondez à la question uniquement en vous basant sur le contexte fourni ci-dessous.
                - Si l'information n'est pas présente dans le contexte, indiquez "Je ne trouve pas cette information dans le contexte fourni" au lieu de deviner.
                - Gardez votre réponse concise et directement liée à la question.
                - Citez les parties pertinentes du contexte pour justifier votre réponse quand c'est approprié.
                - N'inventez pas d'information qui n'est pas explicitement mentionnée dans le contexte.
                </instructions>

                <contexte>
                {context}
                </contexte>

                Question: {input}
                """)

                # Create chain
                doc_chain = create_stuff_documents_chain(llm, prompt)
                retriever = vector_store.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, doc_chain)

                st.session_state.retrieve_chain = retrieval_chain

    # Chat Interface
    st.subheader("Chatbot zone")
    user_question = st.text_input("Ask your question:", key="user_input")
    if st.button("Send"):
        if user_question:
            response = st.session_state.retrieve_chain.invoke({"input": user_question})
            st.markdown(response["answer"], unsafe_allow_html=True)


if __name__ == "__main__":
    main()
