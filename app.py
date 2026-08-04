# ============ STEP 1: LOAD MODULES ============

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import numpy as np
import time
from PIL import Image
from dotenv import load_dotenv

#=================STEP 2 API KEYS======================
st.set_page_config(page_title = "Chat-With-PDF",
                   layout = "wide")
st.sidebar.title("SET API CONFIG")
st.title("RAG based Chat With PDF")
GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY",type = "password")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY




