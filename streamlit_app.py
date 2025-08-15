#!/usr/bin/env python3
"""
Streamlit MCP Chat Interface with Pinecone Database Integration

A beautiful web interface for your MCP chat agent with persistent memory storage.
"""

import streamlit as st
import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
import dotenv
import pathlib
import hashlib

# Load environment variables from project root or script directory
dotenv.load_dotenv(pathlib.Path(__file__).parent / ".env")
dotenv.load_dotenv(".env")  # fallback if running from root

class PineconeMemoryManager:
    """Manages conversation memory using Pinecone vector database."""
    
    def __init__(self, api_key: str, index_name: str = "car-data-index"):
        """Initialize Pinecone memory manager."""
        self.index_name = index_name
        self.dimension = 1024  # MUST match everywhere

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)

        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        self.index = self.pc.Index(self.index_name)
    
    def add_conversation(self, session_id: str, user_message: str, ai_response: str, timestamp: str):
        """Add a conversation turn to Pinecone."""
        turn_id = f"{session_id}_{timestamp}"
        
        metadata = {
            "session_id": session_id,
            "user_message": user_message,
            "ai_response": ai_response,
            "timestamp": timestamp,
            "type": "conversation_turn"
        }
        
        # Dummy vector (replace with embeddings in production)
        vector_value = float(int(hashlib.md5(f"{user_message}{ai_response}".encode()).hexdigest()[:8], 16)) / 1e8
        vector = [vector_value] * self.dimension
        
        self.index.upsert(vectors=[(turn_id, vector, metadata)])
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a session."""
        try:
            results = self.index.query(
                vector=[0] * self.dimension,
                filter={"session_id": session_id},
                top_k=limit,
                include_metadata=True
            )
            
            conversations = [
                match.metadata for match in results.matches if match.metadata
            ]
            
            conversations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return conversations
            
        except Exception as e:
            st.error(f"Error retrieving conversation history: {e}")
            return []


class StreamlitMCPChat:
    """Streamlit interface for MCP Chat Agent with Pinecone memory."""
    
    def __init__(self):
        self.setup_page()
        self.initialize_session_state()
        self.setup_pinecone()
        self.setup_mcp()
    
    def setup_page(self):
        st.set_page_config(
            page_title="MCP Chat Agent",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for Cursor AI-like dark theme
        st.markdown("""
        <style>
        /* Main background */
        .main .block-container {
            background-color: #1E1E1E;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Chat messages styling */
        .stChatMessage {
            background-color: #252526 !important;
            border-radius: 8px;
            margin: 8px 0;
            padding: 12px;
            border-left: 3px solid #007ACC;
        }
        
        /* User message styling */
        .stChatMessage[data-testid="chatMessage"]:has(.stChatMessage__avatar[data-testid="user"]) {
            background-color: #2D2D30 !important;
            border-left-color: #007ACC;
        }
        
        /* Assistant message styling */
        .stChatMessage[data-testid="chatMessage"]:has(.stChatMessage__avatar[data-testid="assistant"]) {
            background-color: #252526 !important;
            border-left-color: #4EC9B0;
        }
        
        /* Chat input styling */
        .stChatInput {
            background-color: #252526 !important;
            border: 1px solid #3E3E42;
            border-radius: 8px;
        }
        
        .stChatInput input {
            background-color: #252526 !important;
            color: #CCCCCC !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #252526 !important;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #007ACC !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
            font-weight: 500 !important;
        }
        
        .stButton > button:hover {
            background-color: #005A9E !important;
            box-shadow: 0 2px 8px rgba(0, 122, 204, 0.3) !important;
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: #CCCCCC !important;
        }
        
        /* Text styling */
        .stMarkdown {
            color: #CCCCCC !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #2D2D30 !important;
            color: #CCCCCC !important;
            border-radius: 6px !important;
        }
        
        /* Info boxes */
        .stAlert {
            background-color: #2D2D30 !important;
            border: 1px solid #3E3E42 !important;
        }
        
        /* Success/Error messages */
        .stAlert[data-baseweb="notification"] {
            background-color: #2D2D30 !important;
            border: 1px solid #3E3E42 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("🤖 MCP Chat Agent with Pinecone Memory")
        st.markdown("---")
    
    def initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
    
    def setup_pinecone(self):
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not pinecone_api_key:
            st.error("❌ Pinecone API key missing in .env file")
            self.memory_manager = None
        else:
            try:
                self.memory_manager = PineconeMemoryManager(pinecone_api_key)
                st.success("✅ Connected to Pinecone database")
            except Exception as e:
                st.error(f"❌ Failed to connect to Pinecone: {e}")
                self.memory_manager = None
    
    def setup_mcp(self):
        try:
            if not os.path.exists("browser_mcp.json"):
                st.error("❌ MCP config file (browser_mcp.json) not found")
                self.mcp_servers = {}
                return
            
            with open("browser_mcp.json", 'r') as f:
                config = json.load(f)
                self.mcp_servers = config.get("mcpServers", {})
            
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                st.error("❌ GROQ_API_KEY missing in .env")
                self.llm = None
                return
            
            self.llm = Groq(api_key=groq_api_key)
            st.success(f"✅ Loaded {len(self.mcp_servers)} MCP servers: {', '.join(self.mcp_servers.keys())}")
        except Exception as e:
            st.error(f"❌ Error setting up MCP: {e}")
            self.mcp_servers = {}
            self.llm = None
    
    def display_sidebar(self):
        with st.sidebar:
            st.markdown("""
            <style>
            .sidebar .sidebar-content {
                background-color: #252526;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.header("🎛️ Controls")
            st.subheader("Session Info")
            st.text(f"Session ID: {st.session_state.session_id[:8]}...")
            st.text(f"Messages: {len(st.session_state.messages)}")
            
            st.subheader("🔌 MCP Servers")
            for server_name in self.mcp_servers:
                st.text(f"• {server_name}")
            
            st.subheader("💾 Memory")
            if st.button("Clear Session Memory"):
                st.session_state.messages = []
                st.rerun()
            
            if st.button("New Session"):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.rerun()
            
            st.subheader("🗄️ Database")
            if self.memory_manager:
                st.success("Pinecone Connected")
                
                # Add button to view stored conversations
                if st.button("📚 View Stored Conversations"):
                    st.session_state.show_all_conversations = True
                    st.rerun()
                    
                # Add button to view current session history
                if st.button("📖 Current Session History"):
                    st.session_state.show_session_history = True
                    st.rerun()
            else:
                st.error("Pinecone Disconnected")
    
    def generate_response(self, user_input: str) -> str:
        if not self.llm:
            return "❌ Groq client not initialized."
        
        context = "Recent conversation:\n"
        for msg in st.session_state.messages[-5:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"
        
        system_prompt = f"""You are a helpful AI assistant with access to MCP servers.
You have access to: {list(self.mcp_servers.keys())}

{context}

Current user input: {user_input}"""

        try:
            response = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": system_prompt}],
                model="llama3-8b-8192",
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def save_to_pinecone(self, user_message: str, ai_response: str):
        if not self.memory_manager:
            return
        timestamp = datetime.now().isoformat()
        self.memory_manager.add_conversation(
            st.session_state.session_id,
            user_message,
            ai_response,
            timestamp
        )
    
    def display_chat_interface(self):
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        if prompt := st.chat_input("Type your message here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    response = self.generate_response(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    self.save_to_pinecone(prompt, response)
    
    def show_stored_conversations(self):
        """Display all stored conversations from Pinecone."""
        if not self.memory_manager:
            st.error("❌ Pinecone not connected")
            return
        
        st.subheader("📚 All Stored Conversations")
        
        try:
            # Get all conversations (simplified query)
            results = self.memory_manager.index.query(
                vector=[0] * self.memory_manager.dimension,
                top_k=100,
                include_metadata=True
            )
            
            if results.matches:
                st.info(f"Found {len(results.matches)} conversation turns")
                
                # Group by session
                sessions = {}
                for match in results.matches:
                    if match.metadata:
                        session_id = match.metadata.get('session_id', 'Unknown')
                        if session_id not in sessions:
                            sessions[session_id] = []
                        sessions[session_id].append(match.metadata)
                
                # Display each session
                for session_id, conversations in sessions.items():
                    with st.expander(f"Session: {session_id[:8]}... ({len(conversations)} turns)", expanded=False):
                        # Sort conversations by timestamp
                        conversations.sort(key=lambda x: x.get('timestamp', ''))
                        
                        for conv in conversations:
                            st.markdown(f"**User:** {conv.get('user_message', 'N/A')}")
                            st.markdown(f"**AI:** {conv.get('ai_response', 'N/A')}")
                            st.markdown(f"*{conv.get('timestamp', 'N/A')}*")
                            st.markdown("---")
            else:
                st.info("No conversations found in database.")
                
        except Exception as e:
            st.error(f"Error retrieving conversations: {e}")
    
    def show_current_session_history(self):
        """Display current session's conversation history from Pinecone."""
        if not self.memory_manager:
            st.error("❌ Pinecone not connected")
            return
        
        st.subheader(f"📖 Current Session History: {st.session_state.session_id[:8]}...")
        
        try:
            conversations = self.memory_manager.get_conversation_history(
                st.session_state.session_id, 
                limit=50
            )
            
            if conversations:
                st.info(f"Found {len(conversations)} conversation turns for current session")
                
                for conv in conversations:
                    st.markdown(f"**User:** {conv.get('user_message', 'N/A')}")
                    st.markdown(f"**AI:** {conv.get('ai_response', 'N/A')}")
                    st.markdown(f"*{conv.get('timestamp', 'N/A')}*")
                    st.markdown("---")
            else:
                st.info("No conversation history found for current session.")
                
        except Exception as e:
            st.error(f"Error retrieving session history: {e}")
    
    def run(self):
        self.display_sidebar()
        
        # Handle display modes
        if getattr(st.session_state, 'show_all_conversations', False):
            self.show_stored_conversations()
            if st.button("🔙 Back to Chat"):
                st.session_state.show_all_conversations = False
                st.rerun()
        elif getattr(st.session_state, 'show_session_history', False):
            self.show_current_session_history()
            if st.button("🔙 Back to Chat"):
                st.session_state.show_session_history = False
                st.rerun()
        else:
            st.header("💬 Chat with MCP Agent")
            self.display_chat_interface()


def main():
    try:
        app = StreamlitMCPChat()
        app.run()
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()