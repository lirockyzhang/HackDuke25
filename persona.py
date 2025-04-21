# -*- coding: utf-8 -*-
"""
Persona class for handling different LLM models in a conversational AI application.
Supports multiple providers: OpenAI, Google Gemini, and Ollama.
"""

import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

class Persona:
    """
    Persona class to encapsulate the LLM processing logic.
    This class handles the conversation with Google's Gemini model.
    """
    
    def __init__(self, temperature=0.7):
        """
        Initialize the Persona with Google's Gemini model.
        
        Args:
            temperature (float): The temperature parameter for generation (0.0-1.0).
        """
        # Load environment variables
        load_dotenv()
        
        self.temperature = temperature
        self.conversation = None
        self.llm = None
        self.memory = None
        self.error = None
        
        # Initialize the LLM and conversation
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Gemini LLM."""
        try:
            # Check for Google API key
            if not os.getenv("GOOGLE_API_KEY"):
                self.error = "GOOGLE_API_KEY not found in environment variables"
                return False
            
            # Create the Gemini LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-001",
                temperature=self.temperature
            )
            
            # Create memory and conversation chain
            self.memory = ConversationBufferMemory()
            self.conversation = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                verbose=False
            )
            
            return True
            
        except Exception as e:
            self.error = f"Error initializing LLM: {str(e)}"
            return False
    
    def generate_response(self, user_input):
        """
        Generate a response to the user input.
        
        Args:
            user_input (str): The user's input text.
            
        Returns:
            str: The AI-generated response text.
            
        Raises:
            Exception: If there's an error during response generation.
        """
        if not self.conversation:
            raise Exception(f"LLM not initialized: {self.error}")
        
        # Process the input and generate a response
        response = self.conversation.predict(input=user_input)
        return response
    
    def is_ready(self):
        """Check if the Persona is properly initialized."""
        return self.conversation is not None
    
    def get_error(self):
        """Get the last error message if initialization failed."""
        return self.error