from gpt4all import GPT4All, Embed4All
from textblob import TextBlob
from rake_nltk import Rake
import pickle
import numpy as np
import json
import os

# Global Configuration

MODEL_NAME = 'C://AI_MODELS/llama2_7b_chat_uncensored.ggmlv3.q4_0.bin'
GUIDANCE_PROMPT = "Do not create lists or itemize, you have all the info you need to make a direct suggestion." # This prompt is used to guide the model to generate a response that is more likely to be a suggestion
TEMP = 0.8  # Adjust as needed
TOP_P = 0.9  #  It truncates the distribution of words to consider only the most probable words such that their cumulative probability exceeds a threshold (Top-p value). High values leading to more randomness
TOP_K = 60  # A smaller value of Top-k can make the output more focused. However, if set too low, it might also introduce some repetition
CHUNK_LIMIT = 500 # The maximum number of tokens allowed in a single chunk of text. This is used to split long texts into smaller chunks to avoid exceeding the model's token limit
KEYWORD_LIMIT = 10  # The maximum number of keywords to extract from a text. This is used to limit the number of keywords used to search the memory

## Utlity functions that are used to save and load data, chunk text, and extract keyword

def save_recorded_data(data, filename="output_data.json"):
    # If the file exists, load the data and append the new record.
    if os.path.exists(filename):
        with open(filename, "r") as json_file:
            existing_data = json.load(json_file)
        existing_data.append(data)
        final_data = existing_data
    else:
        final_data = [data]

    # Save the updated data back to the file.
    with open(filename, "w") as json_file:
        json.dump(final_data, json_file, indent=4)

recorded_data = {
    "user_input": None,
    "available_resources": None,
    "agents": []
}
    # Extract keywords from the summary
    # Analyze the sentiment of the summary
    # Create a dictionary to store the summary data
    # If the file exists, load the data and append the new summary.
    # Check if existing data is a list
    # If it's not a list, create a new list with the old and new data
    # Save the updated data back to the file.

def save_summary_data(summary, user_query, filename="summary_data.json"):

    summary_keywords = extract_keywords(summary)
    summary_sentiment = analyze_sentiment(summary)
    summary_data = {
        "original_user_query": user_query,
        "summary": summary,
        "keywords": summary_keywords,
        "sentiment": summary_sentiment
    }

    if os.path.exists(filename):
        with open(filename, "r") as json_file:
            existing_data = json.load(json_file)

        if isinstance(existing_data, list):
            existing_data.append(summary_data)
            final_data = existing_data
        else:  
            final_data = [existing_data, summary_data]
    else:
        final_data = [summary_data]


    with open(filename, "w") as json_file:
        json.dump(final_data, json_file, indent=4)

def save_data(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def chunk_text(text, max_tokens=CHUNK_LIMIT):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(current_chunk) + len(word) <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
    chunks.append(' '.join(current_chunk))  # add the last chunk

    return chunks

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def extract_keywords(text, limit=KEYWORD_LIMIT):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases()
    return ranked_phrases[:limit]  # Return only the top 'limit' ranked phrases

def generate_embedding(text):
    embedder = Embed4All()
    return embedder.embed(text)

# Memory Management

class Memory:
    def __init__(self):
        self.embeddings = []
        self.texts = []

    def add_memory(self, text, embedding):
        self.embeddings.append(embedding)
        self.texts.append(text)

    def search_memory(self, keyword_embedding, threshold=0.8, ignore_last=False):
        for i, stored_embedding in enumerate(self.embeddings):
            if ignore_last and i == len(self.embeddings) - 1:  # Ignore the last response if the flag is set
                continue
            similarity = np.dot(stored_embedding, keyword_embedding) / (np.linalg.norm(stored_embedding) * np.linalg.norm(keyword_embedding))
            if similarity > threshold:
                return self.texts[i]
        return None
    
memory = Memory()
conversation_history = []

# Load stored memories
try:
    memory_data = load_data('memory.pkl')
    memory.embeddings = memory_data['embeddings']
    memory.texts = memory_data['texts']
except:
    pass  # If there's no file or another error, just move on



# Main answering logic and chat agent, which is the main class that handles the conversation

class ChatAgent:
    def __init__(self):
        self.model = GPT4All(model_name=MODEL_NAME)
        self.embedder = Embed4All()

    def generate_embedding(self, text):
        return self.embedder.embed(text)
    
    def get_salient_points(self, text):
        keywords = extract_keywords(text)
        return list(set(keywords))  # Using set to remove duplicates
    
    def generate_summary(self, conversation_text, salient_points=None):
        if salient_points is None:
            salient_points = self.get_salient_points(conversation_text)
        points = ', '.join(salient_points)
        
        # Chunk the conversation_text
        chunks = chunk_text(conversation_text, max_tokens=400)  # Adjust max_tokens as needed
        summarized_chunks = []

        for chunk in chunks:
            summary_prompt = f"Based on the following salient points: {points}, summarize the conversation: {chunk}"
            chunk_summary = self.model.generate(prompt=summary_prompt, temp=TEMP, top_p=TOP_P, top_k=TOP_K, max_tokens=1000)
            summarized_chunks.append(chunk_summary)

        # Combine the summarized chunks
        combined_summary = ' '.join(summarized_chunks)
        return combined_summary

    
    def process_input(self, user_input, sequence, available_resources):

        responses = []  # List to store each agent's response
        output = user_input  # Start with the user's input
        # Analyze sentiment and extract keywords from user input
        input_sentiment = analyze_sentiment(user_input)
        input_keywords = extract_keywords(user_input)

        for keyword in input_keywords:
            keyword_embedding = generate_embedding(keyword)
            matched_text = memory.search_memory(keyword_embedding, ignore_last=True)
            if matched_text:
                output += " " + matched_text


        print(f"User Input: {user_input}")
        print(f"Input Sentiment: {input_sentiment}")
        print(f"Input Keywords: {input_keywords}\n")

        for agent in sequence:
            index = ord(agent) - ord('a')  # Convert letter to index
            if index < len(prompts):
                prompt_text, prompt_query, complexity = prompts[index]
                token_limit = token_limits[complexity]
                query = prompt_query.format(user_query=output, available_resources=available_resources)
                aligned_prompt = f"Framework: {prompt_text}. {query} {GUIDANCE_PROMPT}"
                response = self.generate_response(query, token_limit)
                responses.append(response)  # Append the response to the list
                conversation_history.append(response)
                output += f" Agent {agent.upper()}'s suggestion: {response}"
                response_sentiment = analyze_sentiment(response)
                response_keywords = extract_keywords(response)
                print(f"Agent {agent.upper()} Response: {response}")
                print(f"Agent {agent.upper()} Response Sentiment: {response_sentiment}")
                print(f"Agent {agent.upper()} Response Keywords: {response_keywords}\n")

        # Add interactions to memory
        memory.add_memory(user_input, generate_embedding(user_input))

        for response in responses:
            memory.add_memory(response, self.generate_embedding(response))

        # Save the updated memory
        save_data('memory.pkl', {'embeddings': memory.embeddings, 'texts': memory.texts})
        
        return responses

    def generate_response(self, query, token_limit):
        chunks = chunk_text(query, token_limit)
        responses = []

        for chunk in chunks:
            with self.model.chat_session():
                aligned_prompt = chunk + " " + GUIDANCE_PROMPT
                response = self.model.generate(prompt=aligned_prompt, temp=TEMP, top_p=TOP_P, top_k=TOP_K, max_tokens=token_limit)
                responses.append(response)

        return ' '.join(responses)



# Define the prompts and their associated complexities
prompts = [
    ("Second-Order Thinking", "Using {available_resources}, consider both immediate and second-level consequences of {user_query}.", "medium"),
    ("Pareto Principle (80/20 Rule)", "With {available_resources}, focus on the 20% of factors that could lead to 80% of the results of {user_query}.", "short"),
    ("First Principle Thinking", "Using {available_resources}, rethink {user_query} from the ground up, separating facts from assumptions.", "medium"),
    ("Regret Minimization Framework", "Considering {available_resources}, think long-term about {user_query} and its emotional impact.", "medium"),
    ("Opportunity Costs", "With {available_resources}, think about what you might give up by choosing {user_query}.", "medium"),
    ("The Sunk Cost Fallacy", "Using {available_resources}, evaluate {user_query} based on its future value, not past investments.", "medium"),
    ("Occam's Razor", "With {available_resources}, find the simplest explanation for {user_query}.", "short"),
    ("Systems Thinking", "Using {available_resources}, see how {user_query} fits into a larger system.", "medium"),
    ("Inversion", "Considering {available_resources}, think about {user_query} from the end and its potential pitfalls.", "medium"),
    ("Leverage", "Using {available_resources}, see how leverage can amplify the effect of {user_query}.", "short"),
    ("Circle of Competence", "With {available_resources}, ensure {user_query} fits within your expertise.", "medium"),
    ("Law of Diminishing Returns", "Considering {available_resources}, see where more effort in {user_query} might yield less value.", "medium"),
    ("Niches", "Using {available_resources}, determine how {user_query} fits into specialized niches.", "short"),
    ("Margin of Safety", "With {available_resources}, ensure a safety margin for {user_query}.", "medium"),
    ("Hanlon's Razor", "Using {available_resources}, interpret {user_query} without assuming ill intent.", "short"),
    ("Randomness", "Considering {available_resources}, think about the randomness affecting {user_query}.", "medium"),
    ("Critical Mass", "With {available_resources}, identify when {user_query} reaches a self-sustaining point.", "short"),
    ("The Halo Effect", "Using {available_resources}, think about how first impressions might affect views on {user_query}.", "medium"),
    ("Feedback Loops", "With {available_resources}, determine the feedback loops present in {user_query}.", "medium"),
    ("Scarcity and Abundance Mindset", "Using {available_resources}, reflect on scarcity or abundance in relation to {user_query}.", "long")
]

# Token limits based on complexity
token_limits = {
    "short": 100,
    "medium": 200,
    "long": 300
}



def print_availableframes():
    print("\nHere are the available reasoning frameworks and their associated letters:\n")
    
    for i, (name, _, _) in enumerate(prompts):
        letter = chr(i + ord('a'))  # Convert index to its corresponding letter (a, b, c, ...)
        print(f"{letter}. {name}")




agent = ChatAgent()

user_input = input("Enter your query: ")
recorded_data["user_input"] = user_input

# Prompt the user for their available resources
available_resources = input("Please specify your available resources: ")
recorded_data["available_resources"] = available_resources
print_availableframes()
sequence = input("Enter the sequence of agents (e.g., 'aabc'): ")

# Update the process_input function call to pass the available resources
responses = agent.process_input(user_input, sequence, available_resources)

# Loop through the responses to record and print them
for i, response in enumerate(responses, 1):
    agent_data = {
        "name": sequence[i-1].upper(),
        "prompt": prompts[ord(sequence[i-1]) - ord('a')][0],
        "response": response,
        "sentiment": analyze_sentiment(response),
        "keywords": extract_keywords(response)
    }
    recorded_data["agents"].append(agent_data)
    print(f"\nAgent {sequence[i-1].upper()} Response: {response}\n")

# Save all the recorded data form the conversation
save_recorded_data(recorded_data)
    
# Summarize the conversation
summary_agent_response = agent.generate_summary(' '.join(conversation_history))
print(f"\nSummary Agent Response: {summary_agent_response}\n")
# Save the summary agent response and the user input
save_summary_data(summary_agent_response, recorded_data["user_input"])
