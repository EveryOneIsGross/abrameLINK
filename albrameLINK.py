from gpt4all import GPT4All, Embed4All
from textblob import TextBlob
from rake_nltk import Rake
import pickle
import numpy as np
import json
import os
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Global Configuration
MODEL_NAME = 'C://AI_MODELS//openorca-platypus2-13b.ggmlv3.q4_1.bin'
#GUIDANCE_PROMPT = "Provide a unique and insightful answer based on the context. Avoid repeating previous responses."
# Verbose: "Your objective is to provide a distinct and insightful answer, grounded in the context provided. Ensure that you are not reiterating previous responses or commonly known facts."
GUIDANCE_PROMPT = "Give a fresh and relevant answer based on the context."
TEMP = 0.7 # TEMP is the temperature of the sampling. It ranges from 0 to 1.0
TOP_P = 0.7 # TOP_P is the cumulative probability of the most likely tokens to sample from it ranges from 0 to 1.0
TOP_K = 80 # TOP_K is the number of the most likely  tokens to sample from it ranges from 0 to infinity
CHUNK_LIMIT = 512
KEYWORD_LIMIT = 6

# Global Variables
global_conversation_history = []
max_tokens_for_context = 1024
tokens = " ".join(global_conversation_history).split()
output = ""
reasoning_used = []
conversation_summaries = []
conversation_history = []


# Define the frameworks and their associated logic
prompts = [
    ("Second-Order Thinking", "Given {available_resources}, first identify the immediate outcomes of {user_query}. Then, dive deeper to discern the secondary, less obvious consequences.", "medium"),
    ("Pareto Principle (80/20 Rule)", "Employing {available_resources}, concentrate on the crucial '20%' of factors that will yield '80%' of the outcomes for {user_query}.", "short"),
    ("First Principle Thinking", "Utilizing {available_resources}, break down {user_query} to its foundational principles. Discern between what's factual and what's assumed.", "medium"),
    ("Regret Minimization Framework", "With {available_resources} in mind, project long-term implications of {user_query}, especially focusing on potential regrets and emotional repercussions.", "medium"),
    ("Opportunity Costs", "Given {available_resources}, contemplate the alternatives you'll forego by opting for {user_query}.", "medium"),
    ("The Sunk Cost Fallacy", "Utilizing {available_resources}, analyze {user_query} by emphasizing its prospective benefits over the costs already incurred.", "medium"),
    ("Occam's Razor", "With {available_resources}, distill {user_query} to its simplest form or explanation, eliminating unnecessary complexities.", "short"),
    ("Systems Thinking", "Using {available_resources}, map out where {user_query} slots into broader systems or networks.", "short"),
    ("Inversion", "Keeping {available_resources} in mind, reverse-engineer {user_query}, highlighting potential challenges and pitfalls.", "medium"),
    ("Leverage", "Employing {available_resources}, explore how you can use leverage to maximize the outcomes of {user_query}.", "short"),
    ("Circle of Competence", "Given {available_resources}, verify that {user_query} aligns well with your domain of expertise and knowledge.", "medium"),
    ("Law of Diminishing Returns", "With {available_resources}, pinpoint areas in {user_query} where additional efforts might produce diminishing outcomes.", "medium"),
    ("Niches", "Using {available_resources}, categorize how {user_query} can be tailored or adapted to cater to specific niches or subgroups.", "short"),
    ("Margin of Safety", "With {available_resources}, strategize to incorporate a buffer or safety net around the outcomes of {user_query}.", "medium"),
    ("Hanlon's Razor", "Given {available_resources}, approach {user_query} with neutrality, avoiding assumptions of malevolence or negative intent.", "short"),
    ("Randomness", "Considering {available_resources}, factor in random elements of unpredictability and chance that could influence {user_query}.", "medium"),
    ("Critical Mass", "Utilizing {available_resources}, determine the tipping point at which {user_query} becomes self-sustaining or gains momentum.", "short"),
    ("The Halo Effect", "Employing {available_resources}, scrutinize how initial perceptions or biases might color subsequent interpretations of {user_query}.", "medium"),
    ("Feedback Loops", "With {available_resources}, trace the cyclic patterns or recurring themes that emerge in relation to {user_query}.", "medium"),
    ("Scarcity and Abundance Mindset", "Given {available_resources}, introspect on the mindset—either scarcity-driven or abundance-oriented—that dominates when addressing {user_query}.", "long")
]

# Token limits based on complexity
token_limits = {
    "short": 128,
    "medium": 256,
    "long": 512
}

# Utility Functions

def format_response(agent_data):
    template = f"""
### System:
You are an AI Agent that employs various reasoning frameworks to provide insightful responses.

### Human:
{recorded_data["user_input"]}

### Agent {agent_data['name']} ({agent_data['prompt']}):
{agent_data['response']}

Sentiment: {agent_data['sentiment']}
Keywords: {', '.join(agent_data['keywords'])}
--------------------------------------------
"""
    return template

if len(tokens) > max_tokens_for_context:
    output += " ".join(tokens[-max_tokens_for_context:])
else:
    output += " ".join(tokens)

def save_to_json(data, filename, mode="append"):
    existing_data = []
  
    if os.path.exists(filename) and mode == "append":
        with open(filename, "r") as json_file:
            try:
                existing_data = json.load(json_file)
                if not isinstance(existing_data, list):
                    # Convert to list if the existing data is not a list
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                # Handle corrupted JSON file or any other reading error
                existing_data = []
 
    if mode == "append":
        existing_data.append(data)
        final_data = existing_data
    else:
        final_data = data
    with open(filename, "w") as json_file:
        json.dump(final_data, json_file, indent=4)


def save_summary_data(summary, user_query, reasoning_used=None, filename="summary_data.json"):
    summary_keywords = extract_keywords(summary)
    summary_sentiment = analyze_sentiment(summary)
    summary_data = {
        "Summary Agent Response": {
            
            "Initial Question": user_query,
            "Sentiment Analysis": {
                "Overall Sentiment": summary_sentiment,
            },
            "Main Themes or Topics": summary_keywords,
            "Reasoning Frameworks Used": reasoning_used if reasoning_used else "Unknown",
        }
    }
    
    # Save the constructed data to the specified JSON file
    save_to_json(summary_data, filename)

def save_data(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def truncate_preserving_sentences(text, max_tokens):
    sentences = sent_tokenize(text)
    truncated_text = ''
    total_tokens = 0
    
    for sentence in reversed(sentences):
        sentence_tokens = len(sentence.split())
        if total_tokens + sentence_tokens <= max_tokens:
            truncated_text = sentence + " " + truncated_text
            total_tokens += sentence_tokens
        else:
            break
    
    return truncated_text.strip()

def chunk_text(text, max_tokens=CHUNK_LIMIT):
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        if len(current_chunk) + len(word.split()) <= max_tokens:
            current_chunk.append(word)
        else:
            chunk = ' '.join(current_chunk)
            chunks.append(truncate_preserving_sentences(chunk, max_tokens))
            current_chunk = [word]
    if current_chunk:
        chunk = ' '.join(current_chunk)
        chunks.append(truncate_preserving_sentences(chunk, max_tokens))
    
    return chunks

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def extract_keywords(text, limit=KEYWORD_LIMIT):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases()
    
    # Filter out multi-word phrases and avoid duplicates
    single_words = list(set(word for phrase in ranked_phrases for word in phrase.split() if ' ' not in phrase))
    
    return single_words[:limit]



def generate_embedding(text):
    embedder = Embed4All()
    return embedder.embed(text)

# Load previous summaries if they exist
if os.path.exists("conversation_summaries.json"):
    with open("conversation_summaries.json", "r") as file:
        conversation_summaries = json.load(file)



# Memory Management
class Memory:
    def __init__(self):
        self.embeddings = []
        self.texts = []

    def add_memory(self, text, embedding):
        self.embeddings.append(embedding)
        self.texts.append(text)

    def search_memory(self, keyword_embedding, threshold=0.8, ignore_last=True):
        for i, stored_embedding in enumerate(self.embeddings):
            if ignore_last and i == len(self.embeddings) - 1:
                continue
            similarity = np.dot(stored_embedding, keyword_embedding) / (np.linalg.norm(stored_embedding) * np.linalg.norm(keyword_embedding))
            if similarity > threshold:
                return self.texts[i]
        return None

# Initialize the memory object to store the embeddings and texts
memory = Memory()


def search_similar_words(keyword, num_results=1):
    global memory  # This line indicates that we are using the global memory object

    # Generate embedding for the provided keyword
    keyword_embedding = generate_embedding(keyword)

    # Search the memory for similar words or phrases
    similarities = []  # To store tuples of (similarity, word)

    for i, stored_embedding in enumerate(memory.embeddings):
        # Using sklearn's cosine_similarity for accuracy
        similarity = cosine_similarity([stored_embedding], [keyword_embedding])[0][0]
        similarities.append((similarity, memory.texts[i]))

    # Sort the similarities in descending order and take the top num_results
    sorted_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)[:num_results]

    # Format and present the results to the user
    if sorted_similarities:
        print(f"Top {num_results} words or phrases similar to '{keyword}':")
        for sim, word in sorted_similarities:
            print(f"- {word} (Similarity: {sim:.4f})")
    else:
        print(f"No words or phrases found similar to '{keyword}'.")



# Chat Agent
class ChatAgent:
    def __init__(self):
        # Initialize the response history
        self.response_history = []
        self.model = GPT4All(model_name=MODEL_NAME)
        self.embedder = Embed4All()
        self.conversation_history = []

    def generate_embedding(self, text):
        return self.embedder.embed(text)
    
    def get_salient_points(self, text):
        keywords = extract_keywords(text)
        return list(set(keywords))
    
    def generate_summary(self, initial_question, agent_responses, sentiment_analysis, main_themes, reasoning_frameworks):
        summary = {}
        summary['Initial Question'] = initial_question
        summary['Agent Responses'] = agent_responses
        summary['Sentiment Analysis'] = sentiment_analysis
        summary['Main Themes or Topics'] = main_themes
        summary['Reasoning Frameworks Used'] = reasoning_frameworks
        summary['Overall Sentiment'] = sentiment_analysis
        complete_agent_responses = "\n\n".join([f"Agent {item['name']}:\n{item['response']}" for item in self.response_history if 'response' in item])
        summary['Complete Agent Responses'] = complete_agent_responses
        detailed_sentiments = [{"Agent": item['name'], "Sentiment": analyze_sentiment(item['response'])} for item in self.response_history if 'response' in item]
        summary['Detailed Sentiment Analysis'] = detailed_sentiments
        formattedsumprompt = (f"Question: {initial_question}\n"
                            f"Agent Responses: {complete_agent_responses}\n"
                            f"Topics: {', '.join(main_themes)}\n"
                            f"Reasoning Frameworks Used: {', '.join(reasoning_frameworks)}\n"
                            f"Detailed Sentiment Analysis: {detailed_sentiments}\n"
                            f"System: Based on the provided details, generate a concise and coherent summary.")

        #print(formattedsumprompt) # Debugging
        # Generate the summary using GPT-4All
        with self.model.chat_session():
            generated_summary = self.model.generate(prompt=formattedsumprompt, temp=0.6, top_p=TOP_P, top_k=TOP_K, max_tokens=token_limits["long"])
        
        # Combine the generated summary with the existing summary components
        summarized_response = f"{generated_summary}\n\n"
        return summarized_response

    def process_input(self, user_input, sequence, available_resources):
        responses = []
        output = user_input
        input_sentiment = analyze_sentiment(user_input)
        input_keywords = extract_keywords(user_input)
        # Recording user input and agent responses in response_history
        self.response_history.append({
            "user_input": user_input,
            "responses": responses,
            "reasoning_frameworks": [prompts[ord(agent) - ord('a')][0] for agent in sequence]
        })
        for keyword in input_keywords:
            keyword_embedding = generate_embedding(keyword)
            matched_text = memory.search_memory(keyword_embedding, ignore_last=True)
            if matched_text:
                output += " " + matched_text
        max_context_responses = 1
        output += " ".join(conversation_history[-max_context_responses:])
        for agent in sequence:
            index = ord(agent) - ord('a')
            if index < len(prompts):
                prompt_text, prompt_query, complexity = prompts[index]
                token_limit = token_limits[complexity]
                # Verbose: "Given the user's query: '{user_input}', you are expected to provide a solution or insight. Your response should be deeply rooted in the context of the question and should not stray from the main topic. Therefore, "
                # OldVer: "User has asked: '{user_input}'. Based on this, "
                # OldVer: "It is imperative to always keep the user's primary question: '{user_input}' at the forefront of your considerations while formulating your response."
                # Existing: "Always consider the user's main question: '{user_input}' in your response."
                query = f"User has asked: '{user_input}'. So, " + prompt_query.format(user_query=output, available_resources=available_resources) + f". Remember the main question: '{user_input}' in your answer."

                aligned_prompt = f"{prompt_text}. Framework: {query} {GUIDANCE_PROMPT}"
                response = self.generate_response(query, token_limit)
                # Update the reasoning_used list with the current reasoning framework
                reasoning_used.append(prompt_text)
                responses.append(response)
                conversation_history.append(response)

                output += f" Agent {agent.upper()}'s suggestion: {response}"
        memory.add_memory(user_input, generate_embedding(user_input))
        for response in responses:
            memory.add_memory(response, self.generate_embedding(response))
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

def print_availableframes():
    print("\nHere are the available reasoning frameworks and their associated letters:\n")
    for i, (name, _, _) in enumerate(prompts):
        letter = chr(i + ord('a'))
        print(f"{letter}. {name}")

recorded_data = {
    "user_input": None,
    "available_resources": None,
    "agents": []
}
def save_summary_to_json(conversation_summaries, filename="conversation_summaries.json"):
    with open(filename, "w") as json_file:
        json.dump(conversation_summaries, json_file, indent=4)

def main():
    global recorded_data, memory
    agent = ChatAgent()
    
    memory = Memory()

    try:
        memory_data = load_data('memory.pkl')
        memory.embeddings = memory_data['embeddings']
        memory.texts = memory_data['texts']
    except:
        pass

    while True:  # Start the conversation loop
        user_input = input("\nEnter your query, type 'search' to find similar words, or 'exit' to end: ")
        # Handle 'search' command
        if user_input.strip().lower() == 'search':
            keyword = input("Enter a keyword to search for similar words or phrases: ")
            search_similar_words(keyword)
            continue  # Skip the rest of the loop and prompt the user again
        
        if user_input.strip().lower() == 'exit':
            break  # End the loop if the user types 'exit'
        
        recorded_data["user_input"] = user_input

        # Prompt the user for their available resources
        available_resources = input("Please specify your available resources: ")
        recorded_data["available_resources"] = available_resources
        print_availableframes()
        sequence = input("Enter the sequence of agents (e.g., 'aabc'): ")

        # Update the process_input function call to pass the available resources
        responses = agent.process_input(user_input, sequence, available_resources)
        
        # List to store conversation summaries
        conversation_summaries = []

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
            print(format_response(agent_data))

        # Calculate overall sentiment for the conversation
        overall_sentiment = "Positive" if sum(agent_data["sentiment"] for agent_data in recorded_data["agents"]) > 0 else "Negative" if sum(agent_data["sentiment"] for agent_data in recorded_data["agents"]) < 0 else "Neutral"

        # Extract main themes from the agents' responses
        main_themes = []
        for agent_data in recorded_data["agents"]:
            main_themes.extend(agent_data["keywords"])
        main_themes = list(set(main_themes))  # Remove duplicates

        # Extract reasoning frameworks used from the agents' prompts
        reasoning_frameworks = [agent_data["prompt"] for agent_data in recorded_data["agents"]]

        # Generate and print the summary
        summarized_response = agent.generate_summary(
            recorded_data["user_input"],
            [agent_data["response"] for agent_data in recorded_data["agents"]],
            overall_sentiment,
            main_themes,
            reasoning_frameworks
        )
        print("\n\n--------------------------------")
        print("Summary of the Conversation:")
        print("--------------------------------")
        print(summarized_response)

        # Append the summarized response to the list
        conversation_summaries.append(summarized_response)

        # Save the conversation summaries to a JSON file
        save_to_json(conversation_summaries, "conversation_summaries.json")

        # Save the recorded data and summary
        save_to_json(recorded_data, "output_data.json")
        save_summary_data(summarized_response, recorded_data["user_input"], reasoning_used=reasoning_used)

        # Clear recorded_data for next iteration
        recorded_data = {
            "user_input": None,
            "available_resources": None,
            "agents": []
        }
if __name__ == "__main__":
    main()
