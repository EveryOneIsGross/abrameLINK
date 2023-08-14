from gpt4all import GPT4All, Embed4All
from textblob import TextBlob
from rake_nltk import Rake
import pickle
import numpy as np
import json
import os

# Global Configuration
MODEL_NAME = 'C://AI_MODELS//vicuna-7b-cot-superhot-8k.ggmlv3.q2_K.bin'
GUIDANCE_PROMPT = ("As a reasoning framework answer: {user_query}. "
                   "List only a. problem b. solution c. action, consider current context, "
                   "use solutions and methods according to suggested logic.")
TEMP = 0.6
TOP_P = 0.8 # TOP_P is the cumulative probability of the most likely tokens to sample from it ranges from 0 to 1.0
TOP_K = 40 # TOP_K is the number of the most likely tokens to sample from. It ranges from 0 to 50257 for GPT-2.
CHUNK_LIMIT = 256
KEYWORD_LIMIT = 5

# Global Variables
conversation_history = []
max_tokens_for_context = 100
tokens = " ".join(conversation_history).split()
output = ""
reasoning_used = []

def format_response(framework_data):
    template = f"""
    Agent Name: {framework_data['name']}
    Reasoning Framework: {framework_data['prompt']}
    --------------------------------------------
    Response: 
    {framework_data['response']}
    
    Sentiment: {framework_data['sentiment']}
    Keywords: {', '.join(framework_data['keywords'])}
    --------------------------------------------
    """
    return template

if len(tokens) > max_tokens_for_context:
    output += " ".join(tokens[-max_tokens_for_context:])
else:
    output += " ".join(tokens)
# Utility Functions

def save_framework_data(data, filename="output_data.json"):
    if os.path.exists(filename):
        with open(filename, "r") as json_file:
            existing_data = json.load(json_file)
        existing_data.append(data)
        final_data = existing_data
    else:
        final_data = [data]
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
                # Add more detailed breakdown here if available from sentiment analysis function
            },
            "Main Themes or Topics": summary_keywords,
            "Reasoning Frameworks Used": reasoning_used if reasoning_used else "Unknown",
        }
    }

    existing_data = []
    
    # Check if the file already exists and read its contents
    if os.path.exists(filename):
        with open(filename, "r") as json_file:
            try:
                existing_data = json.load(json_file)
                if not isinstance(existing_data, list):
                    # Convert to list if the existing data is not a list
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                # Handle corrupted JSON file or any other reading error
                existing_data = []
    
    # Append new summary data
    existing_data.append(summary_data)

    # Save the updated data
    with open(filename, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)

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
    chunks.append(' '.join(current_chunk))
    return chunks

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def extract_keywords(text, limit=KEYWORD_LIMIT):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases()
    return ranked_phrases[:limit]

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

    def search_memory(self, keyword_embedding, threshold=0.8, ignore_last=True):
        for i, stored_embedding in enumerate(self.embeddings):
            if ignore_last and i == len(self.embeddings) - 1:
                continue
            similarity = np.dot(stored_embedding, keyword_embedding) / (np.linalg.norm(stored_embedding) * np.linalg.norm(keyword_embedding))
            if similarity > threshold:
                return self.texts[i]
        return None

memory = Memory()
conversation_history = []
try:
    memory_data = load_data('memory.pkl')
    memory.embeddings = memory_data['embeddings']
    memory.texts = memory_data['texts']
except:
    pass


# Chat Agent
class ChatAgent:
    def __init__(self):
        # Initialize the response history
        self.response_history = []
        self.model = GPT4All(model_name=MODEL_NAME)
        self.embedder = Embed4All()

    def generate_embedding(self, text):
        return self.embedder.embed(text)
    
    def get_salient_points(self, text):
        keywords = extract_keywords(text)
        return list(set(keywords))
    
    def generate_summary(self, initial_question, agent_responses, sentiment_analysis, main_themes, reasoning_frameworks):
        summary = {}

        # 1. User's Initial Question:
        summary['Initial Question'] = initial_question

        # 1.5. Agent Responses:
        summary['Agent Responses'] = agent_responses

        # 2. Sentiment Analysis:
        summary['Sentiment Analysis'] = sentiment_analysis

        # 3. Main Themes or Topics:
        summary['Main Themes or Topics'] = main_themes

        # 4. Reasoning Frameworks Used:
        summary['Reasoning Frameworks Used'] = reasoning_frameworks

        # 5. Overall Sentiment:
        summary['Overall Sentiment'] = sentiment_analysis

        # 6. Complete Agent Responses:
        complete_agent_responses = "\n\n".join([f"Agent {item['name']}:\n{item['response']}" for item in self.response_history if 'response' in item])
        summary['Complete Agent Responses'] = complete_agent_responses

        # 7. Detailed Sentiment Analysis for Each Agent:
        detailed_sentiments = [{"Agent": item['name'], "Sentiment": analyze_sentiment(item['response'])} for item in self.response_history if 'response' in item]
        summary['Detailed Sentiment Analysis'] = detailed_sentiments

        summarized_response = ""
        for key, value in summary.items():
            summarized_response += f"{key}: {value}\n\n"
        # Use GPT-4All to generate a summary of all the information
        formattedsumprompt = f"Generate a summary:\n\n" \
                    f"Initial Question: {initial_question}\n\n" \
                    f"Agent Responses:\n{agent_responses}\n\n" \
                    f"Sentiment Analysis:\n{sentiment_analysis}\n\n" \
                    f"Main Themes or Topics:\n{main_themes}\n\n" \
                    f"Reasoning Frameworks Used:\n{reasoning_frameworks}\n\n" \
                    f"Overall Sentiment:\n{sentiment_analysis}\n\n" \
                    f"Complete Agent Responses:\n{complete_agent_responses}\n\n" \
                    f"Detailed Sentiment Analysis for Each Agent:\n{detailed_sentiments}\n\n"

        with self.model.chat_session():
            # Generate the summary using GPT-4All
            generated_summary = self.model.generate(prompt=formattedsumprompt, temp=0.2, top_p=TOP_P, top_k=TOP_K, max_tokens=1000)

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
                query = prompt_query.format(user_query=output, available_resources=available_resources)
                aligned_prompt = f"Framework: {prompt_text}. {query} {GUIDANCE_PROMPT}"
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

# Define the prompts and their associated complexities
prompts = [
    ("Second-Order Thinking", "Given {available_resources}, first identify the immediate outcomes of {user_query}. Then, dive deeper to discern the secondary, less obvious consequences.", "medium"),
    ("Pareto Principle (80/20 Rule)", "Employing {available_resources}, concentrate on the crucial '20%' of factors that will yield '80%' of the outcomes for {user_query}.", "short"),
    ("First Principle Thinking", "Utilizing {available_resources}, break down {user_query} to its foundational principles. Discern between what's factual and what's assumed.", "medium"),
    ("Regret Minimization Framework", "With {available_resources} in mind, project long-term implications of {user_query}, especially focusing on potential regrets and emotional repercussions.", "medium"),
    ("Opportunity Costs", "Given {available_resources}, contemplate the alternatives you'll forego by opting for {user_query}.", "medium"),
    ("The Sunk Cost Fallacy", "Utilizing {available_resources}, analyze {user_query} by emphasizing its prospective benefits over the costs already incurred.", "medium"),
    ("Occam's Razor", "With {available_resources}, distill {user_query} to its simplest form or explanation, eliminating unnecessary complexities.", "short"),
    ("Systems Thinking", "Using {available_resources}, map out where {user_query} slots into broader systems or networks.", "medium"),
    ("Inversion", "Keeping {available_resources} in mind, reverse-engineer {user_query}, highlighting potential challenges and pitfalls.", "medium"),
    ("Leverage", "Employing {available_resources}, explore how you can use leverage to maximize the outcomes of {user_query}.", "short"),
    ("Circle of Competence", "Given {available_resources}, verify that {user_query} aligns well with your domain of expertise and knowledge.", "medium"),
    ("Law of Diminishing Returns", "With {available_resources}, pinpoint areas in {user_query} where additional efforts might produce diminishing outcomes.", "medium"),
    ("Niches", "Using {available_resources}, categorize how {user_query} can be tailored or adapted to cater to specific niches or subgroups.", "short"),
    ("Margin of Safety", "With {available_resources}, strategize to incorporate a buffer or safety net around the outcomes of {user_query}.", "medium"),
    ("Hanlon's Razor", "Given {available_resources}, approach {user_query} with neutrality, avoiding assumptions of malevolence or negative intent.", "short"),
    ("Randomness", "Considering {available_resources}, factor in elements of unpredictability and chance that could influence {user_query}.", "medium"),
    ("Critical Mass", "Utilizing {available_resources}, determine the tipping point at which {user_query} becomes self-sustaining or gains momentum.", "short"),
    ("The Halo Effect", "Employing {available_resources}, scrutinize how initial perceptions or biases might color subsequent interpretations of {user_query}.", "medium"),
    ("Feedback Loops", "With {available_resources}, trace the cyclic patterns or recurring themes that emerge in relation to {user_query}.", "medium"),
    ("Scarcity and Abundance Mindset", "Given {available_resources}, introspect on the mindset—either scarcity-driven or abundance-oriented—that dominates when addressing {user_query}.", "long")
]

# Token limits based on complexity
token_limits = {
    "short": 200,
    "medium": 400,
    "long": 600
}

def print_availableframes():
    print("\nHere are the available reasoning frameworks and their associated letters:\n")
    for i, (name, _, _) in enumerate(prompts):
        letter = chr(i + ord('a'))
        print(f"{letter}. {name}")

framework_data = {
    "user_input": None,
    "available_resources": None,
    "agents": []
}
def save_data_to_json(data, filename):
    existing_data = []
    
    # Check if the file already exists and read its contents
    if os.path.exists(filename):
        with open(filename, "r") as json_file:
            try:
                existing_data = json.load(json_file)
                if not isinstance(existing_data, list):
                    # Convert to list if the existing data is not a list
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                # Handle corrupted JSON file or any other reading error
                existing_data = []
    
    # Append new data
    existing_data.append(data)

    # Save the updated data
    with open(filename, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)

def main():
    global framework_data
    agent = ChatAgent()

    user_input = input("Enter your query: ")
    framework_data["user_input"] = user_input

    # Prompt the user for their available resources
    available_resources = input("Please specify your available resources: ")
    framework_data["available_resources"] = available_resources
    print_availableframes()
    sequence = input("Enter the sequence of agents (e.g., 'aabc'): ")

    # Update the process_input function call to pass the available resources
    responses = agent.process_input(user_input, sequence, available_resources)

    # List to store conversation summaries
    conversation_summaries = []

    # Loop through the responses to record and print them
    for i, response in enumerate(responses, 1):
        framework_data = {
            "name": sequence[i-1].upper(),
            "prompt": prompts[ord(sequence[i-1]) - ord('a')][0],
            "response": response,
            "sentiment": analyze_sentiment(response),
            "keywords": extract_keywords(response)
        }
        framework_data["agents"].append(framework_data)
        print(format_response(framework_data))

    # Calculate overall sentiment for the conversation
    overall_sentiment = "Positive" if sum(framework_data["sentiment"] for framework_data in framework_data["agents"]) > 0 else "Negative" if sum(framework_data["sentiment"] for framework_data in framework_data["agents"]) < 0 else "Neutral"

    # Extract main themes from the agents' responses
    main_themes = []
    for framework_data in framework_data["agents"]:
        main_themes.extend(framework_data["keywords"])
    main_themes = list(set(main_themes))  # Remove duplicates

    # Extract reasoning frameworks used from the agents' prompts
    reasoning_frameworks = [framework_data["prompt"] for framework_data in framework_data["agents"]]

    # Generate and print the summary
    summarized_response = agent.generate_summary(
        framework_data["user_input"],
        [framework_data["response"] for framework_data in framework_data["agents"]],
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

    # Save the conversation summaries to the summary_data.json file
    save_data_to_json(conversation_summaries, 'summary_data.json')

    # Save the recorded data and summary
    save_data_to_json(framework_data, 'framework_data.json')
    save_summary_data(summarized_response, framework_data["user_input"], reasoning_used=reasoning_used, filename='summary_data.json')

if __name__ == "__main__":
    main()
