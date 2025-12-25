import re
import os
import base64
from PIL import Image
from io import BytesIO
from google import genai
from langchain import hub
from datetime import datetime
from google.genai import types
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.5,
    max_tokens=32768, 
)

def generate_image(prompt: str):
    
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
            )
        )

        quote_text = ""
        img_path = None

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                quote_text = part.text.strip()
                print("Quote:", quote_text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO((part.inline_data.data)))

                # Save the image to a file  
                os.makedirs("images", exist_ok=True)

                # Format filename: images/YYYY-MM-DD-HHMM_quote.png
                timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
                snippet = re.sub(r'[^a-zA-Z0-9]', '_', timestamp)[:20]
                img_path = f"images/{timestamp}_{snippet}.png"
                image.save(img_path)
                image.show()
                return img_path 
            
        return img_path or "No image returned"

    except Exception as e: 
        return f"Image generation failed: {str(e)}"

def generate_hashtags(content: str) -> str:
    prompt = "Based on below content create 4 to 6 best hashtags which cover wide range of public \n"
    query = prompt + content
    response = model.invoke(query)
    return response.content

search = DuckDuckGoSearchRun() 
wikipedia = WikipediaAPIWrapper()

tools = [
    Tool(name="WebSearch", func=search.run, description="Search the web for research"),
    Tool(name="Wikipedia", func=wikipedia.run, description="Get factual information"),
    Tool(name="ImageGeneartor", func=generate_image, description="Generate images based on descriptive prompt"),
    Tool(name="HashtagGenerator", func=generate_hashtags, description="Generate hashtags based on selected motivational quote")
]

# Official LangChain ReAct prompt
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_prompt = """ 
Find one powerful motivational quote.

Then generate a high-quality image that includes the quote as the main text, styled with an inspiring visual background.

The background should be detailed and emotionally uplifting — such as a sunrise over mountains, a peaceful forest, a person climbing a cliff, or a futuristic glowing city. Use soft lighting, cinematic mood, and motivational aesthetic.

Place the quote text clearly in the center or upper half of the image using clean, readable font.
Use a color scheme that contrasts well with the background, ensuring the text is legible.
Ensure the background is slightly blurred or low-opacity to keep the focus on the text.

After generate image, make hashtags according to motivation quotes in json format which has hashtags key and list of hashtags.
"""

# for i in range(5):
response = agent_executor.invoke({"input":agent_prompt})
print(response)
 

 