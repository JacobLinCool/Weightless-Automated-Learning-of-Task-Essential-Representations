from google import genai
from google.genai import types

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

# with open('path/to/small-sample.mp3', 'rb') as f:
#     audio_bytes = f.read()

# client = genai.Client()
# response = client.models.generate_content(
#   model='gemini-3-flash-preview',
#   contents=[
#     'Describe this audio clip',
#     types.Part.from_bytes(
#       data=audio_bytes,
#       mime_type='audio/mp3',
#     )
#   ]
# )

# print(response.text)
