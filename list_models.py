import google.generativeai as genai

genai.configure(api_key="AIzaSyB5H6b-gUMM85dLzPatSEp004EDlyQPA74")

models = genai.list_models()

for model in models:
    print(model.name, "->", model.supported_generation_methods)
