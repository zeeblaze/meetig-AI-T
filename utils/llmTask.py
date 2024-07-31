from openai import OpenAI


system_message = '''
Your name is Meetig-AI-T, you are a meeting Assistant AI that help in summarrizing meeting outcomes,
make your responses very short and concise, don't make up anything that is not in the transcript, and
always end your responses with "Your's Truly Meetig-AI-T"
'''


class summary():
    def __init__(self):
        pass

    def summarize(self, transcript, config, api_key):
        try:
            self.transcript = f"""you are given a transcrit "{transcript}", summarize with bullet points.
            """
            if config == "ollama_mistral":
                chatGPT_client = OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key=api_key
                )

                ollama_response = chatGPT_client.chat.completions.create(
                    model="mistral",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": self.transcript}
                    ]
                )

                return ollama_response.choices[0].message.content
            
            else:
                pass
            # If config does not match expected values, log the issue and return a default message
            print(f"Unexpected config value: {config}")
            return "Error: Unsupported summary config"
        except Exception as e:
            print(e)
            return e

if __name__ == "__main__":
    summarizer = summary()
    text = "what are you?"
    summaries = summarizer.summarize(text, "ollama_mistral", "anything")
    print(summaries)
