from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    # Entry point for running the RAG pipeline interactively
    print("Hello RAG")
    # Example question for the pipeline; change as desired
    # Ingested documents: https://lilianweng.github.io/posts/2023-06-23-agent/
    # https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
    # https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/
    # See ingestion.py for more details

    # So that's the context it has; for anything else, it will search the web
    print(app.invoke(input={"question": "What is agent memory?"}))
