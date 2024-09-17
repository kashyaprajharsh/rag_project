import pandas as pd
import numpy as np
from renumics import spotlight
from renumics.spotlight import layout
from langchain_openai import OpenAIEmbeddings
import colorsys

def generate_distinct_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    return ['#%02x%02x%02x' % tuple(int(x*255) for x in colorsys.hsv_to_rgb(*hsv)) for hsv in HSV_tuples]

def visualize_all_results(question, results, vectorstore):
    embeddings_model = OpenAIEmbeddings()
    
    response = vectorstore.get(include=["metadatas", "documents", "embeddings"])
    df = pd.DataFrame(
        {
            "id": response["ids"],
            "source": [metadata.get("source") for metadata in response["metadatas"]],
            "page": [metadata.get("page", -1) for metadata in response["metadatas"]],
            "document": response["documents"],
            "embedding": response["embeddings"],
            "type": "document",
            "relevance": 0,
            "method": "None"
        }
    )
    
    # Add question to the dataframe
    question_row = pd.DataFrame(
        {
            "id": ["question"],
            "document": [question],
            "embedding": [embeddings_model.embed_query(question)],
            "type": "question",
            "relevance": 1,
            "method": "None"
        }
    )
    df = pd.concat([question_row, df], ignore_index=True)
    
    # Generate distinct colors for each RAG method
    colors = generate_distinct_colors(len(results))
    color_map = {method: color for method, color in zip(results.keys(), colors)}
    
    # Add answers and context from all RAG methods to the dataframe
    for method, result in results.items():
        answer = result['answer']
        answer_row = pd.DataFrame(
            {
                "id": [f"answer_{method}"],
                "document": [answer],
                "embedding": [embeddings_model.embed_query(answer)],
                "type": "answer",
                "relevance": 1,
                "method": method
            }
        )
        df = pd.concat([df, answer_row], ignore_index=True)
        
        # Add context documents
        for i, doc in enumerate(result['context']):
            context_row = pd.DataFrame(
                {
                    "id": [f"context_{method}_{i}"],
                    "document": [doc.page_content],
                    "embedding": [embeddings_model.embed_query(doc.page_content)],
                    "type": "context",
                    "relevance": 1 - (i * 0.1),  # Decrease relevance for each subsequent context
                    "method": method
                }
            )
            df = pd.concat([df, context_row], ignore_index=True)

    # Calculate distances
    question_embedding = embeddings_model.embed_query(question)
    df["dist"] = df.apply(
        lambda row: np.linalg.norm(np.array(row["embedding"]) - question_embedding),
        axis=1,
    )

    # Create a color column
    df['color'] = df['method'].map(lambda x: color_map.get(x, '#CCCCCC'))

    # Create a Spotlight visualization
    spotlight.show(
        df,
        dtype={"embedding": spotlight.Embedding},
        layout=layout.layout(
            [
                [layout.table()],
                [
                    layout.similaritymap(
                        columns=["embedding"],
                        color_by_column="method",
                        size_by_column="relevance",
                    )
                ],
                [
                    layout.histogram(
                        column="type", stack_by_column="method"
                    )
                ],
            ],
            layout.widgets.Inspector(),
        ),
    )

    return df  # Return the DataFrame for further use if needed
