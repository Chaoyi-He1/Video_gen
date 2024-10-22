from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key='')
import os

# Set your OpenAI API key
def summarize_text(content):
    try:
        response = client.chat.completions.create(model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text into one simple sentence, like: A girl is dancing. or like: Nightfall in a metropolis."},
            {"role": "user", "content": content}
        ],
        max_tokens=50)
        summary = response.choices[0].message.content
        return summary.strip()
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None

def process_files(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in tqdm(os.listdir(input_directory)):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)

            if os.path.exists(output_path):
                continue
            
            with open(input_path, 'r', encoding='utf-8') as file:
                content = file.read()

            summary = summarize_text(content)
            assert summary is not None

            if summary:
                with open(output_path, 'w', encoding='utf-8') as file:
                    file.write(summary)

            # print(f"Processed {filename}, summary saved to {output_path}")

if __name__ == "__main__":
    input_directory = '/data/chaoyi_he/Video_gen/data/train_data/labels'  # Replace with your input directory path
    output_directory = '/data/chaoyi_he/Video_gen/data/train_data/summaries'  # Replace with your desired output directory path
    process_files(input_directory, output_directory)
